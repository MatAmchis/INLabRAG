"""
Post-processing pipeline for filling missing fields
"""
import csv
import json
from pathlib import Path
from typing import Optional
import pandas as pd

from .config import PRIORITY_FIELDS
from .llm_client import LLMClient
from .agents import AgentPipeline
from .utils import is_na, get_paper_id

from config.postprocessing import PostprocessingConfig, set_seed
set_seed(PostprocessingConfig.SEED)

TRIM_AFTER_COLUMN = "Major recommendations of the paper based on its findings"


def _trim_dataset(input_csv: Path, output_csv: Path) -> None:
    try:
        df = pd.read_csv(input_csv, keep_default_na=False)
        if TRIM_AFTER_COLUMN in df.columns:
            col_index = df.columns.get_loc(TRIM_AFTER_COLUMN)
            df_trimmed = df.iloc[:, :col_index + 1]
            df_trimmed.to_csv(output_csv, index=False)
            print(f"Trimmed CSV saved: {output_csv}")
        else:
            df.to_csv(output_csv, index=False)
    except Exception as e:
        print(f"Trim failed: {e}")


def run_pipeline(
    csv_path: str,
    outdir: str = "output/post_processing",
    model_name: str = None,
    api_key: Optional[str] = None,
    force_reprocess: bool = False
) -> pd.DataFrame:
    if model_name is None:
        model_name = PostprocessingConfig.POST_MODEL
    
    print(f"\n{'='*60}")
    print("POST-PROCESSING PIPELINE")
    print(f"{'='*60}")
    print(f"Input CSV: {csv_path}")
    print(f"Output dir: {outdir}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    df = pd.read_csv(csv_path, keep_default_na=False, na_filter=False)
    
    # Convert all columns to string type to ensure consistent handling
    for col in df.columns:
        def normalize_value(x):
            if pd.isna(x):
                return ''
            if x is None:
                return ''
            x_str = str(x)
            x_lower = x_str.lower().strip()
            if x_lower in ('nan', 'none', '<na>', '<none>', 'null', ''):
                return ''
            return x_str.strip()
        
        df[col] = df[col].apply(normalize_value)
    
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    
    # Count empty fields per priority field
    priority_set_debug = set(PRIORITY_FIELDS)
    print("\nEmpty field counts (priority fields):")
    for field in PRIORITY_FIELDS:
        if field in df.columns:
            empty_count = df[field].apply(is_na).sum()
            total_count = len(df)
            pct = 100*empty_count/total_count if total_count > 0 else 0
            print(f"  {field}: {empty_count}/{total_count} empty ({pct:.1f}%)")
    
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    llm_client = LLMClient(api_key=api_key, model_name=model_name)
    agent_pipeline = AgentPipeline(llm_client)
    
    log_files = {
        "p1": open(Path(outdir) / "p1.jsonl", "w", encoding="utf-8"),
        "p2": open(Path(outdir) / "p2.jsonl", "w", encoding="utf-8"),
        "checker": open(Path(outdir) / "checker.jsonl", "w", encoding="utf-8"),
        "p3": open(Path(outdir) / "p3.jsonl", "w", encoding="utf-8")
    }
    
    summary_path = Path(outdir) / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()
    summary_header_written = False
    
    updated_df = df.copy()
    priority_set = set(PRIORITY_FIELDS)
    
    total_processed = 0
    total_accepted = 0
    
    for row_idx, row in df.iterrows():
        paper_id = get_paper_id(row, row_idx)
        print(f"\nProcessing paper: {paper_id}")
        
        for field in df.columns:
            if field not in priority_set:
                continue
            
            # Get normalized value directly from dataframe to ensure consistency
            field_value = df.loc[row_idx, field]
            if not is_na(field_value):
                continue
            
            print(f"  Processing field: {field}")
            total_processed += 1
            
            status, results = agent_pipeline.process_field(field, row)
            
            log_files["p1"].write(json.dumps({
                "paper_id": paper_id,
                "field": field,
                "raw": results["p1_raw"]
            }) + "\n")
            
            log_files["p2"].write(json.dumps({
                "paper_id": paper_id,
                "field": field,
                "raw": results["p2_raw"]
            }) + "\n")
            
            log_files["checker"].write(json.dumps({
                "paper_id": paper_id,
                "field": field,
                "raw": results["checker_raw"]
            }) + "\n")
            
            log_files["p3"].write(json.dumps({
                "paper_id": paper_id,
                "field": field,
                "raw": results["p3_raw"]
            }) + "\n")
            
            if status == "ACCEPT":
                updated_df.at[row_idx, field] = results["final_value"]
                total_accepted += 1
                print(f"    ACCEPTED: {results['final_value'][:60]}...")
            else:
                updated_df.at[row_idx, field] = "N/A"
                if results["candidate_val"]:
                    suggested_col = f"{field}__SUGGESTED"
                    if suggested_col not in updated_df.columns:
                        updated_df[suggested_col] = ""
                    updated_df.at[row_idx, suggested_col] = results["candidate_val"]
                print(f"    REJECTED")
            
            def store_tracking(suffix: str, value: str):
                col = f"{field}__{suffix}"
                if col not in updated_df.columns:
                    updated_df[col] = ""
                updated_df.at[row_idx, col] = value or ""
            
            store_tracking("P1", results["p1_val"])
            store_tracking("P2", results["p2_val"] if results["p2_decision"] == "edit" else "")
            store_tracking("CHECKER", results["checker_decision"])
            store_tracking("P3", results["p3_val"] if status == "ACCEPT" else "")
            store_tracking("STATUS", status)
            
            summary_row = {
                "paper_id": paper_id,
                "row": row_idx,
                "field": field,
                "p1_val": results["p1_val"] or "",
                "p2_dec": results["p2_decision"],
                "p2_val": results["p2_val"] or "",
                "checker_dec": results["checker_decision"],
                "p3_dec": results["p3_decision"],
                "p3_val": results["p3_val"] or "",
                "status": status
            }
            
            with summary_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
                if not summary_header_written:
                    writer.writeheader()
                    summary_header_written = True
                writer.writerow(summary_row)
    
    for log_file in log_files.values():
        log_file.close()
    
    output_csv = Path(outdir) / "dataset_with_fills.csv"
    updated_df.to_csv(output_csv, index=False)
    
    # Trim columns after the Major recommendations
    trimmed_csv = Path(outdir) / "dataset_trimmed.csv"
    _trim_dataset(output_csv, trimmed_csv)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total fields processed: {total_processed}")
    print(f"Accepted: {total_accepted}")
    print(f"Rejected: {total_processed - total_accepted}")
    print(f"\nOutput files:")
    print(f"   {output_csv}")
    print(f"   {trimmed_csv}")
    print(f"   {summary_path}")
    print(f"{'='*60}\n")
    
    return updated_df


def run_pipeline_batch(
    csv_folder: str,
    outdir: str = "output/post_processing",
    model_name: str = None,
    api_key: Optional[str] = None,
    force_reprocess: bool = False
) -> pd.DataFrame:
    if model_name is None:
        model_name = PostprocessingConfig.POST_MODEL
    
    print(f"\n{'='*60}")
    print("BATCH POST-PROCESSING PIPELINE")
    print(f"{'='*60}")
    print(f"Input folder: {csv_folder}")
    print(f"Output dir: {outdir}")
    print(f"Model: {model_name}")
    print(f"Force reprocess: {force_reprocess}")
    print(f"{'='*60}\n")
    
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    csv_files = list(Path(csv_folder).glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_folder}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, keep_default_na=False, na_filter=False)
        for col in df.columns:
            def normalize_value(x):
                if pd.isna(x):
                    return ''
                if x is None:
                    return ''
                x_str = str(x)
                x_lower = x_str.lower().strip()
                if x_lower in ('nan', 'none', '<na>', '<none>', 'null', ''):
                    return ''
                return x_str.strip()
            
            df[col] = df[col].apply(normalize_value)
        print(f"  {csv_file.name}: {len(df)} rows")
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} papers x {len(combined_df.columns)} columns")
    
    print("\nEmpty field counts (priority fields):")
    for field in PRIORITY_FIELDS:
        if field in combined_df.columns:
            empty_count = combined_df[field].apply(is_na).sum()
            total_count = len(combined_df)
            pct = 100*empty_count/total_count if total_count > 0 else 0
            print(f"  {field}: {empty_count}/{total_count} empty ({pct:.1f}%)")
    
    processed_csv = Path(outdir) / "dataset_with_fills.csv"
    processed_papers = set()
    
    if processed_csv.exists() and not force_reprocess:
        existing_df = pd.read_csv(processed_csv, keep_default_na=False, na_filter=False)
        for col in existing_df.columns:
            def normalize_value(x):
                if pd.isna(x):
                    return ''
                if x is None:
                    return ''
                x_str = str(x)
                x_lower = x_str.lower().strip()
                if x_lower in ('nan', 'none', '<na>', '<none>', 'null', ''):
                    return ''
                return x_str.strip()
            
            existing_df[col] = existing_df[col].apply(normalize_value)
        for idx, row in existing_df.iterrows():
            paper_id = get_paper_id(row, idx)
            processed_papers.add(paper_id)
        print(f"Already processed: {len(processed_papers)} papers")
    
    if not force_reprocess and processed_papers:
        new_papers_mask = []
        for idx, row in combined_df.iterrows():
            paper_id = get_paper_id(row, idx)
            new_papers_mask.append(paper_id not in processed_papers)
        
        new_df = combined_df[new_papers_mask].reset_index(drop=True)
        print(f"New papers to process: {len(new_df)}")
        
        if len(new_df) == 0:
            print("All papers already processed! Use --force to reprocess.")
            return existing_df
    else:
        new_df = combined_df
        print(f"Papers to process: {len(new_df)}")
    
    llm_client = LLMClient(api_key=api_key, model_name=model_name)
    agent_pipeline = AgentPipeline(llm_client)
    
    log_files = {
        "p1": open(Path(outdir) / "p1.jsonl", "a", encoding="utf-8"),
        "p2": open(Path(outdir) / "p2.jsonl", "a", encoding="utf-8"),
        "checker": open(Path(outdir) / "checker.jsonl", "a", encoding="utf-8"),
        "p3": open(Path(outdir) / "p3.jsonl", "a", encoding="utf-8")
    }
    
    summary_path = Path(outdir) / "summary.csv"
    summary_header_written = summary_path.exists()
    
    updated_df = new_df.copy()
    priority_set = set(PRIORITY_FIELDS)
    
    total_processed = 0
    total_accepted = 0
    
    for row_idx, row in new_df.iterrows():
        paper_id = get_paper_id(row, row_idx)
        print(f"\nProcessing paper: {paper_id}")
        
        for field in new_df.columns:
            if field not in priority_set:
                continue
            field_value = new_df.loc[row_idx, field]
            if not is_na(field_value):
                continue
            
            print(f"  Processing field: {field}")
            total_processed += 1
            
            status, results = agent_pipeline.process_field(field, row)
            
            log_files["p1"].write(json.dumps({
                "paper_id": paper_id,
                "field": field,
                "raw": results["p1_raw"]
            }) + "\n")
            
            log_files["p2"].write(json.dumps({
                "paper_id": paper_id,
                "field": field,
                "raw": results["p2_raw"]
            }) + "\n")
            
            log_files["checker"].write(json.dumps({
                "paper_id": paper_id,
                "field": field,
                "raw": results["checker_raw"]
            }) + "\n")
            
            log_files["p3"].write(json.dumps({
                "paper_id": paper_id,
                "field": field,
                "raw": results["p3_raw"]
            }) + "\n")
            
            if status == "ACCEPT":
                updated_df.at[row_idx, field] = results["final_value"]
                total_accepted += 1
                print(f"    ACCEPTED: {results['final_value'][:60]}...")
            else:
                updated_df.at[row_idx, field] = "N/A"
                if results["candidate_val"]:
                    suggested_col = f"{field}__SUGGESTED"
                    if suggested_col not in updated_df.columns:
                        updated_df[suggested_col] = ""
                    updated_df.at[row_idx, suggested_col] = results["candidate_val"]
                print(f"    REJECTED")
            
            def store_tracking(suffix: str, value: str):
                col = f"{field}__{suffix}"
                if col not in updated_df.columns:
                    updated_df[col] = ""
                updated_df.at[row_idx, col] = value or ""
            
            store_tracking("P1", results["p1_val"])
            store_tracking("P2", results["p2_val"] if results["p2_decision"] == "edit" else "")
            store_tracking("CHECKER", results["checker_decision"])
            store_tracking("P3", results["p3_val"] if status == "ACCEPT" else "")
            store_tracking("STATUS", status)
            
            summary_row = {
                "paper_id": paper_id,
                "row": row_idx,
                "field": field,
                "p1_val": results["p1_val"] or "",
                "p2_dec": results["p2_decision"],
                "p2_val": results["p2_val"] or "",
                "checker_dec": results["checker_decision"],
                "p3_dec": results["p3_decision"],
                "p3_val": results["p3_val"] or "",
                "status": status
            }
            
            with summary_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
                if not summary_header_written:
                    writer.writeheader()
                    summary_header_written = True
                writer.writerow(summary_row)
    
    for log_file in log_files.values():
        log_file.close()
    
    if processed_csv.exists() and not force_reprocess:
        existing_df = pd.read_csv(processed_csv, keep_default_na=False, na_filter=False)
        final_df = pd.concat([existing_df, updated_df], ignore_index=True)
    else:
        final_df = updated_df
    
    final_df.to_csv(processed_csv, index=False)
    trimmed_csv = Path(outdir) / "dataset_trimmed.csv"
    _trim_dataset(processed_csv, trimmed_csv)
    
    print(f"\n{'='*60}")
    print("BATCH PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"New papers processed: {len(updated_df)}")
    print(f"Total fields processed: {total_processed}")
    print(f"Accepted: {total_accepted}")
    print(f"Rejected: {total_processed - total_accepted}")
    print(f"Total papers in dataset: {len(final_df)}")
    print(f"\nConsolidated output files:")
    print(f"   {processed_csv}")
    print(f"   {trimmed_csv}")
    print(f"   {summary_path}")
    print(f"{'='*60}\n")
    
    return final_df
