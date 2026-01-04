import json
import math
import re
from typing import Any, Dict
import pandas as pd


def is_na(value: Any) -> bool:
    if value is None:
        return True  
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except (ImportError, TypeError):
        pass
    if not isinstance(value, str):
        value = str(value)
    
    text = value.strip()
    if not text:
        return True
    
    text_lower = text.lower()
    if text_lower in ('nan', 'none', '<na>', '<none>', 'null'):
        return True
    norm_clean = ''.join(c for c in text_lower if c.isalnum() or c.isspace())
    norm_clean = ' '.join(norm_clean.split())
    if not norm_clean:
        return True
    canonical_missing = {
        "", "na", "n a", "nan", "none",
        "not reported", "not available", "not found", "not applicable",
        "not specified", "missing", "no data", "unknown"
    }
    
    if norm_clean in canonical_missing:
        return True
    
    # Fuzzy patterns (matches preprocessing's _is_missing)
    fuzzy_patterns = [
        "not reported",
        "notreported",
        "not available", 
        "notavailable",
        "not found",
        "notfound",
        "no data",
        "nodata"
    ]
    
    norm_no_spaces = norm_clean.replace(" ", "")
    for pattern in fuzzy_patterns:
        if pattern.replace(" ", "") in norm_no_spaces:
            return True
    
    return False


def row_to_block(row: pd.Series, limit: int = 12000) -> str:
    total = 0
    output = []
    
    for key, value in row.items():
        chunk = f"{key}: {value}"
        if total + len(chunk) > limit:
            output.append(chunk[:limit - total - 3] + "...")
            break
        output.append(chunk)
        total += len(chunk)
    
    return "\n".join(output)


def safe_json_parse(raw_response: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        # Try to extract JSON using regex
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    
    # Return empty dict if all parsing fails
    return {}


def get_paper_id(row: pd.Series, row_index: int) -> str:
    return row.get("paper_name") or row.get("Title") or f"row_{row_index}"
