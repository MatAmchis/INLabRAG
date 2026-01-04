import json
from typing import Dict, Any, Tuple

import pandas as pd

from .config import CONFIG, QUESTION_PROMPT_OVERRIDES
from .llm_client import LLMClient
from .utils import safe_json_parse, row_to_block


class AgentPipeline:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.config = CONFIG

    def run_proposer1(self, field: str, row: pd.Series) -> Tuple[Dict[str, Any], str]:
        instruction = QUESTION_PROMPT_OVERRIDES.get(field, "(none specified)")
        instruction_block = f"INSTRUCTIONS: {instruction}"
        row_block = row_to_block(row, self.config["row_context_char_limit"])

        prompt = self.config["prompts"]["p1_user_template"].format(
            field=field,
            instruction_block=instruction_block,
            row_block=row_block,
        )

        system_msg = self.config["prompts"]["p1_system"]
        raw_response = self.llm.call(system_msg, prompt)

        return safe_json_parse(raw_response), raw_response

    def run_proposer2(
        self, field: str, row: pd.Series, p1_json: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        row_block = row_to_block(row, self.config["row_context_char_limit"])

        prompt = self.config["prompts"]["p2_user_template"].format(
            field=field,
            row_block=row_block,
            p1_json=json.dumps(p1_json, ensure_ascii=False),
        )

        system_msg = self.config["prompts"]["p2_system"]
        raw_response = self.llm.call(system_msg, prompt)

        return safe_json_parse(raw_response), raw_response

    def run_instruction_checker(self, field: str, candidate_value: str | None) -> Tuple[Dict[str, Any], str]:
        instruction = QUESTION_PROMPT_OVERRIDES.get(field, "(none specified)")

        prompt = self.config["prompts"]["checker_user_template"].format(
            field=field,
            instruction=instruction,
            value=candidate_value or "",
        )

        system_msg = self.config["prompts"]["checker_system"]
        raw_response = self.llm.call(system_msg, prompt)

        return safe_json_parse(raw_response), raw_response

    def run_proposer3(
        self,
        field: str,
        row: pd.Series,
        p1_json: Dict[str, Any],
        p2_json: Dict[str, Any],
        checker_json: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        row_block = row_to_block(row, self.config["row_context_char_limit"])

        prompt = self.config["prompts"]["p3_user_template"].format(
            field=field,
            row_block=row_block,
            p1_json=json.dumps(p1_json, ensure_ascii=False),
            p2_json=json.dumps(p2_json, ensure_ascii=False),
            checker_json=json.dumps(checker_json, ensure_ascii=False),
        )

        system_msg = self.config["prompts"]["p3_system"]
        raw_response = self.llm.call(system_msg, prompt)

        return safe_json_parse(raw_response), raw_response

    def process_field(self, field: str, row: pd.Series) -> Tuple[str, Dict[str, Any]]:
        p1_json, p1_raw = self.run_proposer1(field, row)
        p1_val = p1_json.get("inferred_value")

        p2_json, p2_raw = self.run_proposer2(field, row, p1_json)
        p2_decision = p2_json.get("decision", "insufficient")
        p2_val = p2_json.get("edited_value")

        if p2_decision == "accept":
            candidate_val = p1_val
        elif p2_decision == "edit":
            candidate_val = p2_val
        else:
            candidate_val = None

        if isinstance(candidate_val, str) and candidate_val.strip().lower() == "null":
            candidate_val = None

        checker_json, checker_raw = self.run_instruction_checker(field, candidate_val)
        checker_decision = checker_json.get("decision", "noncompliant")

        if checker_decision == "noncompliant":
            candidate_val = None

        p3_json, p3_raw = self.run_proposer3(field, row, p1_json, p2_json, checker_json)
        p3_decision = p3_json.get("decision", "reject")
        p3_val = p3_json.get("final_value")

        if p3_decision in ("accept", "correct") and not p3_val:
            p3_val = candidate_val

        status = "REJECT"
        final_value = "N/A"

        if p3_decision in ("accept", "correct") and p3_val:
            final_value = str(p3_val).strip()
            status = "ACCEPT"

        results = {
            "p1_val": p1_val,
            "p1_raw": p1_raw,
            "p2_decision": p2_decision,
            "p2_val": p2_val,
            "p2_raw": p2_raw,
            "candidate_val": candidate_val,
            "checker_decision": checker_decision,
            "checker_raw": checker_raw,
            "p3_decision": p3_decision,
            "p3_val": p3_val,
            "p3_raw": p3_raw,
            "final_value": final_value,
            "status": status,
        }

        return status, results
