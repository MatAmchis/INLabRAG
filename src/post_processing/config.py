"""
Configuration for the 4-Agent Post-Processing Pipeline
"""
import textwrap
from typing import Dict, List

PRIORITY_FIELDS: List[str] = [
    "Study design (include length of study)",
    "Data collection method",
    "List of all program functions being integrated",
    "Community setting environment",
    "Rationale for integration (why)",
    "Specific activities undertaken to undergo process of integration",
    "Mechanisms for coordinating process/getting input from stakeholders",
    "List of outcomes of integration studied",
    "Facilitators of integration",
    "Barriers to integration / how they tried to overcome",
    "Uptake and coverage",
    "Sustainability",
    "Health sector staff perspectives on proposed integration before implementation"
]

QUESTION_PROMPT_OVERRIDES: Dict[str, str] = {
    "Study design (include length of study)": 
        "Identify the type of study design used in the paper and the time period over which the study was conducted, if available.",
    "Data collection method": 
        "Identify the specific tools, techniques, and protocols used to collect data in the study.",
    "List of all program functions being integrated": 
        "List all the health program functions that were integrated in the intervention.",
    "Community setting environment": 
        "Describe the physical, social, economic, or ecological context of the community where the integration occurred.",
    "Rationale for integration (why)": 
        "Describe the specific reasons and motivations stated in the paper for why the integration was pursued.",
    "Specific activities undertaken to undergo process of integration": 
        "Identify the specific operational and preparatory steps that were taken to support and carry out the integration of the intervention.",
    "Mechanisms for coordinating process/getting input from stakeholders": 
        "Identify any strategies, structures, or processes used to coordinate the integration effort and to gather input from stakeholders during planning or implementation. These mechanisms can be formal or informal.",
    "List of outcomes of integration studied": 
        "Identify the specific outcomes that the study aimed to measure or assess as a result of implementing the integration.",
    "Facilitators of integration": 
        "Identify the specific factors, conditions, or actions that helped make the integration successful or feasible.",
    "Barriers to integration / how they tried to overcome": 
        "Identify the challenges encountered during the integration process and the specific actions taken to address them.",
    "Uptake and coverage": 
        "Provide numeric adoption or coverage indicators with denominators.",
    "Sustainability": 
        "Describe the sustainability of the integrated intervention, specifically whether it continued after the study period or was considered feasible to maintain or what actions would allow it to be maintained for the future.",
    "Health sector staff perspectives on proposed integration before implementation": 
        "Identify any perspectives, opinions, or feedback from health sector staff regarding the proposed integration before it was implemented, including any concerns, expectations, or support expressed."
}

CONFIG = {
    "row_context_char_limit": 12000,
    "prompts": {
        "p1_system":
            "You are a research assistant with a specialization in integrated NTD program analyses. You can infer a concise answer for the "
            "target field by connecting, infering, and synthesizing from existing text from the row. Make sure inferences are not jumps in logic."
            "Justify any inference in one sentence.",
        
        "p1_user_template": textwrap.dedent("""
            FIELD: {field}

            {instruction_block}

            ALL ROW DATA (key -> value):
            {row_block}

            TASK:
              1. inferred_value. Null if impossible.
              2. reasoning (<=40 words) - briefly cite key column names.
            RETURN JSON:
            {{
              "field":"{field}",
              "inferred_value":"<string or null>",
              "reasoning":"<string>"
            }}
        """),
        
        "p2_system":
            "You are a researcher in the field of global health with a specializaation in ensuring that input you get is sufficient and clear. Decide if Proposer-1's answer is acceptable, "
            "needs minor edit, or is insufficient.",
        
        "p2_user_template": textwrap.dedent("""
            FIELD: {field}

            ALL ROW DATA:
            {row_block}

            PROPOSER1_JSON:
            {p1_json}

            TASK:
              - decision: accept | edit | insufficient
              - edited_value
              - justification_editor
            RETURN JSON:
            {{
              "decision":"<accept|edit|insufficient>",
              "edited_value":"<string or null>",
              "justification_editor":"<string>"
            }}
        """),
        
        "checker_system":
            "You are the instruction checker. Verify that the candidate answer fully "
            "conforms to the field specific instructions.  Reject if it violates the "
            "instructions.",
        
        "checker_user_template": textwrap.dedent("""
            FIELD: {field}

            OFFICIAL INSTRUCTIONS:
            {instruction}

            CANDIDATE_VALUE:
            "{value}"

            TASK:
              - decision: valid | noncompliant
              - justification_checker (<=30 words) - why noncompliant.
            RETURN JSON:
            {{
              "decision":"<valid|noncompliant>",
              "justification_checker":"<string>"
            }}
        """),
        
        "p3_system":
            "You are a professional editor. Be STRICT. Accept only if the candidate "
            "value is coherent, not contradicted by any row data, complies with the "
            "instructions, and is not labelled 'insufficient evidence' or similar. "
            "You may correct minor issues. Otherwise reject.",
        
        "p3_user_template": textwrap.dedent("""
            FIELD: {field}

            ALL ROW DATA:
            {row_block}

            PROPOSER1_JSON:
            {p1_json}

            PROPOSER2_JSON:
            {p2_json}

            CHECKER_JSON:
            {checker_json}

            TASK:
              - decision: accept | correct | reject
              - final_value (if accept/correct, no more than 45 words, clean, and compliant)
              - justification_final (no more than 35 words) citing column names.
            RETURN JSON:
            {{
              "decision":"<accept|correct|reject>",
              "final_value":"<string or null>",
              "justification_final":"<string>"
            }}
        """)
    }
}
