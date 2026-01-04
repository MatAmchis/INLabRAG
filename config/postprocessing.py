import os
import random
from typing import Any, Dict, List

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class PostprocessingConfig:
    SEED = int(os.getenv("SEED", "42"))
    POST_MODEL = os.getenv("POST_MODEL", "gpt-5.2")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))
    ROW_CONTEXT_CHAR_LIMIT = int(os.getenv("ROW_CONTEXT_CHAR_LIMIT", "12000"))

    PRIORITY_FIELDS: List[str] = [
        "Study design (include length of study)",
        "Data collection method",
        "List of all program functions being integrated",
        "Community setting environment",
        "Rationale for integration (why)",
        "Specific activities undertaken to undergo process of integration",
        "Mechanisms for coordinating process/getting input from stakeholders",
        "Funding: Cost of implementing integration and who paid",
        "List of outcomes of integration studied",
        "Facilitators of integration",
        "Barriers to integration / how they tried to overcome",
        "Uptake and coverage",
        "Sustainability",
    ]

    QUESTION_PROMPT_OVERRIDES: Dict[str, str] = {
        "First author": (
            "Goal: Identify the first-listed author of the paper.\n"
            "Output format: Return ONLY the author’s last name exactly as printed.\n"
            "Exclude: Do not include given names, initials, titles, degrees, affiliations, or 'et al.'\n"
            "Fallback: If the author list is not present, return 'Not reported'."
        ),
        "Year published": (
            "Goal: Identify the publication year of the paper as printed by the journal.\n"
            "Output format: Return a single four digit year.\n"
            "Exclude: Do not return years for date of submission/received/accepted/online first unless the journal explicitly defines that as the official publication date.\n"
            "Fallback: If the year published is not present, return 'Not reported'."
        ),
        "Title": (
            "Goal: Identify the complete paper title.\n"
            "Output format: Return the full title EXACTLY as printed. INCLUDE any subtitle after a colon or dash.\n"
            "Exclude: Do not paraphrase, shorten, correct capitalization, or remove punctuation.\n"
            "Fallback: If the title is not present, return 'Not reported'."
        ),
        "Language of publication": (
            "Goal: Identify the primary language in which the article is written.\n"
            "Output format: Return a single-word language label (e.g., English, French).\n"
            "Exclude: Do not use vague descriptors such as multilingual.\n"
            "Fallback: If the language is not stated, infer based on the content."
        ),
        "Research question/aim": (
            "Goal: Identify the study’s main research question or primary objective/focus.\n"
            "Output format: Return ONE precise sentence showing the authors’ stated aim. Use the authors’ wording when possible.\n"
            "Exclude: Do not return broad filler such as 'the study looked at health'. Be specific and accurate.\n"
            "Fallback: If the research question/aim is not present, return 'Not reported'."
        ),
        "Study design (include length of study)": (
            "Goal: Identify the study design and the study time span and/or duration.\n"
            "Output format: Return 'Design; duration/time period'. Examples: 'Cross-sectional survey; June–Aug 2019' or 'Randomized controlled trial; 12 months'. If more than one design applies, list it.\n"
            "Exclude: Do not infer a design or duration that is not explicitly stated.\n"
            "Fallback: If design is stated but duration is not, return '[study design]; duration not reported'. If neither is stated, return 'Not reported'."
        ),
        "Data collection method": (
            "Goal: Identify the specific tools, techniques, and protocols used to collect data in the study.\n"
            "Output format: Return an accurate (must be mentioned in the text) yet comprehensive list of methods and/or instruments used.\n"
            "Exclude: Do not restate analysis methods as data collection (e.g., regression) unless the paper frames them as part of collection.\n"
            "Fallback: If the data collection method is not present, return 'Not reported'."
        ),
        "Targeted disease(s)": (
            "Goal: Identify the specific disease(s), infections, or health conditions that the program/intervention targets.\n"
            "Output format: Return a list of explicit disease names as stated. Be as specific as the paper allows.\n"
            "Exclude: Do not return broad categories such as 'NTDs' if the paper names specific diseases.\n"
            "Fallback: If the targeted disease(s) are not mentioned, return 'Not reported'."
        ),
        "List of all program functions being integrated": (
            "Goal: List all the health program functions that were integrated in the intervention.\n"
            "Output format: Return a list of functions. Be as specific as the paper allows, but do not go beyond what is clearly in the paper.\n"
            "Exclude: Do not list general goals unless they are described as operational functions.\n"
            "Fallback: If the program functions being integrated are not present, return 'Not reported'."
        ),
        "Specific activities (related to treatment and/or surveillance) integrated": (
            "Goal: Identify the concrete treatment and/or surveillance activities that were integrated.\n"
            "Output format: Return clear and specific activity descriptions that include what was done and for whom/where if stated.\n"
            "Exclude: Do not use vague phrases such as 'improved surveillance'.\n"
            "Fallback: If no specific activities are described, return 'Not reported'."
        ),
        "Where implemented": (
            "Goal: Describe the country and any sub-national areas (e.g., provinces/regions/districts) where the integration was implemented.\n"
            "Output format: Return 'Country; sub-national areas' as stated.\n"
            "Exclude: Do not return overly broad regions such as 'South America' unless that is the only stated location.\n"
            "Fallback: If location is not stated, return 'Not reported'."
        ),
        "Specific location(s) of integration & number of sites; note ANC/PHC": (
            "Goal: Identify the implementation sites, their type, and the total number of sites. Indicate ANC/PHC if applicable.\n"
            "Output format: Return 'Site types/names; total # sites; ANC/PHC if stated'.\n"
            "Exclude: Report the total number of sites involved by the end of the implementation period, not just the initial phase.\n"
            "Fallback: If sites are mentioned but count is missing, include sites and write 'number of sites not reported'. If neither is stated, return 'Not reported'."
        ),
        "Why setting was selected": (
            "Goal: Describe the specific reasons why the geographical, demographic, or institutional setting was chosen.\n"
            "Output format: Return a concise explanation of the authors’ stated reasons for choosing the setting.\n"
            "Exclude: Do not infer motivations not stated.\n"
            "Fallback: If not stated, return 'Not reported'."
        ),
        "Community setting environment": (
            "Goal: Describe the physical, social, economic, or ecological context of the community where the integration occurred.\n"
            "Output format: Return a brief descriptive summary that reflects what the paper states.\n"
            "Exclude: Do not add assumptions about the setting.\n"
            "Fallback: If not described, return 'Not reported'."
        ),
        "Population receiving treatment before implementation": (
            "Goal: Identify any baseline/pre-intervention data regarding the number or percentage of people receiving treatment.\n"
            "Output format: Prefer quantitative baseline (n/N, %, rates). If unavailable, report explicit qualitative baseline statements (e.g., 'low coverage').\n"
            "Exclude: Do not treat early-intervention numbers as baseline unless explicitly labeled baseline/pre-intervention.\n"
            "Fallback: If neither quantitative nor qualitative baseline information is stated, return 'Not reported'."
        ),
        "Perceptions of population receiving treatment before implementation on the integration before implementation": (
            "Goal: Identify any reported patient/community perceptions, concerns, preferences, or expectations regarding existing services and/or the planned integration prior to implementation.\n"
            "Output format: Return 'Stakeholder group/actor; summary of pre-implementation perceptions' in a structured list.\n"
            "Exclude: Do not use post-implementation satisfaction unless explicitly described as referring to the pre-implementation period.\n"
            "Fallback: If not stated, return 'Not reported'."
        ),
        "Rationale for integration (why)": (
            "Goal: Identify why integration was pursued and the specific problem integration was meant to solve.\n"
            "Output format: Return a concise statement listing the stated drivers.\n"
            "Exclude: Do not generalize beyond what the authors state.\n"
            "Fallback: If not stated, return 'Not reported'."
        ),
        "Who implemented the intervention before integration": (
            "Goal: Identify who implemented the intervention before it was integrated.\n"
            "Output format: Return the named organizations/agencies/groups responsible.\n"
            "Exclude: Do not list partners unless the paper describes them as implementers.\n"
            "Fallback: If not stated, return 'Not reported'."
        ),
        "Specific activities undertaken to undergo process of integration": (
            "Goal: Identify the operational steps taken to execute the integration process.\n"
            "Output format: Return a list of specific process activities.\n"
            "Exclude: Do not list clinical activities unless they are explicitly part of the integration process steps.\n"
            "Fallback: If not described, return 'Not reported'."
        ),
        "People involved & their roles": (
            "Goal: Identify stakeholder groups/actors involved in integration and what each did.\n"
            "Output format: Return 'Stakeholder group/actor; role' in a structured list.\n"
            "Exclude: Do not list groups without an associated role if roles are available.\n"
            "Fallback: If not described, return 'Not reported'."
        ),
        "Health sector staff perspectives on proposed integration before implementation": (
            "Goal: Identify staff views during the pre-implementation stage.\n"
            "Output format: Return a brief summary of stated pre-implementation staff perspectives.\n"
            "Exclude: Do not use post-implementation staff satisfaction unless explicitly framed as pre-implementation views.\n"
            "Fallback: If no pre-implementation staff perspectives are stated, return 'Not reported'."
        ),
        "Assessments done to inform design of integration (if any)": (
            "Goal: Identify any formative studies, baseline assessments, mapping exercises, evaluations, or analyses conducted to guide the design of the integration process.\n"
            "Output format: Return 'Assessment type; how it informed design' in a structured list.\n"
            "Exclude: Do not count outcome evaluation after implementation unless described as formative for design.\n"
            "Fallback: If not stated, return 'Not reported'."
        ),
        "Mechanisms for coordinating process/getting input from stakeholders": (
            "Goal: Identify governance/coordination mechanisms used to manage integration and gather stakeholder input.\n"
            "Output format: Return named structures/processes (committees, TWGs, review meetings, MoUs, advisory boards, feedback loops) with frequency if stated.\n"
            "Exclude: Do not infer structures not stated.\n"
            "Fallback: If not described, return 'Not reported'."
        ),
        "Funding: Cost of implementing integration and who paid": (
            "Goal: Identify any reported implementation cost and who funded/supported the work.\n"
            "Output format: Return 'Cost: <value+currency/metric OR cost not reported>; Funder(s): <names OR funding source not reported>'.\n"
            "Exclude: Do not ignore qualitative information. If quantitative cost is not given but a funder is mentioned to significantly contribute, report that as such.\n"
            "Fallback: If no monetary cost is stated but funders are named, return 'Cost: cost not reported; Funder(s): <named funders>'. If neither cost nor funders are stated, return 'Not reported'."
        ),
        "List of outcomes of integration studied": (
            "Goal: Identify what outcomes the study measured to evaluate the integration.\n"
            "Output format: Return a list of outcome variables (coverage, timeliness, quality, costs, acceptability, disease indicators, etc.) as stated.\n"
            "Exclude: Do not include background problems unless labeled as measured outcomes.\n"
            "Fallback: If not stated, return 'Not reported'."
        ),
        "Facilitators of integration": (
            "Goal: Identify factors reported as enabling or supporting successful integration.\n"
            "Output format: Return 'Facilitator; how it enabled/supported integration' in a structured list.\n"
            "Exclude: Do not add generic facilitators not mentioned in the paper.\n"
            "Fallback: If not stated, return 'Not reported'."
        ),
        "Barriers to integration / how they tried to overcome": (
            "Goal: Identify reported barriers/challenges and any mitigation strategies.\n"
            "Output format: Return 'Barrier; mitigation' pairs when possible. If mitigation is not stated, list barriers alone.\n"
            "Exclude: Do not invent solutions if only barriers are reported.\n"
            "Fallback: If neither barriers nor mitigations are stated, return 'Not reported'."
        ),
        "Acceptability": (
            "Goal: Identify stakeholder perceptions of the integrated approach and any acceptability metrics.\n"
            "Output format: Summarize qualitative perceptions AND include quantitative acceptability/readiness/satisfaction metrics WITH context, if available.\n"
            "Exclude: Do not treat monetary values as acceptability.\n"
            "Fallback: If not stated, return 'Not reported'."
        ),
        "Uptake and coverage": (
            "Goal: Identify uptake (utilization) and coverage achieved by the integrated program.\n"
            "Output format: Report quantitative uptake (numbers screened/tested/treated) and coverage (percentage of target population reached) with timeframe when available.\n"
            "Exclude: Do not invent denominators, target population sizes, or coverage percentages not explicitly reported.\n"
            "Fallback: If no quantitative uptake or coverage data are reported, return 'Not reported'."
        ),
        "Impact on cost – give specific numbers": (
            "Goal: Provide quantitative cost metrics resulting from the integration. ONLY extract data that explicitly mentions monetary costs/expenditures/savings/financial impact with currency units.\n"
            "Output format: Report explicit monetary metrics WITH currency and what they represent (cost per patient/case, total program cost, cost savings, budget impact, ICER, etc.).\n"
            "Exclude: Do not infer costs from non-monetary resources; do not report general study budgets unless explicitly presented as integration cost impact.\n"
            "Fallback: If no quantitative monetary cost metrics are reported, return 'Not reported'."
        ),
        "Impact on infection/disease – give specific numbers": (
            "Goal: Extract quantitative infection or disease outcome measures reported in relation to the program/intervention.\n"
            "Output format: Report numeric disease indicators (incidence, prevalence, case counts, mortality, positivity, treatment success) with timeframe and comparator (before/after or control) when stated.\n"
            "Exclude: Do not report disease impacts from other programs/campaigns mentioned as background unless the excerpt explicitly links the outcome to the integrated intervention under study, and do not report truncated fragments or incomplete metrics that are not related.\n"
            "Fallback: If no quantitative infection/disease outcomes are reported, return 'Not reported'."
        ),
        "Other outcomes measured and results": (
            "Goal: Provide all outcomes measured in the study (quantitative and qualitative) and how they changed (before/after) when applicable.\n"
            "Output format: Pair each outcome with its result, including timeframe/comparator when stated.\n"
            "Exclude: Do not list outcomes without results; if outcomes are listed but results missing, explicitly say 'results not reported' for that outcome.\n"
            "Fallback: If no additional outcomes/results are reported, return 'Not reported'."
        ),
        "Sustainability": (
            "Goal: Identify evidence or discussion of long-term continuation (institutionalization, local ownership, transition to government funding, scale-up, post-study continuation, etc.).\n"
            "Output format: Return a brief summary based on what the paper clearly mentions.\n"
            "Exclude: Do not infer sustainability from short-term outcomes.\n"
            "Fallback: If not discussed, return 'Not reported'."
        ),
        "Major recommendations of the paper based on its findings": (
            "Goal: Identify the authors’ actionable recommendations that follow from their findings.\n"
            "Output format: Return a clear list of recommendations (policy, practice, implementation, research, etc.).\n"
            "Exclude: Do not infer recommendations not clearly outlined in the paper.\n"
            "Fallback: If no recommendations are stated, return 'Not reported'."
        ),
    }

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    @classmethod
    def display(cls) -> None:
        print("Post-processing Configuration:")
        print("=" * 50)
        print(f"  POST_MODEL: {cls.POST_MODEL}")
        print(f"  LLM_TIMEOUT: {cls.LLM_TIMEOUT}")
        print(f"  ROW_CONTEXT_CHAR_LIMIT: {cls.ROW_CONTEXT_CHAR_LIMIT}")
        print(f"  PRIORITY_FIELDS: {len(cls.PRIORITY_FIELDS)} fields")
        print("=" * 50)


__all__ = ["PostprocessingConfig", "set_seed"]
