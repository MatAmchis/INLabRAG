from typing import Dict, List


REASONING_STEPS: Dict[str, List[str]] = {
  "First author": [
    "Start with the highest-ranked excerpts that look like the title page or citation header where author names are listed.",
    "Scan those excerpts for an ordered author list, often shown as comma-separated names with affiliation markers or superscripts.",
    "Extract the first listed author and keep only the surname exactly as written in the excerpt.",
    "If multiple author lists appear, prefer the title/author block list over shortened citation variants or metadata-only fragments.",
    "Format the final answer exactly as required by the field prompt, without adding initials or extra text.",
    "If the excerpts do not contain an author list, return the field prompt’s fallback and do not infer."
  ],

  "Year published": [
    "Start with the highest-ranked excerpts that resemble a journal citation line, publication info box, or header/footer metadata.",
    "Look for a four-digit year that is explicitly tied to publication context, such as volume/issue/pages or a 'Published' line.",
    "Extract the single best publication year candidate and keep the minimal supporting span that shows publication context.",
    "If multiple years appear, prefer the year associated with final publication/citation context over received/accepted/online-ahead dates.",
    "Format the final answer as a single four-digit year, following the field prompt constraints.",
    "If no publication year is explicitly supported by the excerpts, return the field prompt’s fallback."
  ],

  "Title": [
    "Start with the highest-ranked excerpts that look like the paper’s title header or a complete citation line that includes the title.",
    "Scan for the longest headline-like string, including any subtitle separated by a colon or dash.",
    "Extract the title exactly as written, preserving punctuation and capitalization as it appears in the excerpt.",
    "If multiple title variants appear, prefer the title shown in the main header over shortened running headers.",
    "Format the final answer as the exact full title, with no paraphrasing.",
    "If the title cannot be located in any excerpt, return the field prompt’s fallback."
  ],

  "Language of publication": [
    "Start with excerpts that look like metadata or an article information panel rather than general body text.",
    "Look for explicit language statements such as 'Language: English' or clear publication metadata that names the language.",
    "If an explicit language label is present, extract only the language name as written.",
    "If different excerpts suggest different languages, prefer explicit metadata statements over inference from the prose.",
    "Format the final answer as a single language term that matches the field prompt requirements.",
    "If no excerpt explicitly identifies the language, return the field prompt’s fallback."
  ],

  "Research question/aim": [
    "Start with excerpts that resemble the abstract objective/purpose or introductory aim statement.",
    "Look for phrases such as 'aim', 'objective', 'purpose', 'we sought to', or 'to assess/determine/investigate'.",
    "Extract the most explicit single-sentence aim statement and keep the smallest span that fully expresses it.",
    "If multiple aims are listed, prefer the primary aim statement over secondary aims or discussion framing.",
    "Format the final answer as one precise sentence following the field prompt constraints.",
    "If no explicit aim or research question is present in the excerpts, return the field prompt’s fallback."
  ],

  "Study design (include length of study)": [
    "Start with excerpts that look like Methods/Study design/Setting text or that contain date ranges and durations.",
    "Look for explicit design labels (e.g., cohort, cross-sectional, RCT, mixed-methods) and explicit timing language.",
    "Extract the design and the study period/duration exactly as stated, keeping a minimal supporting span.",
    "If design and dates appear in different excerpts, combine them only when the excerpts clearly describe the same study.",
    "Format the final answer following the field prompt, including both design and length when available.",
    "If either element is not explicitly supported, apply the field prompt’s fallback rather than inferring."
  ],

  "Data collection method": [
    "Start with excerpts that describe Methods/protocols and that mention instruments or data sources.",
    "Look for collection verbs and tools such as surveyed/interviewed/extracted/reviewed, plus questionnaire/checklist/HMIS/registry terms.",
    "Extract the specific collection methods and instruments, keeping a minimal span that states what was collected and how.",
    "If multiple collection methods are mentioned, include only those clearly used for the study’s data collection (not analysis methods).",
    "Format the final answer following the field prompt constraints and level of specificity.",
    "If no explicit collection method is described in the excerpts, return the field prompt’s fallback."
  ],

  "Targeted disease(s)": [
    "Start with excerpts that describe the intervention/program focus or the study aim where diseases are usually named.",
    "Look for explicit disease or condition names near terms like 'target', 'focus', 'intervention', or 'program'.",
    "Extract only diseases explicitly named, and keep the minimal span that links them to the program focus.",
    "If both umbrella categories and specific diseases appear, prefer the specific disease names over umbrella labels.",
    "Format the final answer following the field prompt and avoid adding diseases that are not explicitly stated.",
    "If no disease is explicitly named in the excerpts, return the field prompt’s fallback."
  ],

  "List of all program functions being integrated": [
    "Start with excerpts that enumerate intervention components, especially list-like or bullet-like text.",
    "Look for function terms such as surveillance, diagnosis, treatment, referral, reporting, health education, and logistics.",
    "Extract all functions that are explicitly described as being integrated or combined, keeping a minimal supporting span.",
    "If functions are mentioned elsewhere as general activities, include them only when the excerpt clearly frames them as integrated.",
    "Format the final answer as a complete list consistent with the field prompt constraints.",
    "If integrated functions are not explicitly stated in the excerpts, return the field prompt’s fallback."
  ],

  "Specific activities (related to treatment and/or surveillance) integrated": [
    "Start with excerpts that describe implementation details and operational steps rather than high-level summaries.",
    "Look for concrete activities tied to treatment or surveillance, such as screening, case detection, MDA, follow-up, reporting, or contact tracing.",
    "Extract the specific activities with any stated who/where/how-often details, keeping the smallest span that captures each activity.",
    "If multiple activities appear, keep only those clearly described as part of the integrated approach, and separate treatment from surveillance when possible.",
    "Format the final answer as detailed activity descriptions consistent with the field prompt constraints.",
    "If no specific integrated activities are explicitly described, return the field prompt’s fallback."
  ],

  "Where implemented": [
    "Start with excerpts that describe the setting, study area, or implementation geography.",
    "Look for a country name and any sub-national units such as region, province, state, or district.",
    "Extract the stated locations exactly as written, keeping a minimal span that anchors the location to implementation.",
    "If multiple geographies appear, prefer the geography explicitly tied to the integrated program rather than background examples.",
    "Format the final answer following the field prompt, including sub-national areas when explicitly stated.",
    "If the excerpts do not identify an implementation location, return the field prompt’s fallback."
  ],

  "Specific location(s) of integration & number of sites; note ANC/PHC": [
    "Start with excerpts that list facilities/sites or that report site counts, often in tables or rollout summaries.",
    "Look for facility names/types, numeric site totals, and any explicit ANC or PHC labels.",
    "Extract the site names/types and the total number of sites as stated, keeping minimal spans for each piece of evidence.",
    "If both pilot and endline totals appear, prefer the end-of-implementation total when the excerpt clearly distinguishes phases.",
    "Format the final answer following the field prompt, including ANC/PHC context only when explicitly stated.",
    "If any required elements are not explicitly supported by the excerpts, apply the field prompt’s fallback."
  ],

  "Why setting was selected": [
    "Start with excerpts that explicitly justify site selection, typically phrased as 'selected because' or similar rationale statements.",
    "Look for stated reasons such as burden, feasibility, existing partnerships, access, infrastructure, or representativeness.",
    "Extract the reasons exactly as expressed, keeping the minimal span that ties each reason to setting selection.",
    "If multiple reasons appear across excerpts, keep them distinct rather than merging them into a new inferred explanation.",
    "Format the final answer as a concise rationale consistent with the field prompt constraints.",
    "If no explicit selection rationale is present, return the field prompt’s fallback."
  ],

  "Community setting environment": [
    "Start with excerpts that describe the community context and constraints relevant to implementation.",
    "Look for physical, social, economic, ecological, and infrastructure descriptors, especially transport and access constraints.",
    "Extract the explicit descriptors and keep the minimal span that supports each contextual point.",
    "If different excerpts describe different aspects of context, include them only when they refer to the same study setting.",
    "Format the final answer as a structured context description consistent with the field prompt constraints.",
    "If the community environment is not explicitly described, return the field prompt’s fallback."
  ],

  "Population receiving treatment before implementation": [
    "Start with excerpts labeled baseline, pre-intervention, or 'prior to implementation', especially tables and baseline summaries.",
    "Look for numbers or percentages receiving treatment at baseline, or explicit qualitative baseline statements about coverage and gaps.",
    "Extract the baseline value(s) and keep the minimal span that clearly indicates the pre-implementation timeframe.",
    "Do not substitute early rollout values for baseline unless the excerpt explicitly labels them as baseline.",
    "Format the final answer following the field prompt, using quantitative evidence when available and qualitative evidence otherwise.",
    "If neither quantitative nor qualitative baseline information is present, return the field prompt’s fallback."
  ],

  "Perceptions of population receiving treatment before implementation on the integration before implementation": [
    "Start with excerpts from formative research, baseline qualitative findings, or attributed community statements from before rollout.",
    "Look for explicit descriptions of what patients or community members said, expected, experienced, or believed prior to implementation.",
    "Extract only perceptions that are clearly anchored to the pre-implementation period, keeping minimal supporting spans or quotes.",
    "Exclude post-implementation satisfaction unless the excerpt explicitly frames it as reflecting pre-implementation perceptions.",
    "Format the final answer as a short summary of the stated pre-implementation perceptions, following the field prompt constraints.",
    "If no pre-implementation perceptions are explicitly described, return the field prompt’s fallback."
  ],

  "Rationale for integration (why)": [
    "Start with excerpts that explicitly explain why services were integrated, rather than general background burden statements.",
    "Look for problem statements tied to integration, such as fragmentation, duplication, inefficiency, loss to follow-up, or access barriers.",
    "Extract the stated rationale and keep the minimal span that links the problem to the decision to integrate.",
    "If multiple rationales are present, list them as separate points rather than blending them into a new inferred rationale.",
    "Format the final answer as a concise rationale consistent with the field prompt constraints.",
    "If the excerpts do not explicitly provide a rationale for integration, return the field prompt’s fallback."
  ],

  "Who implemented the intervention before integration": [
    "Start with excerpts that describe program history or responsibilities, especially those using verbs like implemented/carried out/led by.",
    "Look for named organizations or agencies explicitly described as implementers prior to integration.",
    "Extract the implementer names and keep the minimal span that indicates their implementation role and the pre-integration timeframe.",
    "If funders or partners are mentioned, treat them as implementers only when the excerpt explicitly assigns implementation responsibility.",
    "Format the final answer following the field prompt, listing the implementers clearly and precisely.",
    "If no implementers are explicitly named for the pre-integration phase, return the field prompt’s fallback."
  ],

  "Specific activities undertaken to undergo process of integration": [
    "Start with excerpts describing rollout, planning, training, governance, or systems changes used to combine services.",
    "Look for explicit integration-process activities such as joint planning meetings, staff training, harmonized tools/registers, SOPs, pilots, or protocol harmonization.",
    "Extract each process activity and keep the minimal span that shows it was part of the integration process.",
    "Keep integration process steps distinct from routine service delivery unless the excerpt explicitly frames routine delivery as an integration activity.",
    "Format the final answer as a clear list of process activities consistent with the field prompt constraints.",
    "If integration process activities are not explicitly described, return the field prompt’s fallback."
  ],

  "People involved & their roles": [
    "Start with excerpts that name staff groups or stakeholders and describe what they did during implementation or support.",
    "Look for actor labels (e.g., CHWs, nurses, supervisors, district officers, NGO staff) paired with concrete responsibilities.",
    "Extract the actor-role pairings and keep minimal spans that clearly connect each actor to a role.",
    "If the same actor appears in multiple excerpts, keep the most specific role description and remove duplicates.",
    "Format the final answer following the field prompt, describing roles in clear, concrete terms.",
    "If no people-and-roles information is explicitly present, return the field prompt’s fallback."
  ],

  "Health sector staff perspectives on proposed integration before implementation": [
    "Start with excerpts from planning, formative research, or baseline qualitative findings that describe staff views at the initial stage.",
    "Look for explicit staff perspectives such as support, concerns, resistance, workload worries, or training needs tied to pre-implementation.",
    "Extract only perspectives anchored to the planning or pre-rollout period, keeping minimal supporting spans or quotes.",
    "Exclude post-implementation reflections unless the excerpt explicitly states they describe initial-stage perspectives.",
    "Format the final answer as a concise summary of the stated staff perspectives, consistent with the field prompt constraints.",
    "If no pre-implementation staff perspectives are explicitly reported, return the field prompt’s fallback."
  ],

  "Assessments done to inform design of integration (if any)": [
    "Start with excerpts that mention assessments, mapping, baseline studies, needs assessments, or feasibility work connected to planning.",
    "Look for explicit assessment descriptions paired with language indicating they informed or guided integration design decisions.",
    "Extract the assessment type, methods, and design influence as stated, keeping minimal spans for each element.",
    "Exclude evaluations that occur only after implementation unless the excerpt describes them as iterative inputs used to refine design.",
    "Format the final answer following the field prompt, focusing on assessments that shaped the integration design.",
    "If no design-informing assessments are explicitly described, return the field prompt’s fallback."
  ],

  "Mechanisms for coordinating process/getting input from stakeholders": [
    "Start with excerpts describing governance, coordination, stakeholder engagement, or management structures.",
    "Look for named mechanisms such as committees, technical working groups, task forces, review meetings, MoUs, advisory boards, or feedback loops.",
    "Extract the mechanism names and any stated cadence (e.g., monthly, quarterly) only when explicitly provided.",
    "If multiple mechanisms appear, include only those clearly used to coordinate integration or gather stakeholder input.",
    "Format the final answer following the field prompt, listing mechanisms clearly and precisely.",
    "If no coordination or input mechanisms are explicitly reported, return the field prompt’s fallback."
  ],

  "Funding: Cost of implementing integration and who paid": [
    "Search for phrases indicating financial details such as cost of implementation, financial cost, expenses incurred, or budget for integration to find specific figures related to the cost.",
    "Look for specific numbers or currency symbols (e.g., $, €, £) near these phrases to identify the exact cost figure.",
    "Identify sentences that mention funding sources or entities involved in financing by looking for keywords like funded by, financed by, sponsored by, or supported by.",
    "Verify if the text explicitly names the funding entity or entities, ensuring the mention is specific and not vague. Note any specific organizations, government bodies, or entities named as the funding source.",
    "Format the final answer following the field prompt, distinguishing cost figures from funder identities.",
    "If cost and/or payer are not explicitly stated, apply the field prompt’s fallback for the missing element(s)."
  ],

  "List of outcomes of integration studied": [
    "Start with excerpts that define outcomes, endpoints, measures assessed, or evaluation indicators, often in Methods or table/figure captions.",
    "Look for explicit lists of outcomes the study measured, rather than general aims or background problems.",
    "Extract the outcomes exactly as listed and keep minimal spans that show they were assessed.",
    "If outcomes appear in multiple places, consolidate duplicates and keep the clearest wording used by the study.",
    "Format the final answer as a clear list consistent with the field prompt constraints.",
    "If the study outcomes are not explicitly stated in the excerpts, return the field prompt’s fallback."
  ],

  "Facilitators of integration": [
    "Start with excerpts that discuss what enabled implementation success, typically in Results, Discussion, or lessons learned sections.",
    "Look for explicit enabling factors framed as facilitators, enablers, supports, or contributors to success.",
    "Extract each facilitator and keep the minimal span that ties it to successful integration.",
    "If multiple facilitators are mentioned, list them separately and avoid adding generic factors that are not stated.",
    "Format the final answer following the field prompt, using the study’s wording as much as possible.",
    "If no facilitators are explicitly described, return the field prompt’s fallback."
  ],

  "Barriers to integration / how they tried to overcome": [
    "Start with excerpts that describe implementation challenges, constraints, limitations, or problems encountered.",
    "Look for explicit barriers and for linked mitigation strategies introduced by phrases like addressed by/mitigated through/to overcome.",
    "Extract barriers and mitigation actions, keeping minimal spans that show the linkage when it is explicitly stated.",
    "If barriers are listed without solutions, report the barriers; if solutions are listed without barriers, report the solutions as described.",
    "Format the final answer following the field prompt, keeping barriers and mitigations clearly separated when they are not linked.",
    "If neither barriers nor mitigation strategies are explicitly described, return the field prompt’s fallback."
  ],

  "Acceptability": [
    "Start with excerpts that contain stakeholder feedback, acceptability metrics, usability assessments, or satisfaction findings.",
    "Look for direct stakeholder statements or for reported acceptability/readiness/satisfaction measures with described meaning.",
    "Extract the acceptability evidence and keep minimal spans that show who the stakeholder group is and what the metric or perception refers to.",
    "If multiple stakeholder groups are reported, keep their perspectives separate rather than combining them into a single summary.",
    "Format the final answer following the field prompt, including context for any numeric measures when they are explicitly provided.",
    "If no acceptability evidence is explicitly present, return the field prompt’s fallback."
  ],

  "Uptake and coverage": [
    "Start with excerpts that report participant flow, cascade tables, coverage indicators, or counts of services delivered.",
    "Look for numbers screened/tested/treated and for any coverage percentages, including any denominators when they are provided.",
    "Extract the uptake and coverage figures exactly as stated, keeping minimal spans that clarify what each number represents.",
    "When both counts and percentages are available, report both; when only one is available, report what is stated without inventing missing values.",
    "Format the final answer following the field prompt, keeping utilization and coverage clearly distinguished.",
    "If uptake and coverage are not reported in the excerpts, return the field prompt’s fallback."
  ],

  "Impact on cost – give specific numbers": [
    "Start with excerpts that contain monetary values, costing tables, budget language, or economic evaluation results.",
    "Look for currency-denominated metrics such as total cost, unit cost, cost per patient/case/visit, savings, budget impact, or cost-effectiveness ratios.",
    "Extract only explicit monetary figures and keep minimal spans that state what each cost number represents and the relevant comparator or timeframe, if provided.",
    "If multiple cost metrics appear, keep them labeled by perspective, comparator, and time horizon when the excerpt provides that context.",
    "Format the final answer following the field prompt, reporting currency and meaning for each metric exactly as stated.",
    "If no explicit monetary cost metrics are present in the excerpts, return the field prompt’s fallback."
  ],

  "Impact on infection/disease – give specific numbers": [
    "Start with excerpts that report clinical or epidemiologic outcomes, especially tables or figure captions with numeric disease indicators.",
    "Look for incidence, prevalence, case counts, mortality, positivity, or treatment success measures that are reported with numbers.",
    "Extract the numeric disease outcomes and keep minimal spans that specify timeframe and comparator (before/after or group comparison) when stated.",
    "Verify the excerpt explicitly links the disease outcome to the integrated intervention evaluated in this paper and not a different campaign, prior program, or background example.",
    "Format the final answer following the field prompt, including denominators and timeframes only when explicitly provided.",
    "If no quantitative disease outcomes are reported in the excerpts, return the field prompt’s fallback."
  ],

  "Other outcomes measured and results": [
    "Start with excerpts describing secondary outcomes, additional endpoints, or reported results beyond the primary outcomes.",
    "Look for each named outcome paired with its corresponding quantitative result or qualitative finding.",
    "Extract outcomes and their results, keeping minimal spans that make the linkage between the outcome and result explicit.",
    "If an outcome is named but no result is provided anywhere in the excerpts, record that the result is not reported rather than inventing it.",
    "Format the final answer following the field prompt, keeping outcomes and results clearly paired.",
    "If no additional outcomes or results are explicitly reported, return the field prompt’s fallback."
  ],

  "Sustainability": [
    "Start with excerpts that discuss continuation, scale-up, institutionalization, ownership, or post-study plans, usually in Discussion/Conclusion.",
    "Look for explicit statements about whether the program continued, was adopted into routine services, or transitioned to local or government support.",
    "Extract sustainability evidence and keep minimal spans that distinguish observed continuation from planned future intentions.",
    "If multiple sustainability claims appear, label them according to the excerpt wording rather than merging them into a single claim.",
    "Format the final answer following the field prompt, emphasizing what is explicitly supported by the excerpts.",
    "If sustainability is not explicitly discussed, return the field prompt’s fallback."
  ],

  "Major recommendations of the paper based on its findings": [
    "Start with excerpts that contain recommendations, implications, or conclusion statements with directive language.",
    "Look for action-oriented phrases such as recommend/should/advise/propose, tied to the study’s findings.",
    "Extract each recommendation and keep minimal spans that show it is presented as a recommendation rather than background commentary.",
    "If multiple recommendations exist, list them separately and prioritize those explicitly grounded in the study’s results.",
    "Format the final answer following the field prompt, keeping recommendations concrete and actionable as stated.",
    "If no recommendations are explicitly stated in the excerpts, return the field prompt’s fallback."
  ]
}
