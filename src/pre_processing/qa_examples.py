from typing import Dict, List


FEW_SHOT_EXAMPLES: Dict[str, List[str]] = {
  "First author": [
    "Okeke",
    "Hernandez",
    "Ndlovu",
  ],

  "Year published": [
    "2018",
    "2021",
    "2024",
  ],

  "Title": [
    "Integrating TB symptom screening and GeneXpert referral into primary health care: An interrupted time-series evaluation in rural Uganda",
    "One-stop ANC-ART model to strengthen PMTCT of HIV: A cluster-randomized trial in Mozambique",
    "Community case management of malaria integrated into routine PHC services: A mixed-methods implementation study in Cambodia",
  ],

  "Language of publication": [
    "English",
    "Spanish",
    "Portuguese",
  ],

  "Research question/aim": [
    "To assess whether integrating routine TB screening and on-site specimen referral into outpatient visits increased bacteriologically confirmed TB detection and treatment initiation.",
    "To evaluate the effect of integrating HIV testing and linkage into antenatal care on maternal ART initiation and early infant diagnosis coverage.",
    "To determine whether integrating malaria case surveillance into routine PHC reporting improved the timeliness and completeness of notification of confirmed malaria cases.",
  ],

  "Study design (include length of study)": [
    "Interrupted time-series analysis; Jan 2016–Dec 2020",
    "Cluster randomized trial; 24 months",
    "Mixed-methods implementation evaluation; Mar–Nov 2022",
  ],

  "Data collection method": [
    "Routine register review (OPD and TB presumptive registers); HMIS/DHIS2 extraction; facility observation (standard checklist); key informant interviews (semi-structured guide)",
    "Household survey (structured questionnaire); ANC/PMTCT register abstraction; patient exit interviews; focus group discussions (topic guide)",
    "Community health worker logs review; rapid facility assessment (readiness checklist); time-motion observation; supervisory meeting minutes review",
  ],

  "Targeted disease(s)": [
    "Tuberculosis, specifically focused on MDR-TB",
    "HIV",
    "Malaria",
  ],

  "List of all program functions being integrated": [
    "Screening/triage; diagnostic testing/specimen referral; treatment initiation; follow-up/adherence support; routine surveillance reporting (HMIS); supervision/quality improvement",
    "ANC service delivery; HIV testing and counseling; ART initiation and refills; infant prophylaxis/early infant diagnosis; referral and tracking; commodity management",
    "Case detection; community-based treatment; stock management; routine surveillance/reporting; supportive supervision; community engagement/health education",
  ],

  "Specific activities (related to treatment and/or surveillance) integrated": [
    "Routine TB symptom screening at OPD triage; sputum collection at PHC sites; scheduled sample transport to a GeneXpert hub; results returned to facilities via SMS; standardized referral for treatment initiation with register-based tracking.",
    "Same-day HIV testing offered during first ANC visit; immediate ART initiation at the ANC clinic for eligible clients; integrated follow-up visits aligned with the ANC schedule; infant DBS collection for early infant diagnosis at postnatal visits; reporting through routine PMTCT registers.",
    "Community health workers used RDTs for febrile patients and provided ACT per protocol; confirmed cases were recorded in CHW logs and submitted weekly to the health center; facility staff compiled cases into HMIS reports; supervisory visits reviewed registers and data quality.",
  ],

  "Where implemented": [
    "Uganda; Wakiso and Mukono districts",
    "India; Bihar (Gaya and Nalanda districts)",
    "Guatemala; Alta Verapaz Department",
  ],

  "Specific location(s) of integration & number of sites; note ANC/PHC": [
    "Government health centers and dispensaries; 18 sites; PHC",
    "ANC clinics within district hospitals and health centers; 12 sites; ANC",
    "Community health posts linked to PHC clinics; 35 sites; PHC",
  ],

  "Why setting was selected": [
    "Selected because the districts had high TB notification gaps and an existing GeneXpert hub network that could support specimen referral from PHC facilities.",
    "Chosen due to low baseline PMTCT coverage, high ANC attendance, and program interest in testing an integrated ANC-based ART delivery model.",
    "Selected because the area was malaria-endemic with frequent outbreaks and relied heavily on CHW-delivered care, making integration into routine reporting operationally feasible.",
  ],

  "Community setting environment": [
    "Remote rural communities with limited transport and seasonal road inaccessibility; most households relied on subsistence farming and used PHC facilities as the first point of care.",
    "Peri-urban settlements with high population density and informal employment; clinics faced crowding and intermittent commodity stockouts.",
    "Mountainous villages with dispersed households and long travel times to facilities; mobile network coverage was variable and affected communication for referrals and reporting.",
  ],

  "Population receiving treatment before implementation": [
    "Baseline ART coverage among eligible adults was 52% (1,040/2,000) in the catchment area.",
    "At baseline, only 9% (270/3,000) of eligible children received deworming through routine services.",
    "Before integration, baseline treatment coverage was described as low due to frequent stockouts and limited clinic days (quantitative baseline not reported).",
  ],

  "Perceptions of population receiving treatment before implementation on the integration before implementation": [
    "- Patients; described long wait times and multiple visits to obtain testing and treatment, and preferred fewer facility visits.\n- Community members; expressed concerns about transport costs and losing a day of work for each clinic visit.",
    "- Pregnant clients; reported privacy concerns about HIV testing at ANC and wanted clearer counseling on confidentiality.\n- Male partners; indicated mixed willingness to attend ANC-based services due to stigma and competing work obligations.",
    "- Caregivers; reported that service hours and travel distance were the main obstacles to accessing treatment for children.\n- Community volunteers; felt that clearer referral pathways would help families navigate services.",
  ],

  "Rationale for integration (why)": [
    "To reduce missed opportunities for case detection by embedding screening and diagnostic referral into routine PHC encounters and standardizing reporting through HMIS.",
    "To improve continuity of care by aligning testing, treatment initiation, and follow-up within a single service platform and reducing loss to follow-up between separate clinics.",
    "To increase coverage and reporting completeness by using the existing PHC workforce and registers rather than running parallel vertical reporting systems.",
  ],

  "Who implemented the intervention before integration": [
    "Ministry of Health TB Program",
    "National Malaria Control Program",
    "District Health Office",
  ],

  "Specific activities undertaken to undergo process of integration": [
    "Joint planning meetings across TB and PHC leadership; revision of triage workflows; training of facility staff; rollout of integrated screening tools; start-up supervision with data-quality checks.",
    "Harmonization of registers and reporting forms; integrated commodity quantification; phased rollout with pilot facilities; quarterly review meetings to refine procedures.",
    "Development of standard operating procedures; mentorship of CHWs and facility staff; integration of indicators into HMIS; establishment of feedback loops for reporting errors.",
  ],

  "People involved & their roles": [
    "(1) Nurses/midwives; delivered integrated ANC services and initiated ART. (2) Facility in-charges; ensured workflow adoption and supervised documentation. (3) District supervisors; conducted mentorship and data-quality audits.",
    "(1) Community health workers; conducted screening/testing and provided first-line treatment per protocol. (2) Data clerks; aggregated and submitted HMIS reports.",
    "(1) Program managers; coordinated implementation and partner alignment. (2) Community leaders; supported outreach and mobilization.",
  ],

  "Health sector staff perspectives on proposed integration before implementation": [
    "Staff supported integration in principle but anticipated increased workload unless documentation was simplified and additional supervision provided.",
    "Facility teams were concerned about maintaining confidentiality for HIV services within ANC and requested training on counseling and client flow.",
    "CHWs expressed enthusiasm for a single reporting pathway but worried about test stockouts and transport for referrals.",
  ],

  "Assessments done to inform design of integration (if any)": [
    "Facility readiness assessment; identified gaps in diagnostics, staffing, and commodity storage and informed the training and supply plan.",
    "Baseline HMIS data-quality audit; highlighted under-reporting and guided indicator definitions and reporting schedules.",
    "Stakeholder mapping and workflow observations; used to redesign patient flow and clarify roles across cadres.",
  ],

  "Mechanisms for coordinating process/getting input from stakeholders": [
    "District Technical Working Group with monthly meetings; quarterly joint review meetings with implementing partners; routine supervisory feedback during site visits.",
    "Steering Committee chaired by the District Health Officer; memorandum of understanding with partners; community advisory meetings held every two months.",
    "Integrated data review meetings at facilities (monthly); WhatsApp/SMS reporting channel for rapid troubleshooting; annual stakeholder workshop to update SOPs.",
  ],

  "Funding: Cost of implementing integration and who paid": [
    "Cost: $145,000 (training, supervision, tools over 18 months); Funder(s): Global Fund; Ministry of Health",
    "Cost: $3.20 per person screened; Funder(s): USAID",
    "Cost: cost not reported; Funder(s): Ethiopian MoH",
  ],

  "List of outcomes of integration studied": [
    "TB case detection; time to treatment initiation; completeness of HMIS reporting; patient visit burden",
    "ANC-based HIV testing uptake; ART initiation at first ANC visit; early infant diagnosis coverage; retention at 6 months",
    "Malaria test positivity; treatment adherence to guidelines; reporting timeliness; stockout frequency",
  ],

  "Facilitators of integration": [
    "(1) Existing PHC supervision structure; enabled routine mentorship and rapid course-correction. (2) Availability of standardized registers; supported consistent documentation and reporting. (3) Reliable commodity supply during rollout; reduced interruptions to integrated service delivery.",
    "(1) Strong district leadership; enabled coordination across programs and partner alignment.",
    "(1) Community trust in CHWs; increased acceptance of screening/testing and follow-up. (2) Simple job aids; improved protocol adherence across cadres.",
  ],

  "Barriers to integration / how they tried to overcome": [
    "(1) Staff workload and documentation burden; simplified registers and added data clerk support. (2) Specimen transport delays; established fixed transport schedules and backup couriers. (3) Commodity stockouts (RDTs/medicines); introduced monthly quantification and emergency redistribution between facilities.",
    "(1) Limited private space for counseling; reorganized client flow and designated confidential counseling areas.",
    "(1) Weak data completeness; implemented monthly data-quality audits and feedback dashboards. (2) CHW turnover; added refresher trainings and peer-mentor support.",
  ],

  "Acceptability": [
    "Patients reported fewer visits and clearer referral pathways; 82% (246/300) of surveyed clients rated the integrated service as satisfactory.",
    "Health workers; 72% (36/50) reported the integrated workflow was acceptable; interviews indicated acceptability depended on staffing levels and availability of commodities.",
    "CHWs reported the integrated register was easier than parallel forms; 90% (45/50) of CHWs said they would recommend continuing the integrated approach.",
  ],

  "Uptake and coverage": [
    "Uptake: 9,840 screened; 1,120 tested; 310 started treatment (12 months). Coverage: 62% (9,840/15,900) of the target catchment population screened.",
    "Uptake: 1,450 pregnant clients tested for HIV (6 months). Coverage: 88% (1,450/1,650) of first-visit ANC attendees received testing.",
    "Uptake: 7,200 febrile patients tested with RDTs; 1,980 treated with ACT (18 months). Coverage: 71% (7,200/10,100) of expected febrile presentations captured in CHW logs.",
  ],

  "Impact on cost – give specific numbers": [
    "Cost per person screened: $3.20; Cost per case detected: $184; Incremental program cost: $145,000 over 18 months.",
    "Total program cost: $1.2 million (24 months); Cost per client initiated on ART: $58; Estimated annual cost savings from reduced duplicate visits: $210,000.",
    "Incremental cost-effectiveness ratio (ICER): $420 per DALY averted; Cost per malaria case averted: $27 (vs. comparator community model).",
  ],

  "Impact on infection/disease – give specific numbers": [
    "Bacteriologically confirmed TB notifications increased from 18 to 27 per 100,000 per quarter after integration (interrupted time-series, 2016–2020).",
    "HIV positivity among first-visit ANC attendees was 3.8% (55/1,450); same-day ART initiation increased from 41% (68/165) pre-integration to 79% (131/165) post-integration (6 months).",
    "Malaria test positivity declined from 18% (450/2,500) at baseline to 11% (308/2,800) at follow-up over 12 months in intervention facilities.",
  ],

  "Other outcomes measured and results": [
    "(1) Reporting timeliness; improved from 62% to 91% on-time submissions (12 months).(2) Median time from test to treatment; decreased from 7 days to 2 days. (3) Stockout days; decreased from 14 to 4 days per quarter. (4) Community awareness of services; increased in qualitative interviews (no quantitative estimate reported)",
    "(1) Client travel cost per visit; decreased from $2.10 to $1.30 (self-report). (2) Retention at 6 months; increased from 68% to 78%. (3) Results not reported for service quality score (outcome listed, no result provided).",
    "(1) Guideline adherence for case management; increased from 73% to 89% of observed encounters. (2) Data completeness; increased from 70% to 94% of required fields in registers.",
  ],

  "Sustainability": [
    "The integrated indicators were incorporated into routine HMIS reporting, and supervision was shifted to the district health team after partner support ended.",
    "The model was adopted as standard practice in participating facilities and included in revised district guidelines, with ongoing commodity procurement managed through the national supply system.",
    "Sustainability was described as dependent on continued funding for supervision and commodities, and the authors noted no confirmed post-study financing commitment.",
  ],

  "Major recommendations of the paper based on its findings": [
    "(1) Integrate screening and referral tools into standard PHC workflows and registers. (2) Maintain routine data-quality audits with feedback to facilities.",
    "(1) Deliver integrated training and mentorship across cadres before rollout. (2) Align commodity forecasting with integrated service volumes. (3) Strengthen confidentiality and patient flow for sensitive services within ANC/PHC.",
    "(1) Scale the model in high-burden districts using a phased approach.(2) Evaluate long-term outcomes and equity impacts with routine data. (3) Standardize indicators to avoid parallel reporting systems.",
  ],
}
