# Proposed model specification

This is a research specification. As of v0.3.0 the following have tested implementations:
- the reference layer (with exact and factorised update rules);
- question partitions and planning (including a stop option and exact finite-horizon planning);
- the evidence ledger;
- exact provenance-dependence inference;
- network generators and paired experiment tooling.

See [implementation status](05_implementation_status.md) and [the v0.3 audit](08_v0.3_audit_and_extensions.md). Provenance and planning are not yet coupled to the network simulator, and nothing has been validated empirically.

## Model name and architecture

Use **KANM** as a working project label and retain **Laputa** for the historical baseline. Separate the software into five layers so the original model remains reproducible:

1. **Laputa reference layer:** binary proposition, inquiry, directed communication, threshold rules, source-specific trust distributions, and original independence assumption.
2. **Question layer:** a goal question plus linked subquestions, refinements, alternatives, and answer spaces.
3. **Evidence layer:** observations/reports with source, source ancestry, question scope, method, time, uncertainty, and support/contradiction relations.
4. **Inquiry allocation layer:** seeker/helper roles, domain competencies, costs, availability, inquiry actions, and an explicit allocation policy.
5. **Evaluation layer:** accuracy, calibration, uncertainty, time/cost to resolve questions, dissent, and group-distribution measures.

The baseline layer must be frozen and versioned. Extensions should be opt-in modules with explicit configuration so a result can be compared to the same run under baseline assumptions.

## Agent and message concepts

An agent may have: current belief state; empirical inquiry performance; a separate subjective trust distribution for each source; domain-specific capability; budget/time; available inquiry actions; communication policy; and role in the current question plan. Actual reliability and perceived trust must remain separate variables.

An evidence item should carry:

- a stable evidence identifier and parent/provenance identifiers;
- source and recipient, if communicated;
- target question and answer option(s) supported;
- generation method and quality parameters;
- scope fields such as population, setting, time range, and assumptions;
- uncertainty and missingness flags;
- whether the item is new, repeated, corrected, or retracted.

Free-text “limitations” alone are not computationally interpretable. A first implementation should use structured fields and preserve the original statement for display.

## Question representation and question substitution

Represent a question as a partition over a finite set of possible states (implemented: `expected_information_gain(..., target=partition)`, `question_relevance`, `refines`). A question graph should record parent-child relations and distinguish:

- **clarification:** resolves ambiguity while preserving the object question;
- **decomposition:** creates subquestions whose answers jointly bear on the parent;
- **refinement:** narrows scope or adds a condition;
- **alternative framing:** investigates a plausible competing formulation;
- **substitution:** changes the active question because the original is unanswerable, unactionable, or lower value under stated criteria;
- **abandonment:** stops inquiry with a recorded reason.

Never silently overwrite the original question. Store the mapping and the reason for every transition; keep the original goal in the results. This permits analysis of scope drift and agenda capture.

## Choosing a question or helper

The action policy should be replaceable. Candidate policies include a random baseline, fixed/expert assignment, expected information gain, and expected value of sample information under the task's terminal utility. The prototype provides finite-hypothesis Bayesian query actions and both an entropy-based objective (`expected information gain - cost_weight × cost`) and a decision-utility-based value-of-information criterion. The action's likelihood table represents the helper/method's competence. Information gain is a useful comparison policy, not a universal definition of good research: the scientific or practical utility function and the conversion between utility and cost must be stated and sensitivity-tested. The implementation selects among supplied actions; it does not generate questions or infer competence from data. Since v0.3.0 it also offers:
- an explicit stop option (`select_action_or_stop`);
- exact backward induction (`optimal_sequential_plan`), which exposes the non-concavity of information value (Radner & Stiglitz 1984), where a myopic rule stops too early;
- a goal-relevance diagnostic that scores a substituted question against the original goal.

Helper selection should distinguish domain competence, observed accuracy, perceived trust, answer availability, cost, and incentive. A helper may answer, abstain, or state that a question lies outside their evidence scope. Do not assume the most central or most trusted agent is the best expert.

## Limitation marking and scope mismatch

Evidence scope should be explicit and queryable. If a current question falls outside evidence scope, the default should be to flag or exclude the evidence from direct updating. If applicability is uncertain, represent it as uncertainty and vary it in sensitivity analysis. A probabilistic transfer model could marginalize over applicability, but its likelihoods require a stated derivation or empirical calibration; do not use an unexplained discount factor.

## Provenance-aware dependence

The evidence prototype records evidence as a directed acyclic provenance graph, tracks exact repeated IDs per recipient, and assesses exact categorical scope matches.

`kanm.dependence` (v0.3.0) adds what the ledger alone cannot supply: an explicit conditional probability table per report (primary with possibly asymmetric error rates, relay, or corroborating relay). With it, it computes the exact posterior and the posterior under an independence assumption. Their ratio in log-odds measures overcounting. It is not yet used inside network runs. A provenance graph or equivalent latent evidence model remains the central inference extension. Two reports that share an ancestor should not automatically contribute two independent likelihood terms, but shared ancestry alone does not determine their joint likelihood. Candidate approaches include:

- exact joint Bayesian modeling for small dependency graphs;
- explicit evidence identity with deduplication for literal repeats;
- conditional-dependence models for shared data or testimony;
- approximation methods with an error comparison against exact small cases.

These methods answer different dependence structures. The software should expose which one is used and include a no-dependence baseline. Provenance alone does not reveal whether two sources are statistically dependent; it helps specify the dependency model but does not solve inference automatically.

## Optional research modules

Keep these separate until the core extension is stable:

- epistemic vigilance (plausibility, argument criticism, domain-specific competence);
- strategic or incentive-driven reporting, funding, and institutional power;
- identity-linked credibility bias and epistemic injustice;
- dynamic or temporal networks and changing competence;
- multiple domains and multi-outcome hypotheses;
- corrections, retractions, and memory of earlier evidence.

Each module needs its own literature map and intended-use statement. More realism is not automatically better: every new mechanism adds assumptions and parameters that need evidence or sensitivity analysis.

## Supported use cases to prioritize

1. **Social epistemology and philosophy of science:** theory-driven experiments on trust, inquiry, communication, and collective competence.
2. **Scientific collaboration:** task division, subquestion assignment, shared evidence, publication/reporting norms, and knowledge integration.
3. **Misinformation and online networks:** provenance, repeated claims, uncertainty, and correction dynamics, with explicit limits on predictive claims.
4. **Human-AI research teams:** model assistants as a distinct source type with version, prompt/tool trace, domain-specific reliability, and review obligations; do not treat LLM statements as independent evidence just because multiple agents produce them.
5. **Policy or institutional analysis:** only after calibrating the model to context-specific data and making normative assumptions visible.

These are candidate application domains, not validated use claims.
