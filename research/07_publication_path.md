# Publication path and readiness

This is planning guidance, not a venue ranking or a claim that KANM is currently publishable.

## Separate the software claim from the scientific claim

| Publication type | Main claim | Evidence required first | Current readiness |
|---|---|---|---|
| Research-software article (for example, JOSS) | The package is a useful, maintained research tool that researchers can install, understand, verify, and cite. | Feature-complete documented package; explicit open-source license and provenance; public Git repository and issue workflow; tests/CI; iterative public development; evidence of actual research use; comparison with related software. | **Not ready.** The v0.5 development snapshot has license metadata, attribution, 158 passing tests, published-value checks, a partial reproduction and a limited public-data benchmark. It lacks the required six-month public development history and evidence of research use. The `laputa_1_6` options are docked against the compiled 1.6 model core, with the original deterministic tolerance miss and the separate precision-emulation caveat recorded. Research modules remain separate. |
| Model/methods article | A specified mechanism or combination changes conclusions about collaborative inquiry under a justified context. | Docked baseline; formal extension equations and ablations; predeclared estimands; adequate Monte Carlo precision; robustness/sensitivity; comparison against neighboring models; ideally empirical data if claims describe people or institutions. | **Not ready, but a candidate exists.**
- **The candidate:** a finite-grid exact-joint update and its cross-item assumed-density approximation under stated assumptions, and their effect on collective outcomes ([08](08_v0.3_audit_and_extensions.md) §3; [09](09_factorization_study_results.md); [11](11_deep_research_and_v0.5_validation.md)). Online crowd aggregation is established prior art; no novelty claim is justified until a matched comparison is completed.
- **Required first:**
  - a systematic prior-art search;
  - engagement with Merdes et al. (2021);
  - predeclared sensitivity analyses;
  - a calibration study.
  - (Docking of the 1.6 model core is done.)
- **No established research result yet.** |
| Application paper | A context-specific finding about a real research group, platform, institution, or policy setting. | Context evidence, ethically appropriate data, calibrated assumptions, stakeholder/user input where relevant, out-of-sample or other context-appropriate validation, and limits on generalization. | **Not ready.** No empirical dataset or target application has been supplied. |

## JOSS-specific constraints to plan around

JOSS is a plausible future home for a software contribution, not a shortcut to establish research impact. Its current author guidance calls for an open-source software project with an obvious research application, an accessible public repository, feature completeness, and evidence that the software has been used in research. Its 2026 screening guidance also expects sustained public development history (more than six months) and iterative improvement, not a one-time repository dump. JOSS currently requires an AI-use disclosure for authors who used generative AI in software, documentation, or paper preparation. Recheck the live guidance at submission time because criteria can change: [JOSS submission guide](https://joss.readthedocs.io/en/latest/submitting.html), [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html).

## Recommended sequence

1. Publish the software prototype with its GPL-3.0-only license, author citation, attribution notice, tests, CI, and explicit validation limits. Treat the first public commit as the beginning of the maintenance history, not as evidence of research impact.
2. ~~Finish historical source/manual audit and implement a docking case.~~ Done for the 1.6 model core (2026-11-16). Only `schedule="laputa_1_6"` with `trust_model="laputa_1_6"` and `assertion_rule="laputa_1_6"` is docked. Other configurations remain Laputa-style variants.
3. Complete the question-planning and evidence model integration, including exact repeated-evidence cases and a precisely specified small-graph joint-dependence model. Compare normative inference with behaviorally supported alternatives.
4. Conduct a balanced simulation study with null ablations, parameter sensitivity, predeclared outcome measures, Monte Carlo precision targets, and negative/adverse outcomes.
5. If real human or institutional claims are intended, collect or identify suitable data and validate against it; simulation alone cannot establish those claims.
6. Maintain the public software through iterative releases and issues, document actual research use, and then decide whether a JOSS article is supported. Write a separate model/application paper for scientific results if the study justifies one.

## Why not promise “top tier” at this stage

Venue reputation cannot compensate for an unverified baseline, unclear novelty, weak identification of mechanisms, or unsupported application claims. The highest-value next step is evidence that the proposed model answers a clearly bounded question better than its baseline and alternatives. Venue selection should follow the resulting contribution and its readership.
