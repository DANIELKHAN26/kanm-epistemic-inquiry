# Verification and validation plan

“Fault-free” cannot be guaranteed for a stochastic simulation. The goal is to make assumptions inspectable, prevent known classes of implementation error, reproduce the baseline before extensions, and quantify uncertainty in model outcomes.

## Stage 0 — Source and license audit

- Obtain and archive the original 1.6 source and user manual from the official SourceForge listing.
- Record exact archive checksum, file inventory, compiler/runtime requirements, and license notices.
- Identify which requirements derive from published equations and which derive only from software behavior.
- If source code is copied or translated, preserve applicable GPLv3 notices and seek license advice when needed. If using equations only, still acknowledge the historical work and review legal/institutional release requirements.

## Stage 1 — Formal baseline

- Transcribe all equations, variables, parameter ranges, and assumptions from the primary paper into an implementation-neutral specification.
- Specify scheduler semantics: synchronous/asynchronous events, order of inquiry and communication, edge direction, listen chance, threshold behavior, and how multiple reports in one round combine.
- Specify how trust densities are represented numerically, how they are normalized, and how edge cases are handled.
- Keep the baseline frozen and make all deviations configurable and visible.

## Stage 2 — Code verification and docking

Before describing the code as a faithful port:

- Test single-report Bayesian updates and trust-function updates against hand-computed values.
- Check invariants: normalized probability distributions; no change with no evidence; symmetry under swapping `p` and `¬p`; neutral reliability `0.5` gives no directional evidence; and source trust changes in the direction predicted by the model's single-message table.
- Run deterministic seeded micro-cases, including the published Sherlock Holmes network example where enough initial settings can be recovered. *(v0.3: the printed single-step values in HHO 2020 App. B are reproduced. The Olsson 2011 log contains lines the published formula cannot reproduce, so it is not used as a test.)*
- Compare the new implementation with the original executable on a small set of fixed configurations if the original can be built. If that is not feasible, report the exact limitation and do not imply full docking parity.
- Reproduce at least one published qualitative or quantitative result using its reported parameterization and uncertainty, then state all differences from the legacy schedule.

## Stage 3 — Statistical behavior and sensitivity

- Use independent, recorded random seeds and enough Monte Carlo replications for predeclared Monte Carlo error targets.
- Report uncertainty summaries, not only means; the current batch runner reports sample standard deviation and Monte Carlo standard error. Publication analyses should add intervals appropriate to the design and estimand; log failed or incomplete runs.
- Sweep key uncertain assumptions: source dependence, inquiry accuracy, prior trust, assertion threshold, topology, evidence scope, missing provenance, and cost.
- Use factorial or otherwise justified designs to estimate mechanism effects and interactions; avoid choosing only scenarios that favor the proposed extension.
- Check robustness to alternate outcome measures and parameterization.

## Stage 4 — Extension tests

For provenance, compare approximate methods with exact joint inference on small graphs. Include repeated identical evidence, independent evidence, partial shared ancestry, wrong provenance labels, and missing provenance.

For question selection, test a known finite-state environment where the optimal query is analytically available, then test noisy helpers, costs, question graphs, scope mismatch, and open-ended question proposals. Ensure substitutions preserve an audit trail and are scored against the original goal as well as the active subquestion.

For limitation markers, vary correct, incomplete, noisy, and strategic metadata. Measure whether evidence is overgeneralized, discarded unnecessarily, or used with appropriate uncertainty.

## Stage 5 — Validation by intended use

Follow the ODD documentation standard and select validation methods based on the target audience and use. Docking/reproduction validates correspondence to an earlier model; it does not validate human realism. For claims about people, use suitable behavioral data and comparison models, including out-of-sample checks. For policy use, consult affected users and stakeholders and do not present exploratory simulations as forecasts.

## Outcomes and reporting

At minimum, record:

- truth accuracy and proper scoring rules (Brier score or log score);
- calibration and uncertainty, with limitations of any binning metric stated;
- time and resource cost to answer the goal question;
- distribution of epistemic gains and costs across roles/groups;
- disagreement/dissent and question transitions;
- sensitivity, random seeds, software version, configuration files, and environment;
- negative, null, and counterintuitive outcomes.

Do not treat consensus, speed, or a positive average V-value as sufficient evidence of epistemic success.

## Release gates

| Gate | Requirement before proceeding |
|---|---|
| G1: baseline specified | **Met.** Published equations and the 1.6 order of operations, trust function, new-evidence policies, default trust metadistribution and assertion behaviour are specified and implemented as named options. |
| G2: baseline verified | Unit-level tests pass, and so do equation-level checks against published worked values (Olsson 2013 appendix; HHO 2020 App. B). The scheduler is verified against the 1.6 source. The Vallinder & Olsson (2014) Fig. 1 reproduction is **partial** (at concentration 16 the high-trust end is within 0.05 of the figure; the low-trust end is about 4 times lower, probably because the trust-function shape is unreported; see section 5 of [08](08_v0.3_audit_and_extensions.md)). **Executable docking of the 1.6 model core, met (2026-11-16):**<br>- **Original run:** in double precision, the criterion fixed in advance was missed in 1 of 200 scenarios (credence difference 1.9e-5 against a tolerance of 1e-5). The stochastic test passed (max \|z\| 2.75).<br>- **Diagnosis:** the miss was traced to Laputa's single-precision trust arithmetic.<br>- **Confirmatory run, declared in advance:** with that arithmetic emulated, 200 of 200 new scenarios passed (credence within 1.5e-14).<br>See the [docking report](../docking/laputa_1_6/README.md#status-and-results-2026-11-16). **Published-result reproduction:** remains partial. |
| G3: extension defined | Every new mechanism has an equation/policy, parameter meaning, and null ablation. |
| G4: robustness checked | Main result survives justified uncertainty and sensitivity analyses, or the dependency is reported. |
| G5: claims scoped | Documentation distinguishes mathematical behavior, simulation findings, empirical validity, and application claims. |
