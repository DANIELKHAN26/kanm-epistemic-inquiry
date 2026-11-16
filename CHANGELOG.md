# Changelog

## 0.5.0 — 2026-11-16

Research-quality and usability update; this remains a pre-alpha research package.

### Fixed and improved
- Randomized within-item crowd-report order with its own reproducible seed stream. The v0.4 sequential result table used lexicographic worker IDs and is now explicitly superseded in the research report.
- Added prequential exact-ADF worker-reliability scores, scored before each item's labels are processed, plus a fixed-prior baseline and paired differences against majority vote.
- Added one-coin and two-coin Dawid–Skene sensitivity runs with estimated class prevalence. The report states the Beta prior and warns about EM local optima.
- Added positive-probability ECE beside confidence ECE, Brier and log loss; stricter validation for crowd records, Bluebirds parsing, metric bins and EM settings.
- Downloads are written atomically after checksum verification and refuse symlink targets. The Laputa docking builder now refuses to overwrite existing output and builds in a private staging directory.
- Added benchmark CLI controls for trust-grid resolution and prior parameters. The new five-dataset report records the 31-point grid used for the randomized 10-order reanalysis.
- Added a prior-art correction: streaming Bayesian aggregation predates KANM. Novelty remains unestablished pending a matched comparison.
- Updated the model card, research references, limitations and public-data report to distinguish simulated behavior, mathematical results, and empirical evidence.
- Regression suite: 158 tests pass.

### Scope
- The crowd benchmark remains a small binary, single-reasoner study; it does not validate network dynamics or human behavior.
- The finite-grid exact posterior is exact for the discretized model. Cross-item reliability propagation is ADF; cross-agent evidence dependence is not automatically modeled.
- The 1.6 docking has a deterministic double-precision tolerance miss and a separate float32 trust expectation miss; see the full report.

## 0.4.0 — 2026-11-15 to 2026-11-16

Outputs for a given seed are unchanged from 0.3.0, except in two kinds of run:
- very long runs in which trust weights used to underflow (D16);
- runs in which a sender's credence equals an assertion threshold exactly, or the threshold is 0 or 1 (D17).

### Fixed (second independent review; see research/08 §1)
- **Trust weights no longer underflow irreversibly (D16).** Weights are held in log space. Previously, after a long run of agreeing reports followed by a long contrary run, a source's estimated reliability could end on the wrong side of 0.5. This affected both update rules.
- **The published assertion rule is exactly mirror-symmetric (D17).** It is now decided in log-odds; rounded probabilities had broken the p/¬p symmetry at threshold ties and at `T ∈ {0, 1}`.
- Other fixes:
  - `paired_comparison` validates `seed_start`;
  - abandoned questions stay abandoned;
  - action selection breaks ties by ID, not rounding noise;
  - `from_ledger` rejects unknown table keys;
  - colliding `final_trust` keys are detected;
  - huge hand-built trust weights no longer overflow;
  - tests skipped when run directly now run (D18–D24).
- Docs: the HHO App. B worked value no longer cited as evidence for the schedule; it does not discriminate between schedules.

### Added
- `trust_model="laputa_1_6"` and `kanm.laputa16.Laputa16Trust`: the 49-node piecewise-linear trust function of the Laputa 1.6 source, with its presets and its default batch trust prior `Beta(1 + 3w, 4 − 3w)`. That default is unnormalised: its first-use expected trust is biased by up to −0.025.
- `schedule="laputa_1_6"` with `evidence_policy` (`none`, `inquiry`, `any`) and `count_prior_as_evidence`: a restatement of the 1.6 step, including its new-evidence policies and their agent-order dependence.
- `docking/laputa_1_6/`: a headless harness that compiles the unmodified, checksum-verified 1.6 model core, and `compare.py`, with deterministic and stochastic acceptance criteria fixed in advance.
- **Docking results (2026-11-16):**
  - **Stochastic:** passed.
  - **Deterministic:** missed in 1 of 200 scenarios (1.9e-5 against a tolerance of 1e-5); counts identical.
  - **Diagnosis:** traced to Laputa's single-precision trust arithmetic.
  - **Confirmatory run, declared in advance:** passed 200 of 200, with that precision emulated by `docking/laputa_1_6/float32.py` (`compare.py --emulate-float32`).
  - Details in `docking/laputa_1_6/README.md`.
- `kanm.reliability` and `kanm.crowd_data`: tests of reliability learning on five public crowdsourcing data sets with gold answers. Methods: majority vote; the Laputa rule online; exact-within-item filtering; Dawid–Skene. Data are downloaded and checksum-verified, not redistributed. Design and results are in research/10.
- `examples/crowd_benchmark.py`; `examples/crowd_posthoc.py`, for post hoc diagnostics labelled as such; research/10, with design fixed before the run and results added after; reference register entries 63–70.

## 0.3.0 — 2026-11-15

This release changes outputs for a given seed: it uses new per-agent and per-edge random streams and a new default schedule. Runs record `model_variant` and `kanm_version`.

### Fixed (audit; see research/08 §1)
- Credences no longer freeze at exactly 0 or 1 through floating-point saturation; they are held as log-odds.
- Certain-reliability trust priors (mean 0 or 1), which could crash a run mid-way, are rejected up front.
- `accuracy_at_half` no longer depends on which proposition is labelled true; ties are reported as `undecided_fraction`.
- `NaN`/`Infinity` JSON is rejected; output JSON is strict.
- Random draws no longer depend on beliefs, so model variants share random inputs (common random numbers).
- Bounded, integer trust grids; `QuestionGraph.add_child` validates `activate` before use; an unchanged prior is reported exactly.
- **Trust discretisation (inherited from v0.2.0):** grids no longer place atoms at r = 0 and 1, which let trust collapse to certainty and crash long runs. Beta priors are now discretised with exact cell masses at exact cell conditional means, so the prior mean is exact. Previously J- and U-shaped priors were biased, e.g. a requested 0.99 became 0.85.
- Dependence inference is computed in log space; log-odds shifts no longer vanish when posteriors saturate; faster and scale-free planner guards; stricter CLI and batch validation (independent review findings D9–D15).

### Added
- `schedule="simultaneous"` (new default): the order of operations of the Laputa 1.6 source, verified by reading the archived code. `inquiry_then_messages` keeps the v0.2 order.
- `assertion_rule="laputa_1_6"`: the 1.6 code's threshold behaviour, which differs from the published rule.
- `update_rule="exact_joint"` and `ExactJointState`: a finite-grid exact joint posterior under Laputa's assumptions, with storage that does not grow with report history for fixed grid resolution.
- `kanm.dependence`: exact Bayesian-network inference for dependent reports, asymmetric error rates, relays, an independence comparator and the overcounting ratio; construction from the evidence ledger.
- Planner: information about any target partition, `question_relevance` (scope drift), `refines`, `select_action_or_stop`, and exact `optimal_sequential_plan`.
- Metrics: V-value, log score, undecided fraction, dispersion, spread, final trust, optional trajectory.
- `kanm.networks` generators; `paired_comparison` and `sweep`; CLI `--compare`, `--trajectory`, generated networks and `schema_version`.
- Published-value tests (Olsson 2013; HHO 2020), a partial reproduction of Vallinder & Olsson (2014, Fig. 1), and an exploratory paired study.
- Research documents 08 (audit and justification) and 09 (study results); a reference register corrected and extended to 62 verified entries.


## 0.2.0 — 2026-11-15

- Added finite-state decision value of sample information alongside expected information gain.
- Added auditable question histories, substitution mappings, and evidence provenance/scope bookkeeping.
- Added seeded batch summaries with complete configurations and runtime metadata.
- Added strict CLI validation, publication metadata, and a Hugging Face model card that identifies the project as an agent-based simulator without trained weights.
- Kept the network runner explicitly labeled as a synchronous Laputa-style variant pending legacy docking.

This is a research development version, not a validated or faithful historical port.
