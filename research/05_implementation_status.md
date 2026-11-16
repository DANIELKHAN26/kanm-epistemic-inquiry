# Implementation status and contribution boundaries (v0.5.0 development)

## What is implemented

| Component | Implemented behavior | Verification | Current boundary |
|---|---|---|---|
| Binary kernel (`kanm.bayes`) | Published credence and trust updates. Log-odds credences; trust held as normalised log-weights (no irreversible underflow, D16). Moment-preserving discretised Beta trust priors (exact cell masses via the incomplete beta function; no atoms at 0 or 1). | Formula and normalisation checks; grid convergence; symmetry; agreement of log-odds and probability forms; published worked values (Olsson 2013 appendix; HHO 2020 App. B) | A continuous density is approximated on a grid; Laputa 1.6 used 49 points |
| Exact joint state (`ExactJointState`) | Exact posterior under the finite trust grid for `p` and each source reliability; order invariant; O(sources × grid points) state with no growth in report history | Brute-force enumeration of the full discretized joint; all-permutation order invariance; equality with Laputa when each source reports once; predictive normalisation | The Beta prior is grid-approximated; exactness is conditional on the agent's model and does not represent the network's true data-generating process |
| Network simulator (`kanm.model`) | Rules: `laputa`, `exact_joint`. Schedules: `simultaneous` (default; the order of operations in the 1.6 source), `inquiry_then_messages` (v0.2) and `laputa_1_6` (`Society::evolve` with the 1.6 new-evidence policies). Assertion rules: `published` (decided in log-odds, mirror-symmetric) and `laputa_1_6`. Trust models: `beta_grid` and `laputa_1_6` (49-node piecewise-linear). Per-agent and per-edge random streams (common random numbers). Metrics: V-value, Brier, log score, accuracy, undecided fraction, dispersion, spread, final trust, optional trajectory | Hand-computed one-step cases; schedule semantics; input-order invariance; common random numbers; rule agreement for single reports; `laputa_1_6` schedule equals `simultaneous` under policy "none" bit for bit; 1.6 trust function against independent restatements; mirror symmetry; regressions D1–D24 (see [08](08_v0.3_audit_and_extensions.md)) | `laputa_1_6` options docked against the compiled 1.6 core (see the [docking report](../docking/laputa_1_6/README.md#status-and-results-2026-11-16)): logic reproduced exactly. In KANM's default double precision, results differ from Laputa's single-precision trust by up to about 5e-5 over 25 steps |
| Networks (`kanm.networks`) | Seeded complete, empty, cycle, star, Erdős–Rényi (directed or undirected) and Watts–Strogatz generators; density and clustering | Edge counts, seeding, limiting cases (lattice clustering 0.5 at k = 4; complete graph clustering 1) | HHO's fixed 36 networks are not included |
| Experiments (`kanm.batch`) | Batches with MCSE; paired comparisons on common seeds; full-factorial sweeps; undefined values counted, not averaged | Seed bookkeeping; paired identity of inquiry events; factorial completeness | MCSE quantifies simulation error only |
| Question history (`kanm.inquiry.QuestionGraph`) | Immutable goal; transitions with rationale and mapping; substitution and abandonment | Root retention; required rationale; branch switching | No automatic question generation |
| Planner (`kanm.inquiry`) | Information gain about any target partition; goal-relevance (scope-drift) diagnostic; EVSI; myopic stop rule; exact finite-horizon Bayes-optimal plan | Analytic information values; data-processing bound; horizon-1 equivalence with the myopic rule; demonstrated non-concavity case; monotone value in horizon; value of perfect information | Supplied likelihoods and utilities; the exact planner is exponential in horizon |
| Evidence ledger (`kanm.evidence`) | Provenance DAG, exact repeats per recipient, exact categorical scope | Duplicate, ancestry, scope tests | Scope is not converted to likelihoods |
| Dependence inference (`kanm.dependence`) | Bayesian network of binary reports with explicit tables (primary, asymmetric primary, relay, corroborating relay); exact posterior; independence comparator; overcounting ratio; construction from the ledger with latent items | Bayes rule; independence equality; literal repeats count once; uninformative relay; hand-computed partial dependence; random DAGs against unpruned enumeration | At most 20 latent ancestors; multi-parent nodes need explicit tables; not yet coupled to network runs |
| Reliability learning on public data (`kanm.reliability`, `kanm.crowd_data`) | Majority vote; Laputa rule online with randomized report order; exact-within-item filtering (ADF); one- and two-coin Dawid–Skene MAP EM with fixed or estimated class prevalence; accuracy, Brier, log loss, confidence and positive-probability ECE; prequential reliability scores | Parser counts and majority-vote checks; checksummed data; single-item exactness; prior-only prequential baseline; randomized-order reproducibility | Five binary data sets; single reasoner, not a network; symmetric reliability only for the KANM rules; offline EM sees all data and can reach local optima; see [11](11_deep_research_and_v0.5_validation.md) |
| CLI (`kanm-run`) | Explicit or generated agents and networks; `schema_version`; strict JSON (duplicate keys, NaN/Infinity, unknown keys rejected); `--trajectory`; `--compare KEY=V1,V2` | Validation and end-to-end compare tests | No YAML or schema file; configs are validated by code |

## Supported claims

The 158 tests establish that the listed mathematical cases, invariants and published worked values hold in this code. They do **not** establish:
- correspondence with full runs of any Laputa version;
- human behavioural validity;
- the generality of any simulation result.

The Vallinder & Olsson (2014) reproduction is **partial** (section 5 of [08](08_v0.3_audit_and_extensions.md)).

## Recommended order of research work

1. **Docking (done for the 1.6 model core, 2026-11-16).** Remaining:
   - dock the GUI batch-society generator, which is not in the harness;
   - extend the reproduction of a published result beyond partial.
2. **Position the exact-joint result.**
   - Compare matched settings against SBIC, online Dawid–Skene and current crowd-aggregation software; online inference is established prior art.
   - Analyze the status of learned reliability against Merdes et al. (2021) and the newer calibration literature.
   - Run preregistered sensitivity analyses of the paired study.
3. **Couple provenance to the network.** Start with an evidence-sharing variant (messages forward primary inquiry results with IDs), where deduplication is exact, before modelling assertions with partial dependence.
4. **Compare descriptive and normative agents** against human data before any behavioural claim.
5. **Application work** only with context data and stakeholder input.

## Provisional contribution statement

The software contribution is a tested, documented research framework. It separates:
- the published Laputa rule;
- the behaviour of the 1.6 code;
- an exact Bayesian reference under the same assumptions;

and it adds provenance-dependence inference, partition-based question planning and paired experimental tooling.

The candidate scientific contribution remains provisional: a finite-grid exact-joint update and its cross-item assumed-density approximation, and their consequences for collective outcomes. Streaming Bayesian crowd inference already has prior art. The diagnosis that Laputa's update is non-exact and order-dependent is **prior art** (Merdes, von Sydow & Hahn 2021; Hahn, Merdes & von Sydow 2018). See the [v0.5 deep-research update](11_deep_research_and_v0.5_validation.md) before making novelty claims.
