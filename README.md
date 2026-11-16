# KANM: Question-Directed Collaborative Inquiry

**Source repository:** [github.com/DANIELKHAN26/kanm-epistemic-inquiry](https://github.com/DANIELKHAN26/kanm-epistemic-inquiry)

KANM is research software for studying how people or artificial agents divide inquiry, assess testimony, and update beliefs in networks. It reimplements the Bayesian network model behind the historical Laputa simulator (Olsson and Angere) and adds:
- an exact Bayesian reference rule;
- provenance-aware dependence inference;
- question planning based on partitions;
- reproducible paired experiments;
- a faithful option for the Laputa 1.6 trust function and step, with a harness for docking against the original C++ core;
- tests of reliability learning on public crowdsourcing data with gold answers.

**This is an agent-based simulation package, not a trained machine-learning model.** It contains no learned weights or training data. Its outputs are conditional on the supplied assumptions and configuration.

## What KANM can do

| Capability | Module | Status |
|---|---|---|
| Laputa's published credence and trust updates, numerically stable (log-odds) | `kanm.bayes`, `kanm.model` | Equation-level checks against published worked values pass |
| Order of operations of the Laputa 1.6 source (simultaneous updates from start-of-step credences) | `schedule="simultaneous"` (default) | Verified against the archived source; not numerically docked |
| Published assertion rule **and** the 1.6 code's behaviour, which differs | `assertion_rule="published" \| "laputa_1_6"` | Both implemented; the published rule is decided in log-odds (exactly mirror-symmetric) |
| The 1.6 trust function (49 nodes, piecewise linear) and step, including its new-evidence policies | `trust_model="laputa_1_6"`, `schedule="laputa_1_6"`, `kanm.laputa16` | Checked against independent restatements; equal to `simultaneous` when no policy is set |
| Docking harness for the original C++ model core | `docking/laputa_1_6/` | Run on 200 deterministic and 5 stochastic scenarios. Logic reproduced. Fixed numerical tolerance missed in 1 of 200 scenarios, a miss traced to Laputa's single-precision trust. A confirmatory run with that precision emulated, declared in advance, passed 200 of 200; see the [docking report](docking/laputa_1_6/README.md#status-and-results-2026-11-16) |
| Reliability learning on public data (five crowdsourcing data sets with gold answers) | `kanm.reliability`, `kanm.crowd_data`, `examples/crowd_benchmark.py` | Checksum-verified parsers, randomized report-order study, prequential reliability metrics, class-prevalence sensitivity; results and limits in [research/11](research/11_deep_research_and_v0.5_validation.md) |
| Exact joint posterior for a single inquirer's finite reliability-grid model | `update_rule="exact_joint"`, `ExactJointState` | Checked against brute-force enumeration; memory is O(sources × grid points), independent of history length. Cross-agent dependence is not represented automatically. |
| Exact inference for dependent reports in a provenance graph, with an independence comparator that measures overcounting | `kanm.dependence` | Checked against unpruned enumeration on random DAGs |
| Questions as partitions; information about the original goal versus a substituted question; EVSI; stop rule; exact finite-horizon planning | `kanm.inquiry` | Analytic cases, including a demonstrated case where the myopic rule fails |
| Seeded network generators (complete, cycle, star, Erdős–Rényi, Watts–Strogatz) | `kanm.networks` | Limiting-case tests |
| Paired comparisons on common random numbers, factorial sweeps, Monte Carlo standard errors | `kanm.batch`, `kanm-run --compare` | Tested |

## Research status

- **The diagnosis is prior art.** Laputa's update keeps only marginal credence and marginal trust. It is exact for one report per source, but an order-dependent approximation once sources repeat. This is already published (Merdes, von Sydow & Hahn 2021; Hahn, Merdes & von Sydow 2018).
- **KANM's candidate contribution** is narrower and provisional: an exact joint update on a finite reliability grid with no growth in stored history, plus a cross-item assumed-density approximation. Online crowd aggregation already has prior art; see the [prior-work audit](research/02_prior_work_and_gap.md). A matched comparison against SBIC and other streaming methods is still missing.
- **Exploratory results.** Swapping the exact filter in changes collective outcomes: under one illustrative design the exact rule is *more cautious and tracks the truth less well in sparse networks*. See [the study](research/09_factorization_study_results.md). Novelty is provisional pending a systematic search.
- **Docking.** The stochastic test passed. The original deterministic comparison missed its 1e-5 criterion in 1 of 200 scenarios (maximum credence difference 1.94e-5); message and inquiry counts matched in all 200. Float32 emulation passed the scenario criterion on 200 new scenarios, with maximum credence difference 1.5e-14, but trust differed by up to 3e-8 against a separate pre-run expectation of 5e-10. KANM's double-precision results differ from Laputa's single-precision arithmetic. This does not dock a published paper's full simulation settings. See the [full docking record](docking/laputa_1_6/README.md#status-and-results-2026-11-16).
- **Public data.** The updated five-set study randomizes within-item arrival order, adds prequential worker-reliability scoring, and compares fixed versus estimated class prevalence. The exact ADF learner improves the prequential Brier score over its prior on four of five datasets, but underperforms on Bluebirds; a two-coin Dawid–Skene reference helps on Bluebirds and imbalanced Product. This is an exploratory benchmark with only five old binary datasets, no network agents, and no human-behaviour validation. See [research/11](research/11_deep_research_and_v0.5_validation.md).
- **Validity.** Nothing here is validated against human data. Do not use KANM as a predictor of behaviour or policy outcomes.

## Start here

- [v0.3–v0.4 audit history and justified extensions](research/08_v0.3_audit_and_extensions.md)
- [Exploratory results: factorised vs exact updating](research/09_factorization_study_results.md)
- [Public-data test of reliability learning](research/10_public_data_benchmark.md)
- [Deep research and v0.5 validation update](research/11_deep_research_and_v0.5_validation.md)
- [Docking against the Laputa 1.6 source](docking/laputa_1_6/README.md)
- [Research scope and method](research/00_scope_and_method.md)
- [Laputa baseline](research/01_laputa_baseline.md)
- [Prior work and candidate contribution](research/02_prior_work_and_gap.md)
- [Proposed model](research/03_proposed_model.md)
- [Verification and validation plan](research/04_validation_plan.md)
- [Implementation status](research/05_implementation_status.md)
- [ODD model description](research/06_odd_model_description.md)
- [Publication readiness](research/07_publication_path.md)
- [Verified source register](research/references.md)

The preliminary Laputa note is kept under [archive](archive/README.md) only as research history. It is not evidence and should not be cited.

## Install and run

Requires Python 3.11 or newer; no runtime dependencies.

```sh
python -m pip install -e .
kanm-run examples/minimal_network.json                      # one run
kanm-run examples/generated_network.json --replicates 100   # Monte Carlo batch with MCSE
kanm-run examples/generated_network.json --replicates 100 \
    --compare update_rule=laputa,exact_joint                # paired comparison on common seeds
kanm-run examples/generated_network.json --trajectory       # mean credence in the truth per step
```

Configuration keys:
- `truth`, `steps`, `seed`, `trust_points`;
- `update_rule` (`laputa` | `exact_joint`);
- `schedule` (`simultaneous` | `inquiry_then_messages` | `laputa_1_6`);
- `assertion_rule` (`published` | `laputa_1_6`);
- `trust_model` (`beta_grid` | `laputa_1_6`);
- `evidence_policy` (`none` | `inquiry` | `any`) and `count_prior_as_evidence`, both for `schedule="laputa_1_6"` only;
- per-agent `inquiry_trust_values` and per-edge `trust_values` (49 node values), for `trust_model="laputa_1_6"` only;
- `publish_only_after_inquiry`;
- agents, either an explicit `agents` list or `n_agents` + `agent_defaults`;
- edges, either an explicit `edges` list or `network` + `edge_defaults`;
- optional `schema_version: 1` and `description`.

Duplicate keys, unknown keys and `NaN`/`Infinity` are rejected. With `--replicates`, every replicate uses the same generated network (the `network` seed is fixed); vary the network seed explicitly to sample networks.

Demonstrations and studies:

```sh
python -m examples.planning_demo                       # EIG vs EVSI, stopping, exact planning, scope drift
python -m examples.provenance_demo                     # repeats, scope, exact dependence-aware inference
python -m examples.reproduce_vallinder_olsson_2014 50  # partial reproduction (takes minutes)
python -m examples.study_factorization 50              # paired exact-vs-Laputa study (takes minutes)
python -m examples.crowd_benchmark DATA_DIR --download # public-data reliability test (downloads ~3 MB)
```

## Python API sketch

```python
from kanm.model import Simulation
from kanm.networks import agent_ids, agents_from_ids, edges_from_pairs, watts_strogatz
from kanm.batch import paired_comparison

ids = agent_ids(12)
agents = agents_from_ids(ids, inquiry_chance=0.4, inquiry_accuracy=0.7)
edges = edges_from_pairs(watts_strogatz(ids, 4, 0.1, seed=1), listen_chance=0.5)
report = paired_comparison(
    agents, edges,
    {"laputa": {"update_rule": "laputa"}, "exact": {"update_rule": "exact_joint"}},
    replicates=100, steps=50,
)
print(report["paired_difference_vs_baseline"]["exact"]["veritistic_value"])
```

## Verify

```sh
python -m unittest discover -s tests -v
```

The 158 tests cover:
- equation-level cases and published worked values;
- exactness against brute-force enumeration;
- order invariance and common random numbers;
- schedule and assertion semantics;
- regression tests for every defect found in the v0.3 audit and the second review (D16–D24);
- the Laputa 1.6 trust function and step, checked against independent formulas;
- the single-precision emulation used to diagnose docking residuals (skipped without numpy);
- the reliability-learning methods, on synthetic data with known answers;
- public-data validation guards, report-order randomization and prequential reliability scoring;
- dependence inference, planning optimality cases, network generators and CLI validation.

Passing them does not establish empirical validity or correspondence to full runs of the historical software.

## Attribution

KANM is a new implementation based on published equations and descriptions. The Laputa 1.6 source (GPLv3) was read to specify historical behaviour, which KANM restates in its own code. The optional docking harness compiles the user's own unmodified copy of the 1.6 model files; see [NOTICE](NOTICE.md). We acknowledge Erik J. Olsson and Staffan Angere and cite the historical work in the [reference register](research/references.md). The software record is on [SourceForge](https://sourceforge.net/projects/epistemenet/).

## License and citation

GPL-3.0-only. See [LICENSE](LICENSE), [NOTICE](NOTICE.md) and [CITATION.cff](CITATION.cff).
