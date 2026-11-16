# Deep research and validation update (v0.5 development)

**Date:** 16 November 2026. **Purpose:** re-audit the post-v0.4 changes, check current prior art, make the public-data benchmark more defensible, and record a reproducible reanalysis. This is a development report, not a preregistered study or a claim of human-behaviour validation.

## 1. What KANM can currently do

KANM is research software for simulation and analysis. It contains a Laputa-style binary belief/trust network, an exact-joint reference update under a discretized symmetric-reliability model, a dependence-aware evidence-network module, and a finite-state question planner. The parts are not yet one integrated end-to-end research agent: provenance-aware inference is not wired into the social-network message loop, and the planner is not learning its action likelihoods from data.

The current evidence supports these limited claims:

- The historical Laputa 1.6 core has an executable docking harness. The original double-precision comparison missed its stated deterministic tolerance in 1 of 200 scenarios. Float32 emulation passed the scenario criterion on 200 new scenarios, but missed its separate pre-run trust-expectation target (3e-8 observed versus 5e-10 expected). Preserve both facts; the harness does not reproduce a published paper's full simulation.
- The exact-joint inference is exact for the finite reliability grid and the stated conditional-independence model. The trust prior itself approximates a continuous Beta density. Exactness inside one reasoner's model does not remove dependence between agents whose testimony shares upstream evidence.
- Public crowd labels can test aggregation and reliability learning. They cannot validate the social-network dynamics, question planner, human belief revision, or policy effects.

## 2. Prior-art update and the actual research gap

Online inference is established prior art. Zhu, Xu & Yan describe an online Dawid–Skene algorithm with a stationary-point convergence result ([arXiv:1512.02393](https://arxiv.org/abs/1512.02393)); Manino, Tran-Thanh & Jennings describe SBIC, a streaming Bayesian method for binary crowd classification with stated asymptotic guarantees ([arXiv:1911.05712](https://arxiv.org/abs/1911.05712)). Therefore KANM must not claim novelty for “online Bayesian aggregation,” “streaming reliability,” or online learning generally.

The narrower open question is whether this particular finite-grid exact-joint update, followed by cross-item assumed-density filtering, gives a useful accuracy/calibration/compute trade-off against established streaming Bayesian methods and offline aggregators under matched assumptions. It has not yet been benchmarked against SBIC, spectral initializers, Crowd-Kit, or the full set of methods in current crowd-aggregation papers. A systematic literature and software comparison is still required before novelty or state-of-the-art claims.

A recent 2026 arXiv preprint reports calibration analyses over 39 human crowdsourcing datasets and LLM-annotated tasks. It reports that common aggregation methods can be miscalibrated, and proves calibration for a Bayesian aggregator under its own data-generating model. The paper text says both 39 and 40 in different places; its result is treated here as a strong motivation, not a settled benchmark fact ([Singer et al., 2026](https://arxiv.org/abs/2610.16815)).

## 3. Code and protocol changes made in this audit

| Change | Why it matters | Check |
|---|---|---|
| Randomize report arrival order within each item for every benchmark replicate, using a separate deterministic seed stream from item order | Sequential Laputa updates are order dependent. Lexicographic worker IDs were an arbitrary hidden schedule in the prior report. | Seeded repeatability and invalid-permutation tests |
| Add exact-ADF prequential worker-reliability scoring | Final reliability estimates compared with all gold-scored reports are in-sample. The new measure predicts worker correctness at the start of each item using earlier items only; gold is scored after prediction and never used by the learner. | Hand-checked temporal protocol and report-count tests |
| Compare online reliability predictions to the fixed Beta(2,1) prior baseline | A reliability score without a baseline cannot establish learning value. | Per-report MAE/Brier deltas; worker-macro MAE also reported |
| Add one-coin and two-coin Dawid–Skene runs with estimated class prevalence | Fixed prevalence 1/2 can fail on imbalanced tasks; the public Product set has a 12.2% positive rate. | Class prior, convergence, accuracy, Brier, log loss, and calibration reported |
| Add positive-probability ECE alongside confidence ECE, Brier and log loss | Binary confidence ECE can hide calibration differences in the positive-probability scale; ECE is bin-dependent and remains descriptive. | Unit tests and cross-check against published Snow RTE calibration values |
| Validate label IDs, integer classes, duplicate item-worker pairs, parser structure, score bins and EM settings | Malformed data and invalid method options should fail explicitly, not silently alter a result. | Regression tests |
| Make public-data downloads checksum-verified, HTTPS where available, atomic, and symlink-safe | Avoid partial/corrupt downloads and unintended writes through a pre-existing symlink. | All downloaded archives/files matched their recorded SHA-256 values |
| Make the Laputa docking builder refuse an existing output path and compile in a private staging directory | The previous `rm -rf "$OUT"` could destroy a caller's build directory. | Sentinel-file regression test verifies existing output remains untouched |
| Expose trust prior and grid resolution in the benchmark CLI; report paired differences against majority vote | Makes runtime/precision a visible trade-off and the main baseline comparison easier to interpret. | CLI run and deterministic benchmark test |

## 4. Public-data reanalysis

### Protocol

- Data: the same five public binary sets in [research/10](10_public_data_benchmark.md). Files were downloaded to `/tmp/kanm-public-data` and verified against the SHA-256 values in `kanm/crowd_data.py`; data are not added to the repository.
- Exact ADF and both Laputa schedules: 10 randomized item orders, seed 0. The sequential rule also receives 10 reproducible within-item report-order permutations from a separate seed stream (1). The corresponding permutations are shared between methods.
- Reliability prior: moment-preserving discretized Beta(2,1), 31 trust points. This lower resolution was chosen to make the five-dataset 10-order run tractable; do not merge these scores with the prior 101-point table as though they were a direct grid-convergence check.
- Offline Dawid–Skene baselines see all items. They are comparison references with an information advantage, not online competitors. Fixed-class-prior results use 1/2. Estimated-class-prior variants use a separate Beta(1,1) prior and one majority-vote initialization; EM can reach local optima.
- Item-aggregation accuracy is scored on gold after the item's labels are aggregated. This is not a prediction-before-label task. The worker-reliability prequential score is stricter: before each item is processed, reliability predictions use only earlier items; the gold truth is only used to score worker correctness.
- SD is across randomized orders only. It is not a confidence interval over a population of tasks or annotators. No significance tests are claimed.

### Accuracy and prior-sensitive reference results

| Dataset | Gold positive share | Majority vote accuracy | Laputa sequential accuracy | Exact ADF accuracy | Two-coin DS, fixed π=0.5 | Two-coin DS, estimated π (estimate) | Exact ADF prequential Brier Δ vs fixed prior |
|---|---:|---:|---:|---:|---:|---:|---:|
| Snow RTE | 0.500 | 0.897 | 0.923 ± 0.004 | 0.922 ± 0.001 | 0.928 | 0.926 (0.482) | −0.026 |
| Snow TEMP | 0.439 | 0.936 | 0.941 ± 0.003 | 0.941 ± 0.003 | 0.942 | 0.942 (0.455) | −0.044 |
| Bluebirds | 0.444 | 0.759 | 0.699 ± 0.050 | 0.686 ± 0.074 | 0.898 | 0.898 (0.437) | +0.009 |
| Zheng Product | 0.122 | 0.897 | 0.927 ± 0.001 | 0.927 ± 0.001 | 0.837 | 0.939 (0.107) | −0.062 |
| Zheng Sentiment | 0.472 | 0.934 | 0.949 ± 0.003 | 0.950 ± 0.003 | 0.960 | 0.960 (0.451) | −0.037 |

Accuracy SD is across ten item orders (and, for the sequential variant, randomized report arrival orders). The prequential Brier delta is online exact-ADF worker-correctness Brier minus the constant-prior baseline; negative is better. The positive delta on Bluebirds means this reliability predictor did worse than its prior under Brier score, despite its small MAE improvement. This agrees with the primary failure signal: a symmetric reliability model is not adequate for every crowd.

On Zheng Product, estimating class prevalence changes two-coin Dawid–Skene accuracy from 0.837 to 0.939; the estimated positive prevalence is 0.107 versus gold prevalence 0.122. This is a sensitivity result, not a universal improvement. On Bluebirds, the two-coin model is much stronger than either symmetric model. On Snow RTE and Sentiment, the fixed 1/2 prevalence is not harmful to two-coin accuracy. The one-coin estimated-prior Bluebirds run collapses to a poor local solution (estimated prevalence 0.029); this illustrates the known non-convexity and initialization risk of EM rather than a justified recommendation to estimate prevalence blindly.

The exact ADF one-coin output improves on majority-vote accuracy on four sets, but is worse on Bluebirds. Its online worker-correctness Brier score improves on the fixed prior on four of five sets and worsens on Bluebirds. This is more informative than comparing final reliability estimates to all gold labels, but it remains a five-dataset, fixed-prior, binary, one-coin study. The method is not yet compared with SBIC or other specialized streaming methods.

The calibration analysis gives a second useful external check. On Snow RTE, this run obtains confidence-ECE 0.169 for majority vote and about 0.066 for Dawid–Skene; Singer et al. report 16.9% and 6.8% respectively on RTE. This close match supports the parser and this particular ECE implementation on that benchmark; it is not a full replication of their cross-dataset study. Calibration error depends on binning, and ECE is not a proper scoring rule. Brier score and log loss remain necessary companions.

### What changed from research/10

The prior v0.4 sequential results sorted workers lexicographically within each item. That is not a neutral order for an order-dependent update. The old sequential accuracy/reliability table is therefore superseded by this randomized-arrival reanalysis. The simultaneous and exact-ADF item posteriors are theoretically report-order invariant under the implementation's assumptions; new report randomization is not expected to change those results. New 31-point exact-ADF and Dawid–Skene scores above are separately identified and should not silently replace the 101-point table.

## 5. Remaining validity limits and next work

### Highest-priority research work

1. **Matched comparison with established streaming methods.** Implement or reproduce SBIC and online Dawid–Skene with their stated assumptions. Compare runtime, memory, calibration, prequential log loss/Brier, and recovery under synthetic data generated both inside and outside each model. Preserve method-specific initialization and convergence rules.
2. **Reliability model fit.** Add a clearly separate asymmetric sensitivity/specificity (two-coin) network variant, plus a chance/below-chance source model. Use Bluebirds and imbalanced Product as known stress cases. Do not silently retrofit asymmetric behavior into the historical Laputa rule.
3. **Behavioral validation.** Use appropriately licensed human-subject data, including the cited Madsen/Pilditch corpora where usage permits, to compare the normative Bayesian agent with a dependency-insensitive behavioral model. Crowd-label accuracy is not a substitute for this.
4. **Network dependence.** Connect evidence identity and shared ancestry to network messages. A relay should carry source evidence IDs or a calibrated dependence model; repeated claims must not be treated as independent merely because they arrive over different edges.
5. **Calibration protocol.** Keep test data untouched. If post-hoc temperature scaling or an aggregator ensemble is added, fit it only on a held-out calibration split or through nested cross-validation, and report calibration-set size and subgroup results.
6. **Robustness and application fit.** Sweep trust priors, class prevalence, graph topology, report order, source heterogeneity, redundancy, missingness, and distribution shift. Specify the target decision and utility before calling an information-seeking policy “better.”

### Scope that remains unsupported

KANM is not validated as a human-behavior predictor or a policy tool. Its current public-data branch is single-reasoner binary aggregation, not a test of network dynamics. The exact update is exact only under the finite grid and stated independence model; the across-item carry-forward is assumed-density filtering. The latest calibration paper is a preprint and is cited as motivation, not settled consensus. The five public data sets include limited task and domain diversity.

## 6. Reproduction

```sh
python -m examples.crowd_benchmark /path/to/data --download --orders 10 --seed 0 --trust-points 31
```

To compare grid resolution, rerun with 51 or 101 points and report the change separately. Do not compare raw timing across machines as an algorithmic guarantee. The downloaded data are not included in KANM because the dataset sources do not state a redistribution licence for most files.

Relevant primary sources are registered in [references.md](references.md), including the [Snow RTE source paper](https://aclanthology.org/D08-1027/), [Zheng et al. truth-inference study](https://doi.org/10.14778/3055540.3055547), [SBIC](https://arxiv.org/abs/1911.05712), [online Dawid–Skene](https://arxiv.org/abs/1512.02393), and the recent [crowd-aggregator calibration preprint](https://arxiv.org/abs/2610.16815).
