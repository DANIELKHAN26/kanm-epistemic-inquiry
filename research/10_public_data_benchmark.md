# Public-data test: learning source reliability from testimony

> **Revision notice (16 November 2026):** The original v0.4 sequential results below used lexicographic worker order within each item. Because sequential Laputa updates are order dependent, those numbers are not a neutral estimate of performance. They are retained as the original record but are **superseded for interpretation** by the randomized within-item reanalysis in [research/11](11_deep_research_and_v0.5_validation.md). The new run also adds prequential reliability scoring and estimated-prevalence Dawid–Skene references. Do not combine the two result tables as if they used the same protocol or trust-grid resolution.

**Status:** design fixed on 2026-11-15, before any method was run on the real data. Results are in [section 5](#5-results), added afterwards without changing sections 1–4.

## 1. Why this test

Laputa's trust update is a proposal for one specific task: learning how reliable a source is from its reports, without ever observing whether those reports were correct. Crowdsourcing truth inference studies the same task, and the field's public data sets include gold answers. They therefore give an empirical test that KANM's own simulations cannot give.

The correspondence with Laputa:
- **Items** are binary questions (propositions).
- **Workers** are sources with fixed, unknown reliability.
- **Source symmetry** is Laputa's assumption; it is the one-coin model of the crowdsourcing literature.
- **Gold answers** play the role of the truth that the inquirer never sees.

What it does **not** test: network dynamics, assertion thresholds, or human belief updating. The workers are annotators, not inquirers in a social network, and the reasoner here is a single agent who reads all their reports.

## 2. Data

Five binary data sets with gold answers for every item, from three independent sources. Download and parsing are implemented in `kanm.crowd_data`, with SHA-256 verification of every file. None of the data is redistributed with KANM: no licence is stated for most of it.

| Data set | Items | Workers | Labels | Reports per item | Source |
|---|---|---|---|---|---|
| `snow2008_rte` | 800 | 164 | 8 000 | 10 | Snow et al. (2008) |
| `snow2008_temp` | 462 | 76 | 4 620 | 10 | Snow et al. (2008) |
| `welinder2010_bluebirds` | 108 | 39 | 4 212 | 39 | Welinder et al. (2010) |
| `zheng2017_product` | 8 315 | 176 | 24 945 | 3 | Zheng et al. (2017), from Wang et al. (2012) |
| `zheng2017_sentiment` | 1 000 | 85 | 20 000 | 20 | Zheng et al. (2017) |

**Parser validation:**
- Every item, worker and label count above matches the original papers and an independent count.
- Majority-vote accuracy reproduces the published figures: RTE 0.897 (Snow et al.: 89.7%) and Product 0.8966 (Zheng et al.: 89.66%).

**Excluded:**
- Toloka Relevance-2: gold for only 10% of items, and 475 000 labels, which is too many for a pure-Python exact filter within this study's compute budget.
- Multi-class data sets.

## 3. Methods

All methods use a class prior of 1/2.

The trust-learning methods start from the same reliability prior, Beta(2, 1): mean 2/3, concentration 3, which says sources are probably better than chance. The Dawid–Skene methods use the same prior as pseudo-counts.

The online methods use a 101-point trust grid and process items one at a time in random orders. Each method is run over 10 orders (seed 0); the standard deviation over orders measures order sensitivity.

| Method | Kind | Description |
|---|---|---|
| `majority_vote` | baseline | Vote; the score is the share of positive votes |
| `laputa_simultaneous` | KANM, online | Laputa rule, simultaneous schedule: trust updated with the credence held before the item (1/2) |
| `laputa_sequential` | KANM, online | Laputa rule, reports applied one by one; trust updated with the running credence |
| `exact_adf` | KANM, online | Exact joint posterior within each item (`ExactJointState`); each worker's exact marginal trust carried to the next item (assumed-density filtering) |
| `dawid_skene_one_coin` | reference, offline | MAP EM, symmetric worker accuracy (Dawid & Skene 1979, one-coin form) |
| `dawid_skene_two_coin` | reference, offline | MAP EM, separate sensitivity and specificity |

**Prediction made before running.** `laputa_simultaneous` cannot learn. A report received at credence 1/2 carries no information about reliability, so every trust stays at its prior. With equal priors its decisions must therefore equal majority vote. This is proved in the `kanm.reliability` docstring and tested in `tests/test_reliability.py`. The real-data run is a check of this, not a discovery.

## 4. Outcomes and questions

**Outcomes** are computed on gold items:
- accuracy (ties count as half correct);
- Brier score;
- log loss (clipped at 1e-12; the number of clipped items is reported);
- expected calibration error with 10 bins.

**Reliability recovery:**
- the Pearson correlation and mean absolute error between each worker's estimated reliability and their empirical accuracy against gold;
- computed for workers with at least 20 gold-scored reports.

**Questions:**
- **Q1.** Does any form of the Laputa update learn reliability, and does that improve on majority vote?
- **Q2.** How does the online exact filter (`exact_adf`) compare with majority vote and with the offline Dawid–Skene references, which see all data at once?
- **Q3.** How order-sensitive are the online methods?

**Limits stated in advance:**
- Five data sets is a small sample.
- Differences between methods on one data set are not tested for significance beyond the order-to-order SD.
- The offline references have an information advantage: they see every item before deciding any of them.
- Worker reliability on real data is not symmetric: some workers are biased towards one answer. The two-coin model can represent this; the KANM rules cannot.

## 5. Results

**Run:**
- Command: `python -m examples.crowd_benchmark <data dir> --orders 10` (seed 0).
- Code: KANM 0.4.0.
- Data: all files matched their recorded SHA-256 checksums.
- Repeatability: two complete runs agreed to within 1.2e-16 on every reported number. The only differences came from a fix to summary rounding made between them: exact summation for means, and no Pearson value when there is no variance.

**Reporting:**
- Online methods: mean ± SD over 10 item orders.
- Offline methods: deterministic, so a single run.
- Accuracy differences of a few thousandths on one data set are within what this design cannot distinguish (§4).

### 5.1 Accuracy against gold

| Method | RTE | TEMP | Bluebirds | Product | Sentiment |
|---|---|---|---|---|---|
| `majority_vote` | 0.897 | 0.936 | 0.759 | 0.897 | 0.934 |
| `laputa_simultaneous` | 0.897 | 0.936 | 0.759 | 0.897 | 0.934 |
| `laputa_sequential` | 0.836 ± 0.045 | 0.596 ± 0.079 | 0.756 ± 0.031 | 0.925 ± 0.002 | 0.950 ± 0.003 |
| `exact_adf` | 0.922 ± 0.002 | 0.941 ± 0.003 | 0.686 ± 0.074 | 0.927 ± 0.001 | 0.950 ± 0.003 |
| `dawid_skene_one_coin` | 0.928 | 0.939 | 0.722 | 0.928 † | 0.951 |
| `dawid_skene_two_coin` | 0.928 | 0.942 | **0.898** | 0.837 | **0.960** |

† Stopped at the 1000-iteration cap without converging. A later check with a 10 000 cap converged after 2 463 iterations, with accuracy 0.92796, the same as at the cap.

### 5.2 Reliability recovery

Pearson correlation between each worker's estimated reliability and their empirical accuracy against gold. Only workers with at least 20 gold-scored reports are included: 164, 38, 39, 97 and 85 workers respectively.

| Method | RTE | TEMP | Bluebirds | Product | Sentiment |
|---|---|---|---|---|---|
| `laputa_simultaneous` | — (no variance) | — | — | — | — |
| `laputa_sequential` | 0.07 ± 0.09 | −0.10 ± 0.05 | 0.76 ± 0.05 | 0.75 ± 0.01 | 0.85 ± 0.00 |
| `exact_adf` | 0.90 ± 0.01 | 0.96 ± 0.00 | 0.52 ± 0.18 | 0.96 ± 0.00 | 0.98 ± 0.00 |
| `dawid_skene_one_coin` | 0.91 | 0.95 | 0.59 | 0.96 | 0.98 |
| `dawid_skene_two_coin` | 0.88 | 0.92 | 0.96 | −0.16 | 0.98 |

Mean absolute error tells the same story. For `exact_adf` it is 0.028–0.041 on the four data sets other than Bluebirds, and 0.149 on Bluebirds.

### 5.3 Probability quality

Brier score and log loss, with the number of items whose log loss was clipped at 1e-12 shown in brackets:

| Method | RTE | TEMP | Bluebirds | Product | Sentiment |
|---|---|---|---|---|---|
| `majority_vote` | 0.104 / 0.35 | 0.092 / 0.40 [1] | 0.164 / 0.51 | 0.092 / 0.72 [149] | 0.112 / 0.39 |
| `laputa_simultaneous` | 0.075 / 0.26 | 0.051 / 0.21 | 0.224 / 1.68 | 0.096 / 0.33 | 0.050 / 0.19 |
| `exact_adf` | 0.064 / 0.30 | 0.052 / 0.40 | 0.305 / 4.47 [3.2] | 0.062 / 0.32 | 0.046 / 0.44 |
| `dawid_skene_one_coin` | 0.067 / 0.78 [14] | 0.059 / 1.12 [11] | 0.276 / 4.42 [2] | 0.063 / 0.53 [48] | 0.046 / 0.54 [3] |
| `dawid_skene_two_coin` | 0.068 / 1.19 [24] | 0.058 / 1.30 [19] | 0.102 / 1.85 [3] | 0.199 / 2.56 [70] | 0.038 / 0.47 [5] |

### 5.4 Answers to the questions

**Q1. Does any form of the Laputa update learn reliability, and does that improve on majority vote?**
- **Simultaneous schedule: no.** The prediction made before the run is confirmed. On all five data sets, the accuracy of `laputa_simultaneous` equals majority vote exactly, and every worker's trust stays at the prior: the correlation is undefined because there is no variance.
- **Why its Brier score and log loss still differ from majority vote:** its score is a posterior computed with fixed trust, whereas majority vote's score is the vote share. This is a different scoring rule, not learning.
- **Sequential schedule: sometimes, and not reliably.**
  - **Where it learns:** on Product and Sentiment it improves on majority vote by 0.028 and 0.016, with reliability correlations of 0.75 and 0.85.
  - **Where it hurts:** on RTE and TEMP it is worse than majority vote by 0.061 and 0.340. Its reliability estimates are uncorrelated with true accuracy (0.07 and −0.10), and it is strongly order-dependent.
  - **Untested explanation:** a plausible mechanism is self-reinforcement. Trust is updated against a credence formed partly from the same item's earlier reports, so an early wrong lead can lower the trust of the workers who were in fact right. We did not test this.
- **Conclusion:** Laputa's update is not a dependable reliability learner on these data. It either does not learn at all or learns unstably, depending on the schedule.

**Q2. How does the online exact filter (`exact_adf`) compare with majority vote and with the offline Dawid–Skene references?**
- **Data sets other than Bluebirds:**
  - It improves on majority vote by 0.005 to 0.031.
  - Its accuracy is within 0.006 of the offline one-coin reference, which assumes the same symmetric worker model, despite deciding each item having seen only earlier items.
  - Its reliability recovery matches that reference (0.90–0.98 against 0.91–0.98).
  - Its log loss is lower than both offline references on all four, and no item's log loss was clipped. The offline EM methods put probability below 1e-12 on the true answer for 3–70 items, which is overconfidence.
- **Bluebirds:**
  - It is worse than majority vote, by 0.073, and is unstable across orders (SD 0.074).
  - The one-coin reference also falls below majority vote there (0.722).
  - Only the two-coin model, which lets a worker's accuracy differ by true class, does well (0.898).
  - This is the failure mode named in the limits stated before the run (§4): asymmetric workers, which no symmetric model, KANM's included, can represent.

**Q3. How order-sensitive are the online methods?**
- **`exact_adf`:** SD over orders is at most 0.003 in accuracy on four data sets, and 0.074 on Bluebirds.
- **`laputa_sequential`:** SD is 0.045 on RTE, 0.079 on TEMP and 0.031 on Bluebirds.
- **Large SD goes with failure:** order sensitivity is large exactly where the method's model does not fit or its learning is unstable.

### 5.5 Post hoc diagnostics

**These were chosen after seeing the results. They are exploratory and are not tests of anything stated in §1–4.**

**(a) Worker asymmetry.** For each worker we computed empirical sensitivity and specificity against gold. We included workers with at least 20 gold-scored reports and at least 5 on each class (script: `python -m examples.crowd_posthoc <data dir>`).

| | RTE | TEMP | Bluebirds | Product | Sentiment |
|---|---|---|---|---|---|
| Workers | 161 | 37 | 39 | 66 | 85 |
| Median \|sensitivity − specificity\| | 0.136 | 0.066 | 0.408 | 0.375 | 0.154 |
| Share of workers with gap > 0.3 | 0.17 | 0.22 | 0.69 | 0.53 | 0.32 |
| Workers below chance overall | 2 | 3 | 7 | 8 | 9 |

- **Bluebirds** is the most asymmetric, which is consistent with Q2. With the class-balance condition relaxed to at least one report per class, the figures are almost the same: Bluebirds is unchanged, and Product rises to 0.378 across 95 workers.
- **Product** is also strongly asymmetric, yet `exact_adf` does well on it. Asymmetry alone therefore does not predict the symmetric model's failure. On Product, with only 3 reports per item, a strong class imbalance (12% positive) seems to matter as well; see (b).

**(b) Two-coin Dawid–Skene with the class prior re-estimated** in each M-step instead of being fixed at 1/2. Same script; same Beta(2, 1) pseudo-counts on sensitivity and specificity; at most 3 000 iterations.


| | RTE | TEMP | Bluebirds | Product | Sentiment |
|---|---|---|---|---|---|
| Estimated class prior | 0.482 | 0.455 | 0.437 | 0.107 | 0.451 |
| Accuracy (fixed 1/2, from 5.1) | 0.928 | 0.942 | 0.898 | 0.837 | 0.960 |
| Accuracy (estimated prior) | 0.926 | 0.942 | 0.898 | 0.939 * | 0.960 |

\* Not converged after 3 000 iterations.

- **Product:** the two-coin model's collapse there is caused by the fixed class prior of 1/2 that §3 imposed on all methods. The gold positive share is 0.122.
- **Other data sets:** the change is negligible.

### 5.6 What these results support, and what they do not

**Supported, on these five data sets:**
- Under the simultaneous schedule, Laputa's trust update cannot learn reliability from testimony alone. Its decisions reduce to majority vote; this is proved and was confirmed on the data.
- The sequential variant learns erratically.
- Exact within-item inference, carried forward by assumed-density filtering, recovers worker reliability about as well as offline EM when workers are roughly symmetric, and does so with better-calibrated probabilities.

**Not supported:**
- Any claim about networks of inquirers, assertion thresholds or human reasoning (§1).
- Generalisation beyond binary tasks, or beyond symmetric workers.
- Ranking `exact_adf` above the Dawid–Skene references. Those references are offline, and differences of a few thousandths are not resolved by this design.

**Limits beyond those stated before the run (§4):**
- One trust prior and one class prior were used throughout. The post hoc check (b) shows the class prior matters on imbalanced data.
- Bluebirds has only 108 items, so one item moves accuracy by 0.009.
