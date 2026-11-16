# Exploratory results: factorised (Laputa) versus exact-joint updating

**Status: exploratory simulation, not a validated finding.**
- **Script:** `python -m examples.study_factorization 300`.
- **Design:** stated in the script docstring before the run.
- **Settings:** illustrative and uncalibrated: 10 agents, Erdős–Rényi link chance in {0, 0.1, 0.3, 0.6, 1.0}, random priors, accuracies and activity per replicate, listen chance 0.5, threshold 0.75, 50 steps, simultaneous schedule.
- **Pairing:** both rules ran on identical random inputs (common random numbers) for 300 replicates per density.
- **Uncertainty:** "±" is one Monte Carlo standard error.
- **Interpretation:** with 5 densities × 4 outcomes, differences near 2 SE should be read cautiously.
- **Code version:** final v0.3.0, rerun after the D9–D10 discretisation fix in [08](08_v0.3_audit_and_extensions.md). No reported mean changed by more than 0.0001 relative to the pre-fix run, as expected: the fix matters for trust means near 0 or 1, and this design uses 0.6.
- **Re-checked for v0.4.0:**
  - **Log-space trust (D16):** rerun with log-space trust weights. Every reported mean agreed with the v0.3.0 run to 1e-16.
  - **Credence-½ tie handling and the log-odds assertion rule (D17):** these changed after that rerun. Instrumenting 120 paired replicates of this design found that neither code path is ever reached: 0 trust updates at credence exactly ½, and 0 assertion decisions at an exact threshold tie. The table therefore holds for v0.4.0.

## Results

| Link chance | V-value Laputa | V-value exact | Paired diff (exact − Laputa) | Brier diff | Accuracy diff | Wrong majority, Laputa / exact |
|---|---|---|---|---|---|---|
| 0.0 | 0.292 ± 0.004 | 0.192 ± 0.003 | **−0.100 ± 0.002** | **+0.033 ± 0.002** | −0.035 ± 0.005 | 0.000 / 0.003 |
| 0.1 | 0.335 ± 0.008 | 0.277 ± 0.007 | **−0.058 ± 0.006** | −0.002 ± 0.006 | −0.012 ± 0.007 | 0.027 / 0.037 |
| 0.3 | 0.297 ± 0.017 | 0.297 ± 0.019 | +0.000 ± 0.011 | −0.018 ± 0.011 | +0.004 ± 0.012 | 0.177 / 0.190 |
| 0.6 | 0.169 ± 0.024 | 0.174 ± 0.025 | +0.005 ± 0.009 | −0.006 ± 0.009 | +0.005 ± 0.009 | 0.337 / 0.327 |
| 1.0 | 0.100 ± 0.027 | 0.081 ± 0.027 | −0.019 ± 0.010 | +0.019 ± 0.010 | −0.019 ± 0.010 | 0.403 / 0.420 |

## Reading the results

1. **The exact posterior is more cautious, and under these settings that costs truth-tracking.** Isolated or sparsely connected agents end closer to the truth under Laputa's rule (V higher by 0.10 ± 0.002 at link chance 0).
   - **Mechanism (synthesis):** the factorised rule counts repeated inquiry results from a source of uncertain reliability as fresh evidence (section 3 of [08](08_v0.3_audit_and_extensions.md)).
   - Because every inquiry accuracy in the design exceeds 0.5, this overconfidence usually points the right way.
   - This parallels Vallinder & Olsson's (2014) theme of the value of overconfidence. Coherence with the agent's own model and veritistic performance are different criteria.
2. **In dense networks both rules perform poorly, and their difference is small.** Wrong majorities rise from about 0% to about 40% as link chance goes from 0 to 1 under both rules.
   - This is consistent in direction with published Laputa findings that connectivity can hurt collective accuracy under some conditions (HHO 2020).
   - The settings here are not theirs, so this is not a reproduction.
   - Neither rule accounts for dependence *between* agents, whose assertions share upstream evidence, so exactness about one's own sources does not protect against network-level double counting.
3. **The single-seed example in the development log was not representative.** A dense run in which Laputa agents locked onto the falsehood and exact agents did not was one draw. The paired study shows no systematic advantage for exact agents in dense networks (−0.019 ± 0.010 at link chance 1).
4. **Common random numbers worked as intended.** At link chance 0.6, the paired-difference SE (0.009) is about one third of the per-rule SE (≈ 0.025).

## What this does and does not show

- **Shows:** under one illustrative design, replacing the factorised update by the exact one changes collective outcomes substantially for sparse networks and little for dense ones, and the exact rule is not uniformly better on veritistic criteria.
- **Does not show:** anything about human groups; robustness to trust priors, thresholds, accuracies below 0.5, the number of steps or topology families (no sensitivity analysis has been run); or any calibration claim. The Brier and log scores here measure accuracy against the truth for a single proposition per run, not calibration across many propositions.

## Next analyses (predeclare before running)

- Sensitivity over trust-prior mean and concentration, accuracy ranges that include values below 0.5, step counts, and topology families.
- **Calibration study:** many independent propositions with random truth values, so that the calibration of credences can be assessed. This is where the coherent exact rule is expected to differ most, but that expectation is a hypothesis.
- An evidence-sharing network variant, to separate within-source from between-agent dependence.
