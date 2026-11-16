# Docking KANM against the Laputa 1.6 source

This directory builds the **unmodified** model core of the archived Laputa 1.6 source as a headless program. It then compares KANM with it step by step. Laputa 1.6 (GPLv3) is not included; download it from the SourceForge project [epistemenet](https://sourceforge.net/projects/epistemenet/).

## What is compiled, and what is replaced

`build.sh` checks every Laputa file it uses against `laputa_1_6_sources.sha256`, which holds the checksums of the 1.6 archive. It refuses to build if any file differs.

**Compiled unchanged:**
- `Society.cpp`, which contains `Society::evolve`, the step;
- `Inquirer.cpp`, which contains `Inquirer::doInquiry`: credence and trust updates and the new-evidence policies;
- `Link.cpp`;
- `Trust.cpp`, the 49-node trust function;
- `Distribution.cpp`, `MetaDistribution.cpp` and `Parameters.cpp`, which build trust priors;
- `StatisticsBlock.cpp`;
- all headers except `App.h`;
- TinyXML.

These are the files that the original VC++ and Xcode projects compile. `Amount.cpp` is in neither project; `Amount.h` is header-only.

**Replaced by `stubs/`.** None of these contain model logic:
- **`App.h`:** the GUI application object.
- **FLTK headers:** the drawing widgets in `Trust.cpp` and `Distribution.cpp` are compiled but never instantiated.
- **The GSL functions the core calls:**
  - a seeded `std::mt19937_64` generator;
  - the Beta and Gaussian densities and distribution functions, written from their standard definitions.
- **`prefix.h`:** a lower-case include that resolves only on case-insensitive file systems.

**Supplied by `harness.cpp`:**
- `Simulation`'s constructor and its logging and statistics members;
- a text formatter;
- the per-run initialisation of `Simulation::reset` (lines 126–134 of the 1.6 `Simulation.cpp`), restated: evidence counters are set to −1 and initial link messages are set from starting beliefs.

## Use

```sh
docking/laputa_1_6/build.sh /path/to/Laputa-1.6 build/laputa16
python -m docking.laputa_1_6.compare build/laputa16/laputa16_harness              # deterministic
python -m docking.laputa_1_6.compare build/laputa16/laputa16_harness --stochastic # distributions
python -m docking.laputa_1_6.compare build/laputa16/laputa16_harness --emulate-float32  # post hoc: Laputa's float precision (needs numpy)
```

## Acceptance criteria (fixed before any comparison)

### Deterministic docking

Every chance (inquiry chance, inquiry accuracy, listen chance) is exactly 0 or 1, so the two trajectories must agree step by step.

There are 200 random scenarios, each run for 25 steps, mixing:
- 2–8 inquirers;
- all three new-evidence policies;
- count-prior flags;
- liar thresholds;
- three kinds of trust function.

A scenario passes when all of the following hold:
- credences agree within 1e-5 at every step;
- trust expectations agree within 1e-5 at every step;
- message and inquiry counts are identical.

The tolerance reflects Laputa's single-precision trust storage.

A scenario is counted as *indeterminate*, not failed, if a decision depends on a comparison within 1e-6 of its cut point.

### Stochastic docking

The two programs use different random generators, so only distributions can be compared. Mean final credences are compared with a two-sample z statistic, and the scenario passes when every |z| ≤ 4.

## Status and results (2026-11-16)

**Run setup:**
- **Machine:** the author's, Linux x86-64, harness built with g++. The author ran the first deterministic test; the stochastic, diagnostic and confirmatory runs were made on the same build.
- **Code:** KANM 0.4.0.
- **Defaults:** 200 scenarios × 25 steps for the deterministic test; 5 scenarios × 400 runs for the stochastic test; seed 0.

| Test | Criterion (fixed in advance) | Result | Verdict |
|---|---|---|---|
| Deterministic | every \|Δ\| ≤ 1e-5, counts identical | 199 passed, **1 failed**, 0 indeterminate. Message and inquiry counts identical in all 200. Largest differences: credence 1.94e-5, trust 1.28e-5, both in scenario 19 | **Not met** |
| Stochastic | every \|z\| ≤ 4 | 25 statistics, max \|z\| = 2.75 | **Met** |

**Post hoc diagnosis of the deterministic failure.** This analysis was chosen after seeing the failure.

*Where the difference arises.* In scenario 19 (8 inquirers, 30 links, policy `none`), the difference is not a jump at a decision. It grows steadily, always with the same sign:
- it builds up in one inquirer's credence: inquirer 2, which never inquires and listens to three sources;
- it shows up in the trust on one of that inquirer's incoming links.
At step 0, every trust expectation already differs by about 1e-7.

*The likely cause.* The build sets `t_float` to `double`, but `TrustFunction` hard-codes `float` for:
- the node values;
- the credence passed into the update;
- the running node position;
- the trapezoid total;
- the expectation.
Errors of about 1e-7 in the expectation enter the receiver's log-odds at every message. Because they share a sign, they accumulate instead of cancelling.

*The test.* `float32.py` makes KANM reproduce that mix of precisions operation by operation. With it, `compare.py --emulate-float32` gives:
- **all 200 scenarios pass**;
- maximum credence difference 6.4e-14;
- maximum trust difference 5.0e-10, which equals the rounding of the harness's 9-significant-digit output.

*Conclusion.*
- **Logic:** KANM's restatement of the 1.6 step, trust function, new-evidence policies and assertion behaviour reproduces the compiled 1.6 core on every tested configuration. The only exception is arithmetic precision.
- **Precision:** with KANM's default double precision, differences from Laputa reached 1.9e-5 over 25 steps here, and 5.0e-5 in the companion run below.

*Why the criterion stands.* The tolerance of 1e-5 was justified in advance as reflecting single-precision storage. That justification ignored accumulation, and the criterion was not met. We record the miss and do not relax the criterion after the fact.

**Confirmatory run, declared before it was executed (2026-11-16).** The diagnosis above was post hoc. It is therefore tested on 200 *new* scenarios with the following declaration.

*Command:*
```sh
python -m docking.laputa_1_6.compare HARNESS --emulate-float32 --seed 1
```

*Criteria:*
- **Pass:** the same criterion as the original deterministic test: every |Δ| ≤ 1e-5 and identical counts in every scenario that is not indeterminate.
- **Failure:** any failed scenario counts against the diagnosis.

*Expectations stated before the run:*
- maximum credence difference below 1e-9;
- maximum trust difference no larger than the output rounding of 5e-10.

*Companion run:* the same seed in double precision (without the flag), reported alongside for comparison.

*Result:*

| Run | Passed | Failed | Indeterminate | Max credence diff | Max trust diff |
|---|---|---|---|---|---|
| Float emulation (declared test) | **200** | **0** | 0 | 1.5e-14 | 3.0e-8 |
| Double precision (companion) | 195 | 5 | 0 | 5.0e-5 | 1.4e-5 |

- **The pass criterion is met.**
- **The credence expectation is met** (1.5e-14 < 1e-9).
- **The trust expectation is not met.** Trust differs by up to 3.0e-8, not ≤ 5e-10.

*Follow-up on the unmet expectation (post hoc).* Only 5 of 97 006 trust values exceed print rounding.
- **What they are:** they come from 3 events, each a difference of 1–2 units in the last place of a single-precision value.
- **How they were traced:**
  1. Verbatim compiled copies of Laputa's `update` and `expectation` loops agree bit for bit with `float32.py` on the traced inputs (all 25 updates and 26 states of one affected function).
  2. Replaying each affected function's updates with *Laputa's own* printed credences reproduces Laputa's trust expectations to within print rounding (≤ 4.9e-10).
- **The cause:** the residual lies in the credence handed to the trust update, not in the trust arithmetic. Laputa reads credence as `Amount::v()` = 0.5 + 0.5 · (lower − upper), whose absolute precision is about 1e-16. At credences around 1e-10 (reached after steps 10–16 in these runs), Laputa's value and KANM's differ by about 1e-6 in relative terms. That is enough to round to a different `float`, which leaves the two trust functions 1–2 ulp apart.
- **Not emulated:** reproducing this would require emulating Laputa's `Amount` arithmetic.
- **Correction:** a first explanation, that `Amount::v()` rounds credences below 1e-16 to 0, was tested by emulating that readout. It changed no result, and was withdrawn.

*The double-precision companion:* all 5 failed scenarios had identical counts, and all 5 pass under float emulation (first row).

**Scope:** the default `compare.py` settings only. These are the 1.6 model core, the new-evidence policies, count-prior flags, liar thresholds and three kinds of trust function, on networks of 2–8 inquirers.

## What docking can and cannot establish

**Passing establishes** that KANM's `schedule="laputa_1_6"`, `assertion_rule="laputa_1_6"` and `trust_model="laputa_1_6"` reproduce the 1.6 model core on the tested configurations.

**It does not establish:**
- agreement with any *published* simulation result, because the papers do not report their full configurations;
- agreement with the GUI-generated batch societies, whose network generator is not part of this harness.
