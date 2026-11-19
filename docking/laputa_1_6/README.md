# Laputa 1.6 compatibility harness

This optional developer harness compiles the **unmodified** model core of a separately obtained Laputa 1.6 archive as a headless program, then compares its trajectories with KANM step by step. Laputa 1.6 (GPLv3) is not included; it is available from the SourceForge project [epistemenet](https://sourceforge.net/projects/epistemenet/). `build.sh` checks every Laputa file it uses against `laputa_1_6_sources.sha256` and refuses to build if any file differs.

## What is compiled, and what is replaced

**Compiled unchanged:**
- `Society.cpp`, `Inquirer.cpp` and `Link.cpp`;
- `Trust.cpp`;
- `Distribution.cpp`, `MetaDistribution.cpp` and `Parameters.cpp`;
- `StatisticsBlock.cpp`;
- all headers except `App.h`;
- TinyXML.

**Replaced by `stubs/`.** None of these contain model logic:
- `App.h`;
- the FLTK headers, whose drawing widgets are compiled but never instantiated;
- the GSL functions the core calls, rewritten as a seeded `std::mt19937_64` and the Beta and Gaussian densities from their standard definitions;
- a lower-case `prefix.h` include.

**Supplied by `harness.cpp`:**
- `Simulation`'s constructor and its logging and statistics members;
- a text formatter;
- the per-run initialisation of `Simulation::reset` (lines 126–134 of the 1.6 `Simulation.cpp`), restated.

## Use

```sh
docking/laputa_1_6/build.sh /path/to/Laputa-1.6 /tmp/kanm-laputa-build
python -m docking.laputa_1_6.compare /tmp/kanm-laputa-build/laputa16_harness                    # step by step
python -m docking.laputa_1_6.compare /tmp/kanm-laputa-build/laputa16_harness --stochastic       # distributions
python -m docking.laputa_1_6.compare /tmp/kanm-laputa-build/laputa16_harness --emulate-float32  # Laputa's float precision (needs NumPy)
```

**Deterministic comparison.** Every chance is set to 0 or 1, so the trajectories must agree step by step. A scenario passes when credences and trust expectations agree within 1e-5 at every step and message and inquiry counts are identical.

**Stochastic comparison.** The two programs use different random-number generators, so only distributions can be compared. It compares mean final credences with a two-sample z statistic.

## What it shows

On 200 random scenarios of 2–8 inquirers, with all new-evidence policies, count-prior flags, liar thresholds and three trust functions:
- **With float32 emulation**, KANM reproduces the compiled 1.6 core in every scenario, with credence differences below 1e-13.
- **In KANM's default double precision**, differences of up to about 5e-5 accumulate over 25 steps. They come from Laputa's single-precision trust function, not from a difference in logic.

Passing establishes agreement with the 1.6 model core on the tested configurations only. It does not establish:
- agreement with published simulation results, whose full configurations are not reported;
- agreement with the GUI's batch societies, whose network generator is not part of this harness.
