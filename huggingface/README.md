---
license: gpl-3.0
tags:
  - agent-based-model
  - social-epistemology
  - bayesian-inference
  - collaborative-inquiry
  - research-software
---

<!-- Upload this file as README.md at the root of the Hugging Face repository, alongside the research/ folder; the relative links below resolve from there, not from huggingface/. -->

# KANM: Question-Directed Collaborative Inquiry

**Software version:** 0.5.0 (development snapshot, 16 November 2026).

KANM is a research prototype for simulating inquiry and testimony in networks. It combines a Laputa-style binary belief/trust update kernel with separate experimental modules for question planning and evidence provenance.

> **Repository type clarification:** This Hugging Face Model repository contains research software, not a pretrained machine-learning model. It has no trained weights, training data, inference endpoint, or machine-learning pipeline task. The Model repository type is used here only because the project owner requested a Model Hub listing.

## Intended use

Use KANM to run transparent, assumption-driven simulations for methodological exploration in social epistemology, collaborative science, and research on information aggregation. It is not validated as a predictor of human behavior or policy outcomes.

## What is included

- A Bayesian single-binary-proposition network simulator: Laputa's published update, the Laputa 1.6 order of operations (default), and an exact joint-posterior reference rule.
- Exact inference for dependent reports in provenance graphs, with an independence comparator.
- Seeded network generators and paired experiments on common random numbers.
- Reproducible batch runs with full configurations, seeds, and Monte Carlo standard errors.
- A finite-state question planner with expected information gain and decision-utility value of sample information.
- A question-history graph that preserves the original goal and records reasons and mappings for question changes.
- An evidence ledger for source ancestry, exact repeated IDs, and exact categorical scope matching.
- A reproducible five-dataset binary crowdsourcing benchmark with checksum-verified inputs, report-order sensitivity, prequential worker-reliability scores, and fixed/estimated-prevalence Dawid–Skene comparisons.

The question-planning, evidence-ledger, dependence, and public-crowd modules are separate from the network simulator; they do not silently alter its belief updates. Online Bayesian crowd aggregation already has prior art, so KANM makes no state-of-the-art or broad novelty claim. Its exact-joint implementation is a reference under a finite reliability grid and the stated model assumptions; cross-item filtering and cross-agent dependence have separate limitations. See the [prior-art analysis](research/02_prior_work_and_gap.md) and [deep validation update](research/11_deep_research_and_v0.5_validation.md).

## Quick start

Clone the repository and run with Python 3.11 or newer:

```sh
python -m pip install -e .
python -m kanm.cli examples/minimal_network.json --replicates 20
python -m unittest discover -s tests -v
```

The project has no runtime dependencies. See the [research scope](research/00_scope_and_method.md), [validation gates](research/04_validation_plan.md), and [implementation boundaries](research/05_implementation_status.md).

## Model details and limitations

The historical Laputa model concerns one binary proposition and assumes symmetric source reliability and conditional source independence. KANM's network runner reproduces the published update equations and printed worked values, and follows the order of operations of the Laputa 1.6 source. The original double-precision docking comparison missed its deterministic criterion in 1 of 200 scenarios; float32 emulation passed the scenario criterion on 200 new scenarios but missed a separate trust-precision expectation. A reproduction of Vallinder & Olsson (2014, Fig. 1) is partial. No published run has been fully reproduced. Its provenance ledger detects exact repeated IDs and records ancestry, but it does not derive the joint likelihood of partially dependent sources. Expected-information and decision-value policies are supplied-action comparisons, not universal measures of good research.

The code has 158 unit and invariant tests. The public crowd benchmark is a limited empirical test of binary aggregation, not human-behaviour validation. KANM has no policy validation or established novel empirical finding. See the [verification gates](research/04_validation_plan.md), [implementation boundaries](research/05_implementation_status.md), and [latest research update](research/11_deep_research_and_v0.5_validation.md).

## Attribution

KANM is a new implementation based on published equations and descriptions; no historical Laputa source code is included. We acknowledge Erik J. Olsson and Staffan Angere. Relevant citations are listed in the [reference register](research/references.md); see the [SourceForge record](https://sourceforge.net/projects/epistemenet/) and [Olsson's 2013 model exposition](https://lucris.lub.lu.se/ws/portalfiles/portal/5684445/7758617.pdf).

## License

GPL-3.0-only. The software license and attribution notice are included in the repository.
