# Contributing

KANM is an early research prototype. Contributions should preserve clear separation between the historical baseline and optional extensions.

## Before changing a model mechanism

1. State the scientific question and intended use.
2. Identify relevant prior work and distinguish normative from descriptive claims.
3. Write down the mechanism, assumptions, parameters, and null comparison before implementation.
4. Add tests for mathematical cases and boundary conditions.
5. Update the research specification, ODD description, and validation gates.

## Development checks

Use Python 3.11 or newer. The runtime has no third-party dependencies.

```sh
python -m unittest discover -s tests -v
python -m kanm.cli examples/minimal_network.json --replicates 5
python -m kanm.cli examples/generated_network.json --replicates 5 --compare update_rule=laputa,exact_joint
python -m examples.planning_demo
python -m examples.provenance_demo
```

Keep the published rule, the Laputa 1.6 code behaviour and KANM's extensions as separately named options. Do not claim the non-exactness of the Olsson update as new (it is published; see research/08 §3). Do not describe the network runner as a faithful Laputa port until a documented docking or published-result reproduction has passed. Report failed, null, and adverse results as well as positive findings.

## Scope and review

Do not add a dependency-discount heuristic for shared evidence without specifying its generative model and validating it against exact small cases. Do not add identity-based competence assumptions as intrinsic group traits. Keep private identifiers, access tokens, and unpublished human data out of commits and issue reports.
