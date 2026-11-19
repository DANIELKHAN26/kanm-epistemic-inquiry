# KANM

If five people tell you the same thing, how much should you believe it? That depends on where they got it. Five people who each checked for themselves are strong evidence. Five people repeating one news story are worth about one story.

KANM is a small Python library for working this out exactly. You describe who heard what from whom, and it tells you what the reports are actually worth. It also simulates groups of people who talk, trust each other more or less, and change their minds, building on the Laputa model of Erik Olsson and Staffan Angere.

```python
from kanm.dependence import ProvenanceNetwork, relay, symmetric_primary

net = ProvenanceNetwork()
net.add(symmetric_primary("study", 0.75))         # one study, never seen directly
for k in range(5):
    net.add(relay(f"person{k}", "study", 0.95))    # five people passing it on

heard = {f"person{k}": True for k in range(5)}
net.posterior(0.5, heard)                # 0.75: worth about one study
net.independence_posterior(0.5, heard)   # 0.992: what you get if you count them separately
```

## Install

Python 3.11 or newer. No other dependencies.

```sh
python -m pip install git+https://github.com/DANIELKHAN26/kanm-epistemic-inquiry
```

## What is inside

- `kanm.dependence`: exact belief for reports with known sources, including people who copy, relay, or partly check for themselves.
- `kanm.testimony`: chains of informants who hear each other before speaking, as in the classic balls-and-urns experiments.
- `kanm.model`: network simulations of belief and trust, with Laputa's update rule or an exact one.
- `kanm.reliability`: learning how reliable each source is from their track record, tested on five public crowdsourcing data sets.
- `kanm.inquiry` and `kanm.collaborative`: deciding which question to ask next, and who should check what.
- `docking/`: a harness that compiles the original Laputa 1.6 code and compares it with KANM step by step.

## Run a simulation

Describe a network in a JSON file:

```json
{
  "steps": 50,
  "seed": 7,
  "n_agents": 10,
  "agent_defaults": {"prior_belief": 0.5, "inquiry_accuracy": 0.7, "inquiry_chance": 0.4},
  "network": {"type": "erdos_renyi", "p": 0.3, "seed": 7},
  "edge_defaults": {"listen_chance": 0.5, "assertion_threshold": 0.75}
}
```

Then run it once, or compare Laputa's update with the exact one over many seeds:

```sh
kanm-run network.json
kanm-run network.json --replicates 100 --compare update_rule=laputa,exact_joint
```

The data loaders (`kanm.crowd_data`, `kanm.testimony_data`, `kanm.structure_data`) download public data sets only when asked, check each file against a fixed hash, and never redistribute them.

## Tests

```sh
python -m unittest discover -s tests
```

## Cite

Khan, D. (2026). *KANM* (version 0.10.0) [Computer software]. https://github.com/DANIELKHAN26/kanm-epistemic-inquiry

GitHub's "Cite this repository" button gives the same in BibTeX. If you use the Laputa-style updates, please also cite Olsson (2011) and Vallinder & Olsson (2014).

## Licence

GPL-3.0-only. See `LICENSE` and `NOTICE.md`.

Made by Daniel Khan · [danielkhan.org](https://danielkhan.org)
