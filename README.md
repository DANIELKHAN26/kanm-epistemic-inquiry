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
- `kanm.model`: network simulations of belief and trust, with Laputa's update rule or an exact one. Run them from the command line with `kanm-run examples/minimal_network.json`.
- `kanm.reliability`: learning how reliable each source is from their track record, tested on five public crowdsourcing data sets.
- `kanm.inquiry` and `kanm.collaborative`: deciding which question to ask next, and who should check what.
- `docking/`: a harness that compiles the original Laputa 1.6 code and compares it with KANM step by step.

The `examples/` folder has short demos and the scripts behind each study. Some studies download public data sets on request; nothing is redistributed, and every file is checked against a fixed hash. The study scripts need NumPy, and some need SciPy (`pip install ".[analysis]"`).

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
