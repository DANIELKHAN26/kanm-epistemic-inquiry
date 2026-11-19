# KANM

KANM is a Python research library for comparing models of testimony, source reliability, shared evidence, and collaborative inquiry. It includes a network simulator in the Laputa tradition and exact inference tools for explicitly specified small models.

## Install

Python 3.11 or newer is required. The library has no runtime dependencies.

```sh
python -m pip install .
kanm-run --help
```

## Core modules

- `kanm.model`: seeded agent-based simulations with explicit inquiry, communication, and update rules.
- `kanm.bayes`: Laputa-style belief updates and a finite-grid exact-joint comparator for static source reliabilities.
- `kanm.dependence`: exact inference in an analyst-specified binary provenance DAG with conditional probability tables.
- `kanm.model_uncertainty`: posterior and predictive sensitivity across candidate provenance DAGs, with optional Bayesian averaging under explicit model weights.
- `kanm.testimony`: small testimony-chain and urn-task models.
- `kanm.reliability`: majority vote, one-coin and two-coin Dawid–Skene MAP-EM, and online comparison methods.
- `kanm.inquiry` and `kanm.collaborative`: question selection and finite-horizon collaborative planning.
- `kanm.robust_planning`: maximin and minimax-regret inquiry plans when the provenance graph is uncertain, and the value of learning which graph is true.

For example, explicit provenance distinguishes five independent checks from five reports copied from one study:

```python
from kanm.dependence import ProvenanceNetwork, relay, symmetric_primary

network = ProvenanceNetwork()
network.add(symmetric_primary("study", 0.75))
for index in range(5):
    network.add(relay(f"report-{index}", "study", 0.95))

posterior = network.posterior(0.5, {f"report-{index}": True for index in range(5)})
```

When lineage is disputed, `ProvenanceModelSet` can report the posterior range
across candidate graphs without assigning them probabilities. Bayesian model
averaging is a separate option and requires explicit prior model weights; the
weights are updated after each observed report and can be used directly by the
collaborative planner.

```python
from kanm.model_uncertainty import ProvenanceModelSet

independent = ProvenanceNetwork()
independent.add(symmetric_primary("study", 0.75))
for index in range(5):
    independent.add(symmetric_primary(f"report-{index}", 0.75))
observed = {f"report-{index}": True for index in range(5)}

alternatives = ProvenanceModelSet({"copied": network, "independent": independent})
interval = alternatives.posterior_envelope(0.5, observed)

weighted = ProvenanceModelSet(
    {"copied": network, "independent": independent},
    model_weights={"copied": 0.4, "independent": 0.6},
)
posterior = weighted.model_averaged_posterior(0.5, observed)
```

Without defensible weights, an inquiry plan can still be chosen by a stated
criterion. `robust_collaborative_plan` treats the graph as fixed but unknown
and returns the maximin or minimax-regret plan, possibly randomized, together
with a bound showing that no other plan does better. Weights known only within intervals
are accepted through `lower` and `upper`, and `credal_posterior` gives the
matching posterior range.

```python
from kanm.collaborative import CollaborativeAction
from kanm.robust_planning import robust_collaborative_plan, value_of_provenance_information

for net in (network, independent):
    net.add(symmetric_primary("audit", 0.8))
actions = [CollaborativeAction(f"read-{index}", f"report-{index}", "reader", cost=0.02)
           for index in range(3)]
actions.append(CollaborativeAction("check", "audit", "auditor", cost=0.08))
utilities = {"adopt": {True: 1.0, False: -1.0}, "reject": {True: -0.5, False: 0.5}}

graphs = ProvenanceModelSet({"copied": network, "independent": independent})
plan = robust_collaborative_plan(0.5, graphs, actions, utilities, horizon=3,
                                 criterion="minimax_regret")
plan.guarantee                 # worst expected regret 0.0275
plan.first_step_probabilities  # audit first with probability 9/34, else read a report

weighted = ProvenanceModelSet({"copied": network, "independent": independent},
                              model_weights={"copied": 0.4, "independent": 0.6})
value_of_provenance_information(0.5, weighted, actions, utilities, horizon=3).value  # 0.0149
```

## Assumptions and limits

Exact results are exact only for the supplied model. `ProvenanceNetwork` uses the graph and conditional probabilities provided by the analyst; it does not infer unknown sources, causal history, or speakers' reporting policies. `ProvenanceModelSet` makes a finite, analyst-chosen set of alternatives visible; its envelope is not a guarantee about omitted models, and model-averaged results depend on the supplied prior model weights. Exact enumeration is exponential in the number of unobserved ancestors per network and is capped at 20.

`ExactJointState` models one binary hypothesis with static source reliabilities and independent reliability priors conditional on the hypothesis. Robust plans are chosen once, at the start: their guarantee holds for the plan as committed, and re-solving after some reports can recommend something else. Dawid–Skene methods use their stated one-coin or two-coin assumptions and EM can converge to a local MAP solution. None of these mechanisms by itself establishes human predictive validity or a universal rule for trust.

Public-data loaders fetch data only when called and verify checksums. No third-party datasets or trained model weights are distributed.

## Development

Run the package tests with:

```sh
python -m unittest discover -s tests -v
```

The optional `docking/` harness compares KANM's Laputa 1.6 compatibility behavior with a separately obtained historical source archive; that source archive is not included.

## Citation and licence

Citation metadata are in [CITATION.cff](CITATION.cff). KANM is licensed under GPL-3.0-only; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for third-party attributions and data handling.
