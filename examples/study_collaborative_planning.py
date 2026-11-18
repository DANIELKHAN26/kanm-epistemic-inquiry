"""Paired synthetic study of evidence-aware inquiry planning.

The experiment varies how much a methods audit independently checks a claim,
then compares exact provenance-aware planning with an independence comparator.
It is deliberately a small decision-theory demonstration, not human data or a
claim that KANM outperforms active-learning algorithms such as ECED.

Run from the repository root::

    python -m examples.study_collaborative_planning --replicates 500 \
      --seed-start 2026 --output research/data/collaborative_planning.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from kanm.collaborative import CollaborativeAction, optimal_collaborative_plan
from kanm.dependence import ProvenanceNetwork, corroborating_relay, relay, symmetric_primary


MIX_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
COST_WEIGHTS = (0.25, 0.5, 1.0)
METRICS = ("net_utility", "accuracy", "brier", "cost")
DECISION_UTILITIES = {
    # An intentionally asymmetric decision: false adoption costs twice as much
    # as a missed true claim. This makes the target utility explicit.
    "adopt": {True: 1.0, False: -2.0},
    "reject": {True: 0.0, False: 0.0},
}


def make_case(audit_mix: float):
    network = ProvenanceNetwork()
    network.add(symmetric_primary("pilot", 0.72, "Initial study"))
    network.add(relay("summary", "pilot", 0.98, "Summary of the initial study"))
    network.add(symmetric_primary("replication", 0.78, "Independent replication"))
    network.add(corroborating_relay(
        "audit", "pilot", fidelity=0.98, own_reliability=0.90,
        mix=audit_mix, description="Methods audit with declared independent-check mix",
    ))
    actions = (
        CollaborativeAction("pilot", "pilot", "initial_study", cost=0.15),
        CollaborativeAction("summary", "summary", "literature_summary", cost=0.05,
                            requires=("pilot",)),
        CollaborativeAction("replication", "replication", "independent_replication",
                            cost=0.70, requires=("pilot",)),
        CollaborativeAction("audit", "audit", "methods_audit", cost=0.35,
                            requires=("pilot",)),
    )
    return network, actions


def sample_world(network: ProvenanceNetwork, rng: random.Random):
    truth = rng.random() < 0.5
    reports: dict[str, bool] = {}
    for node_id, node in network.nodes.items():
        parent_values = tuple(reports[parent] for parent in node.parents)
        p_true = node.cpd[(truth, parent_values)]
        reports[node_id] = rng.random() < p_true
    return truth, reports


def enumerate_worlds(network: ProvenanceNetwork):
    """Every (truth, reports) combination with its exact probability (prior 1/2)."""
    nodes = list(network.nodes.items())
    for truth in (True, False):
        for values in itertools.product((True, False), repeat=len(nodes)):
            reports = dict(zip((node_id for node_id, _ in nodes), values))
            probability = 0.5
            for node_id, node in nodes:
                p_true = node.cpd[(truth, tuple(reports[parent] for parent in node.parents))]
                probability *= p_true if reports[node_id] else 1.0 - p_true
            if probability > 0.0:
                yield probability, truth, reports


def evaluate_policy(network, actions, plan, truth, reports, inference, cost_weight):
    by_id = {action.action_id: action for action in actions}
    observations: dict[str, bool] = {}
    queried: list[str] = []
    cost = 0.0
    while True:
        state_key = tuple(sorted(observations.items()))
        action_id = plan.policy[state_key]
        if action_id is None:
            break
        action = by_id[action_id]
        observations[action.evidence_node] = reports[action.evidence_node]
        queried.append(action.role_id)
        cost += action.cost

    if inference == "joint":
        posterior = network.posterior(0.5, observations)
    else:
        posterior = network.independence_posterior(0.5, observations)
    expected = {
        decision: posterior * utility[True] + (1.0 - posterior) * utility[False]
        for decision, utility in DECISION_UTILITIES.items()
    }
    decision = min(expected, key=lambda name: (-expected[name], name))
    utility = DECISION_UTILITIES[decision][truth]
    return {
        "decision": decision,
        "posterior": posterior,
        "accuracy": float((decision == "adopt") == truth),
        "brier": (posterior - float(truth)) ** 2,
        "cost": cost,
        "net_utility": utility - cost_weight * cost,
        "roles": queried,
        "first_role": queried[0] if queried else "stop",
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": fmean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "monte_carlo_se": stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
    }


def run(replicates: int, seed_start: int) -> dict[str, Any]:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("replicates must be an integer >= 2")
    rows = []
    for mix in MIX_LEVELS:
        network, actions = make_case(mix)
        worlds = [sample_world(network, random.Random(seed_start + i)) for i in range(replicates)]
        for cost_weight in COST_WEIGHTS:
            plans = {
                mode: optimal_collaborative_plan(
                    0.5, network, actions, DECISION_UTILITIES, horizon=3,
                    cost_weight=cost_weight, inference=mode,
                )
                for mode in ("joint", "independent")
            }
            outputs = {
                mode: [
                    evaluate_policy(network, actions, plans[mode], truth, reports, mode, cost_weight)
                    for truth, reports in worlds
                ]
                for mode in ("joint", "independent")
            }
            exact = {mode: {metric: 0.0 for metric in METRICS} for mode in plans}
            exact_difference = {metric: 0.0 for metric in METRICS}
            purchase = {mode: {action.role_id: 0.0 for action in actions} for mode in plans}
            for probability, truth, reports in enumerate_worlds(network):
                result = {
                    mode: evaluate_policy(network, actions, plans[mode], truth, reports, mode,
                                          cost_weight)
                    for mode in plans
                }
                for mode in plans:
                    for role in result[mode]["roles"]:
                        purchase[mode][role] += probability
                for metric in METRICS:
                    for mode in plans:
                        exact[mode][metric] += probability * result[mode][metric]
                    exact_difference[metric] += probability * (
                        result["joint"][metric] - result["independent"][metric])
            differences = {}
            for metric in METRICS:
                paired = [
                    outputs["joint"][i][metric] - outputs["independent"][i][metric]
                    for i in range(replicates)
                ]
                differences[f"joint_minus_independent:{metric}"] = summarize(paired)
            role_frequency = {}
            for mode in ("joint", "independent"):
                counts: dict[str, dict[str, int]] = {}
                for result in outputs[mode]:
                    for position in range(3):
                        role = result["roles"][position] if position < len(result["roles"]) else "stop"
                        position_key = f"query_{position + 1}"
                        position_counts = counts.setdefault(position_key, {})
                        position_counts[role] = position_counts.get(role, 0) + 1
                role_frequency[mode] = {
                    position: {
                        role: count / replicates for role, count in sorted(position_counts.items())
                    }
                    for position, position_counts in sorted(counts.items())
                }
            rows.append({
                "audit_independent_check_mix": mix,
                "cost_weight_utility_per_cost_unit": cost_weight,
                "replicates": replicates,
                "joint": {
                    metric: summarize([result[metric] for result in outputs["joint"]])
                    for metric in METRICS
                },
                "independent": {
                    metric: summarize([result[metric] for result in outputs["independent"]])
                    for metric in METRICS
                },
                "paired_differences": differences,
                "exact_expectation": {
                    "joint": exact["joint"],
                    "independent": exact["independent"],
                    "joint_minus_independent": exact_difference,
                    "purchase_probability": purchase,
                },
                "planner_value_check": {
                    "planner_value": plans["joint"].value,
                    "exact_expected_net_utility": exact["joint"]["net_utility"],
                    "difference": plans["joint"].value - exact["joint"]["net_utility"],
                    "interpretation": "The joint planner's value equals the exact expected net utility of its own policy under the generating model; an implementation check, not external validation.",
                },
                "role_frequency_by_query": role_frequency,
            })
    return {
        "study": "Exact-provenance versus independence-based collaborative inquiry planning",
        "design_version": 2,
        "interpretation": (
            "Synthetic finite-model decision study. It shows consequences of the declared "
            "CPDs, costs, and asymmetric decision utility; it is not human validation, "
            "real-study evidence, a calibrated cost model, or a comparison with ECED."
        ),
        "settings": {
            "truth_prior": 0.5,
            "private_study_reliability": 0.72,
            "summary_fidelity": 0.98,
            "replication_reliability": 0.78,
            "audit_independent_reliability": 0.90,
            "audit_independent_check_mix_levels": list(MIX_LEVELS),
            "action_costs": {action.action_id: action.cost for action in actions},
            "cost_weights": list(COST_WEIGHTS),
            "decision_utilities": DECISION_UTILITIES,
            "horizon": 3,
            "replicates_per_cell": replicates,
            "seed_start": seed_start,
            "common_random_numbers": "Both policies see the same generated truth and all potential reports.",
            "world_reuse": (
                f"World i is drawn from seed seed_start + i in every cell, so the {replicates} "
                "simulated worlds are shared by all 15 cells. Truth and the pilot, summary and "
                "replication reports are identical across audit mixes; the audit report reuses "
                "the same uniform draw. Cells in a row are therefore strongly correlated."
            ),
            "exact_expectation": "The model has 2 x 2^4 possible worlds; exact_expectation enumerates them, so it carries no Monte Carlo error. Monte Carlo summaries are reported alongside as a check.",
            "uncertainty": "Monte Carlo SE across simulated worlds; model uncertainty excluded.",
        },
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run(args.replicates, args.seed_start)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
