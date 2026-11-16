"""Exploratory paired study: factorized (Laputa) versus exact-joint updating.

Question: holding everything else fixed (agents, network, inquiry results and
listening opportunities), how much do collective outcomes change when agents
use the exact joint posterior instead of Laputa's product-of-marginals rule?

Design (stated before running; all settings are illustrative, not calibrated):
* 10 agents; directed Erdos-Renyi networks with link chance in {0, 0.1, 0.3, 0.6, 1.0}.
* Per replicate: prior ~ U(0.2, 0.8), inquiry accuracy ~ U(0.55, 0.85),
  inquiry chance ~ U(0.2, 0.6); trust priors Beta(mean 0.6, concentration 8);
  listen chance 0.5; published assertion rule, threshold 0.75; 50 steps;
  simultaneous schedule; truth = p.
* Each replicate runs both rules with the same seed (common random numbers);
  we report per-rule means and the mean paired difference with its Monte Carlo
  standard error.

Outcomes: veritistic value, Brier score, accuracy at 0.5, and the share of
runs in which a majority ends with credence below 0.5 in the truth ("wrong
majority"). Monte Carlo SEs quantify simulation noise for this design only.

Run:  python -m examples.study_factorization [replicates]
"""

from __future__ import annotations

import json
import random
import sys

from kanm.batch import summarize
from kanm.model import AgentSpec, Simulation
from kanm.networks import agent_ids, edges_from_pairs, erdos_renyi

LINK_CHANCES = (0.0, 0.1, 0.3, 0.6, 1.0)
RULES = ("laputa", "exact_joint")
METRICS = ("veritistic_value", "mean_brier_score", "accuracy_at_half", "wrong_majority")


def replicate(seed: int, link_chance: float) -> dict[str, dict[str, float]]:
    rng = random.Random(json.dumps(["factorization-study", seed]))
    ids = agent_ids(10)
    agents = [
        AgentSpec(
            agent_id,
            prior_belief=rng.uniform(0.2, 0.8),
            inquiry_accuracy=rng.uniform(0.55, 0.85),
            inquiry_chance=rng.uniform(0.2, 0.6),
        )
        for agent_id in ids
    ]
    edges = edges_from_pairs(
        erdos_renyi(ids, link_chance, seed=seed), listen_chance=0.5, assertion_threshold=0.75
    )
    outcome = {}
    for rule in RULES:
        result = Simulation(agents, edges, steps=50, seed=seed, update_rule=rule,
                            trust_points=101, record_events=False).run()
        wrong = sum(b < 0.5 for b in result.final_beliefs.values()) > len(agents) / 2
        outcome[rule] = {
            "veritistic_value": result.veritistic_value,
            "mean_brier_score": result.mean_brier_score,
            "accuracy_at_half": result.accuracy_at_half,
            "wrong_majority": float(wrong),
        }
    return outcome


def main(replicates: int = 200) -> dict:
    rows = []
    for link_chance in LINK_CHANCES:
        runs = [replicate(seed, link_chance) for seed in range(replicates)]
        row = {"link_chance": link_chance, "replicates": replicates}
        for metric in METRICS:
            for rule in RULES:
                stats = summarize([r[rule][metric] for r in runs])
                row[f"{metric}:{rule}"] = [stats["mean"], stats["monte_carlo_se"]]
            diff = summarize([r["exact_joint"][metric] - r["laputa"][metric] for r in runs])
            row[f"{metric}:exact_minus_laputa"] = [diff["mean"], diff["monte_carlo_se"]]
        rows.append(row)
    return {"design": __doc__.split("Outcomes")[0].strip(), "rows": rows}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(json.dumps(main(n), indent=2))
