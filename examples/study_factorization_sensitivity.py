"""Paired robustness sweep for Laputa versus exact-joint updating.

This controlled simulation varies inquiry accuracy, assumed source reliability,
and Erdős–Rényi network density. Each seed generates one common world and one
common random input stream for both update rules. The study is exploratory: it
tests sensitivity inside KANM's stated simulator, not human behavior.

Run::

    python -m examples.study_factorization_sensitivity --replicates 100 \
        --output research/data/factorization_sensitivity.json
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from math import sqrt
from pathlib import Path
import random
from statistics import mean, stdev
import sys

from kanm.batch import summarize
from kanm.model import AgentSpec, Simulation
from kanm.networks import agent_ids, edges_from_pairs, erdos_renyi


ACCURACY_REGIMES = {
    "above_chance": (0.55, 0.85),
    "crosses_chance": (0.35, 0.85),
}
LINK_CHANCES = (0.0, 0.1, 0.3, 0.6, 1.0)
TRUST_MEANS = (0.6, 0.8)
RULES = ("laputa", "exact_joint")
METRICS = ("veritistic_value", "mean_brier_score", "accuracy_at_half", "wrong_majority")


def _replicate(seed: int, link_chance: float, trust_mean: float,
               accuracy_regime: str, trust_points: int) -> dict[str, dict[str, float]]:
    rng = random.Random(json.dumps(["kanm-factorization-sensitivity-v1", seed]))
    ids = agent_ids(10)
    accuracy_low, accuracy_high = ACCURACY_REGIMES[accuracy_regime]
    agents = [
        AgentSpec(
            agent_id,
            prior_belief=rng.uniform(0.2, 0.8),
            inquiry_accuracy=rng.uniform(accuracy_low, accuracy_high),
            inquiry_chance=rng.uniform(0.2, 0.6),
            inquiry_trust_mean=trust_mean,
            inquiry_trust_concentration=8.0,
        )
        for agent_id in ids
    ]
    pairs = erdos_renyi(ids, link_chance, seed=seed + 1_000_003)
    edges = edges_from_pairs(
        pairs,
        listen_chance=0.5,
        assertion_threshold=0.75,
        trust_mean=trust_mean,
        trust_concentration=8.0,
    )
    truth = bool(seed % 2)
    outcomes = {}
    for rule in RULES:
        result = Simulation(
            agents,
            edges,
            truth=truth,
            steps=50,
            seed=seed,
            update_rule=rule,
            trust_points=trust_points,
            record_events=False,
        ).run()
        wrong_votes = sum(
            belief < 0.5 if truth else belief > 0.5
            for belief in result.final_beliefs.values()
        )
        outcomes[rule] = {
            "veritistic_value": result.veritistic_value,
            "mean_brier_score": result.mean_brier_score,
            "accuracy_at_half": result.accuracy_at_half,
            "wrong_majority": float(wrong_votes > len(result.final_beliefs) / 2),
        }
    return outcomes


def _summary(values: list[float]) -> dict[str, float | int]:
    stats = summarize(values)
    return {
        "n": int(stats["n"]),
        "mean": float(stats["mean"]),
        "sample_sd": float(stats["sample_sd"]),
        "monte_carlo_se": float(stats["monte_carlo_se"]),
    }


def run_study(*, replicates: int = 100, seed_start: int = 0,
              trust_points: int = 41) -> dict[str, object]:
    """Run the predeclared factorial sensitivity study and return summaries."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("replicates must be an integer >= 2")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int):
        raise ValueError("seed_start must be an integer")
    if isinstance(trust_points, bool) or not isinstance(trust_points, int) or trust_points < 3:
        raise ValueError("trust_points must be an integer >= 3")

    rows = []
    for accuracy_regime, trust_mean, link_chance in product(
        ACCURACY_REGIMES, TRUST_MEANS, LINK_CHANCES
    ):
        simulations = [
            _replicate(seed, link_chance, trust_mean, accuracy_regime, trust_points)
            for seed in range(seed_start, seed_start + replicates)
        ]
        row: dict[str, object] = {
            "accuracy_regime": accuracy_regime,
            "inquiry_accuracy_range": list(ACCURACY_REGIMES[accuracy_regime]),
            "trust_prior_mean": trust_mean,
            "trust_prior_concentration": 8.0,
            "link_chance": link_chance,
            "replicates": replicates,
        }
        for metric in METRICS:
            for rule in RULES:
                row[f"{metric}:{rule}"] = _summary(
                    [run[rule][metric] for run in simulations]
                )
            differences = [
                run["exact_joint"][metric] - run["laputa"][metric]
                for run in simulations
            ]
            row[f"{metric}:exact_minus_laputa"] = _summary(differences)
        rows.append(row)
    return {
        "study": "Laputa versus exact-joint sensitivity under generated network conditions",
        "design_version": 1,
        "interpretation": (
            "Exploratory simulator robustness only; Monte Carlo uncertainty is conditional "
            "on this design and does not establish human behavioral validity."
        ),
        "settings": {
            "agents": 10,
            "steps": 50,
            "listen_chance": 0.5,
            "assertion_threshold": 0.75,
            "trust_prior_concentration": 8.0,
            "trust_points": trust_points,
            "accuracy_regimes": ACCURACY_REGIMES,
            "link_chances": LINK_CHANCES,
            "trust_prior_means": TRUST_MEANS,
            "common_random_numbers": True,
            "truth_alternates_by_seed": True,
            "seed_start": seed_start,
            "replicates_per_cell": replicates,
            "standard_error": "sample SD divided by sqrt(replicates); paired within seed for rule differences",
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trust-points", type=int, default=41)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_study(
        replicates=args.replicates,
        seed_start=args.seed_start,
        trust_points=args.trust_points,
    )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
