"""Post hoc diagnostics for research/10 §5.5. These were chosen after seeing the results and are exploratory.

Usage::

    python -m examples.crowd_posthoc DATA_DIR [--datasets snow2008_rte ...]

The script computes two diagnostics:

(a) **Worker asymmetry.** For each worker, it computes empirical sensitivity and specificity against gold. A worker is included only with at least 20 gold-scored reports and at least ``--min-per-class`` reports on each class.

(b) **Two-coin Dawid–Skene with the class prior re-estimated** in every M-step. It uses Beta(2, 1) pseudo-counts on sensitivity and specificity, starts from majority vote, and stops when posteriors change by less than 1e-9 or after 3000 iterations.

It prints one JSON document.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from math import log

from kanm.bayes import expit
from kanm.crowd_data import DATASETS, load
from kanm.reliability import CrowdData, Estimate, majority_vote, score


def worker_asymmetry(data: CrowdData, *, min_reports: int = 20, min_per_class: int = 5) -> dict:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # tp, positives, tn, negatives
    for item, worker, label in data.labels:
        if item not in data.gold:
            continue
        c = counts[worker]
        if data.gold[item] == 1:
            c[0] += label == 1
            c[1] += 1
        else:
            c[2] += label == 0
            c[3] += 1
    kept = [c for c in counts.values()
            if c[1] + c[3] >= min_reports and c[1] >= min_per_class and c[3] >= min_per_class]
    gaps = [abs(c[0] / c[1] - c[2] / c[3]) for c in kept]
    return {
        "workers": len(kept),
        "median_gap": statistics.median(gaps) if gaps else None,
        "share_gap_above_0_3": sum(g > 0.3 for g in gaps) / len(gaps) if gaps else None,
        "below_chance": sum((c[0] + c[2]) / (c[1] + c[3]) < 0.5 for c in kept),
    }


def two_coin_estimated_prior(data: CrowdData, *, max_iterations: int = 3000,
                             tolerance: float = 1e-9) -> dict:
    posterior = dict(majority_vote(data).posteriors)
    converged = False
    for iteration in range(1, max_iterations + 1):
        prior = min(max(sum(posterior.values()) / len(posterior), 1e-6), 1 - 1e-6)
        stats: dict[str, list[float]] = {}
        for item, worker, label in data.labels:
            p = posterior[item]
            s = stats.setdefault(worker, [0.0, 0.0, 0.0, 0.0])
            s[0] += p * (label == 1)
            s[1] += p
            s[2] += (1 - p) * (label == 0)
            s[3] += 1 - p
        sens = {w: min(max((s[0] + 1) / (s[1] + 1), 1e-12), 1 - 1e-12) for w, s in stats.items()}
        spec = {w: min(max((s[2] + 1) / (s[3] + 1), 1e-12), 1 - 1e-12) for w, s in stats.items()}
        log_odds = {item: log(prior) - log(1 - prior) for item in posterior}
        for item, worker, label in data.labels:
            if label == 1:
                log_odds[item] += log(sens[worker]) - log(1 - spec[worker])
            else:
                log_odds[item] += log(1 - sens[worker]) - log(spec[worker])
        updated = {item: expit(value) for item, value in log_odds.items()}
        change = max(abs(updated[item] - posterior[item]) for item in updated)
        posterior = updated
        if change < tolerance:
            converged = True
            break
    accuracy = score(Estimate("two_coin_estimated_prior", posterior, {}), data.gold)["accuracy"]
    return {"estimated_class_prior": prior, "accuracy": accuracy,
            "iterations": iteration, "converged": converged}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--min-per-class", type=int, default=5)
    args = parser.parse_args(argv)
    report = {}
    for name in args.datasets:
        data = load(name, args.data_dir)
        report[name] = {
            "post_hoc": True,
            "worker_asymmetry": worker_asymmetry(data, min_per_class=args.min_per_class),
            "two_coin_estimated_prior": two_coin_estimated_prior(data),
        }
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
