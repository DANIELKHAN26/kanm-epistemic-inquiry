"""Qualitative reproduction attempt: Vallinder & Olsson (2014), Fig. 1.

Published setup (V&O 2014, footnote 9, as verified from the author copy):
no communication (link chance 0), 20 agents, 100 steps, starting belief and
inquiry chance flat on [0, 1], inquiry reliability R in {0.6, 0.7, 0.8}, and
veritistic value V plotted against the expected inquiry trust. The published
curves rise steeply from E[trust] ~ 0.52 to about 0.7 and are nearly flat from
0.7 to 1 (approximate values read from Fig. 1: R=0.6: 0.16 -> 0.385 -> 0.38;
R=0.7: 0.28 -> 0.46 -> 0.47; R=0.8: 0.35 -> 0.475 -> 0.485).

Not reported in the paper: the shape/spread of the inquiry-trust function and
the number of trials. We therefore run two Beta concentrations and report
both; agreement can only be qualitative. E[trust] = 1 is replaced by 0.99
because KANM rejects certain-reliability priors.

Run:  python -m examples.reproduce_vallinder_olsson_2014 [trials]
"""

from __future__ import annotations

import json
import random
import sys

from kanm.batch import summarize
from kanm.model import AgentSpec, Simulation

RELIABILITIES = (0.6, 0.7, 0.8)
TRUST_MEANS = (0.52, 0.6, 0.7, 0.8, 0.9, 0.99)
CONCENTRATIONS = (4.0, 16.0)


def trial(seed: int, reliability: float, trust_mean: float, concentration: float) -> float:
    rng = random.Random(json.dumps(["vo2014", seed]))
    agents = [
        AgentSpec(
            f"a{i:02d}",
            prior_belief=rng.random(),
            inquiry_chance=rng.random(),
            inquiry_accuracy=reliability,
            inquiry_trust_mean=trust_mean,
            inquiry_trust_concentration=concentration,
        )
        for i in range(20)
    ]
    result = Simulation(agents, [], steps=100, seed=seed, trust_points=101,
                        record_events=False).run()
    return result.veritistic_value


def main(trials: int = 100) -> dict:
    rows = []
    for concentration in CONCENTRATIONS:
        for reliability in RELIABILITIES:
            for trust_mean in TRUST_MEANS:
                # Same seeds (agent priors/activity and inquiry draws) in every cell.
                values = [trial(s, reliability, trust_mean, concentration) for s in range(trials)]
                stats = summarize(values)
                rows.append({
                    "concentration": concentration,
                    "reliability": reliability,
                    "trust_mean": trust_mean,
                    "v_mean": stats["mean"],
                    "v_mcse": stats["monte_carlo_se"],
                })
    return {"trials_per_cell": trials, "rows": rows}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    output = main(n)
    print(json.dumps(output, indent=2))
