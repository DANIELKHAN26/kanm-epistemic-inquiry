"""Dock KANM against the compiled Laputa 1.6 model core.

Usage (after ``build.sh``)::

    python -m docking.laputa_1_6.compare HARNESS [--scenarios 200] [--steps 25] [--seed 0]
    python -m docking.laputa_1_6.compare HARNESS --stochastic [--runs 400]
    python -m docking.laputa_1_6.compare HARNESS --emulate-float32   # post hoc diagnostic

Deterministic docking (default)
-------------------------------
Every chance (inquiry chance, inquiry accuracy, listen chance) is exactly 0 or
1, so neither program consumes randomness that affects the outcome and the two
trajectories must agree step by step. Each scenario draws a random directed
network (2-8 inquirers), starting beliefs, thresholds (including some below
0.5, i.e. liars), new-evidence policies, count-prior flags and trust functions
(1.6 default-batch draws, 1.6 presets and Beta samples). KANM runs with
``schedule="laputa_1_6"``, ``assertion_rule="laputa_1_6"``,
``trust_model="laputa_1_6"`` and the same node values; inquirer ``i`` is mapped
to ID ``a{i:03d}`` so that sorted-ID order equals Laputa's index order.

Pre-declared acceptance criteria (fixed before the first comparison):

* every credence agrees to ``|delta| <= 1e-5`` at every step;
* every link and inquiry trust expectation agrees to ``|delta| <= 1e-5``;
* cumulative counts of delivered messages and inquiry results are identical.

The tolerance reflects that Laputa stores trust in single precision (about 7
significant digits) and KANM in double precision. A scenario is reported as
*indeterminate* rather than failed if, at some step, a decision in either
program depends on a comparison within ``1e-6`` of its cut point (a credence
within 1e-6 of 0.5, or of a link threshold under the "inquiry"/"any"
policies), because single-precision rounding can then legitimately flip it.

Stochastic docking (``--stochastic``)
-------------------------------------
With chances strictly between 0 and 1 the programs use different random
generators, so only distributions can be compared. For each of a few fixed
scenarios both programs are run ``--runs`` times with independent seeds and the
mean final credence of each inquirer is compared with a two-sample z statistic.
Pre-declared criterion: every ``|z| <= 4`` (with at most ~40 statistics, a
false alarm under correct docking has probability below 0.003).

Single-precision emulation (``--emulate-float32``, post hoc)
-------------------------------------------------------------
Added after the first deterministic run failed the criterion in 1 of 200
scenarios (see README.md). It runs KANM with Laputa's single-precision trust
arithmetic (``float32.py``) to test whether precision explains the residuals.
It is a diagnostic, not an acceptance test, and is labelled in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from typing import Any

from kanm.laputa16 import NODES, Laputa16Trust
from kanm.model import AgentSpec, EdgeSpec, Simulation

TOLERANCE = 1e-5
CUT_MARGIN = 1e-6
Z_LIMIT = 4.0
POLICIES = {"none": 0, "inquiry": 1, "any": 2}


def random_trust(rng: random.Random) -> tuple[float, ...]:
    kind = rng.random()
    if kind < 0.5:
        return Laputa16Trust.default_batch(rng.random()).values
    if kind < 0.75:
        return Laputa16Trust.from_value_sharpness(rng.uniform(0.05, 0.95), rng.uniform(0.2, 0.8)).values
    mean = rng.uniform(0.25, 0.75)
    concentration = rng.uniform(max(1 / mean, 1 / (1 - mean)), 30.0)
    return Laputa16Trust.from_beta(mean, concentration).values


def random_belief(rng: random.Random) -> float:
    while True:
        value = rng.uniform(0.02, 0.98)
        if abs(value - 0.5) > 0.02:
            return value


def random_scenario(rng: random.Random, deterministic: bool) -> dict[str, Any]:
    n = rng.randint(2, 8)
    policy = rng.choice(list(POLICIES))

    def chance() -> float:
        return float(rng.random() < 0.6) if deterministic else rng.uniform(0.1, 0.9)

    inquirers = [
        {
            "belief": random_belief(rng),
            "chance": chance(),
            "accuracy": float(rng.random() < 0.7) if deterministic else rng.uniform(0.4, 0.9),
            "trust": random_trust(rng),
        }
        for _ in range(n)
    ]
    links = []
    density = rng.uniform(0.2, 0.8)
    for s in range(n):
        for t in range(n):
            if s != t and rng.random() < density:
                threshold = rng.uniform(0.55, 0.95) if rng.random() < 0.85 else rng.uniform(0.05, 0.45)
                links.append({
                    "source": s, "target": t, "listen": chance(), "threshold": threshold,
                    "count_prior": rng.random() < 0.3, "trust": random_trust(rng),
                })
    return {"inquirers": inquirers, "links": links, "policy": policy,
            "count_prior": rng.random() < 0.3}


def harness_input(scenario: dict[str, Any], steps: int, seed: int) -> str:
    policy = POLICIES[scenario["policy"]]
    count_prior = int(scenario["count_prior"])
    lines = ["mode run", f"seed {seed}", f"steps {steps}", f"inquirers {len(scenario['inquirers'])}"]
    for q in scenario["inquirers"]:
        lines.append(" ".join([repr(q["belief"]), repr(q["chance"]), repr(q["accuracy"]), "1",
                               *map(repr, q["trust"])]))
    lines.append(f"links {len(scenario['links'])}")
    for link in scenario["links"]:
        lines.append(" ".join([str(link["source"]), str(link["target"]), repr(link["listen"]),
                               repr(link["threshold"]), "1", str(policy), str(count_prior),
                               *map(repr, link["trust"])]))
    return "\n".join(lines) + "\n"


def run_harness(harness: str, scenario: dict[str, Any], steps: int, seed: int) -> list[dict]:
    completed = subprocess.run(
        [harness], input=harness_input(scenario, steps, seed), capture_output=True,
        text=True, timeout=600, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"harness failed ({completed.returncode}): {completed.stderr.strip()}")
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def agent_id(index: int) -> str:
    return f"a{index:03d}"


def kanm_simulation(scenario: dict[str, Any], steps: int, seed: int) -> Simulation:
    agents = [
        AgentSpec(agent_id(i), q["belief"], q["accuracy"], q["chance"],
                  inquiry_trust_values=tuple(q["trust"]))
        for i, q in enumerate(scenario["inquirers"])
    ]
    edges = [
        EdgeSpec(agent_id(l["source"]), agent_id(l["target"]), l["listen"], l["threshold"],
                 trust_values=tuple(l["trust"]))
        for l in scenario["links"]
    ]
    return Simulation(
        agents, edges, truth=True, steps=steps, seed=seed, schedule="laputa_1_6",
        assertion_rule="laputa_1_6", trust_model="laputa_1_6",
        evidence_policy=scenario["policy"], count_prior_as_evidence=scenario["count_prior"],
        record_events=False,
    )


def kanm_trajectory(scenario: dict[str, Any], steps: int) -> list[dict[str, Any]]:
    """KANM state after 0..steps steps (fresh deterministic runs of each length)."""
    out = []
    for k in range(steps + 1):
        result = kanm_simulation(scenario, k, seed=0).run()
        out.append({
            "beliefs": [result.final_beliefs[agent_id(i)] for i in range(len(scenario["inquirers"]))],
            "inquiry": [result.final_trust[f"{agent_id(i)}:inquiry"]
                        for i in range(len(scenario["inquirers"]))],
            "links": {f"{l['source']}>{l['target']}":
                      result.final_trust[f"{agent_id(l['source'])}->{agent_id(l['target'])}"]
                      for l in scenario["links"]},
            "messages": result.message_events,
            "inquiries": result.evidence_events,
        })
    return out


def near_cut(scenario: dict[str, Any], states: list[list[float]]) -> bool:
    thresholds = [l["threshold"] for l in scenario["links"]]
    for beliefs in states:
        for b in beliefs:
            if abs(b - 0.5) < CUT_MARGIN:
                return True
            if scenario["policy"] != "none":
                for t in thresholds:
                    if abs(b - t) < CUT_MARGIN or abs(b - (1 - t)) < CUT_MARGIN:
                        return True
    return False


def deterministic(harness: str, scenarios: int, steps: int, seed: int,
                  emulate_float32: bool = False, records: str | None = None) -> int:
    rng = random.Random(seed)
    passed = failed = indeterminate = 0
    worst_belief = worst_trust = 0.0
    failures = []
    rows = []
    for index in range(scenarios):
        scenario = random_scenario(rng, deterministic=True)
        laputa = run_harness(harness, scenario, steps, seed=index)
        kanm = kanm_trajectory(scenario, steps)
        if near_cut(scenario, [s["beliefs"] for s in laputa] + [s["beliefs"] for s in kanm]):
            indeterminate += 1
            rows.append({"scenario": index, "policy": scenario["policy"],
                         "inquirers": len(scenario["inquirers"]), "links": len(scenario["links"]),
                         "outcome": "indeterminate"})
            continue
        belief_diff = max(abs(a - b) for L, K in zip(laputa, kanm)
                          for a, b in zip(L["beliefs"], K["beliefs"]))
        trust_diff = max(
            [abs(a - b) for L, K in zip(laputa, kanm)
             for a, b in zip(L["inquiry_trust_expectation"], K["inquiry"])]
            + [abs(L["link_trust_expectation"][key] - K["links"][key])
               for L, K in zip(laputa, kanm) for key in K["links"]]
        )
        counts_equal = all(L["messages_delivered"] == K["messages"]
                           and L["inquiry_results"] == K["inquiries"] for L, K in zip(laputa, kanm))
        worst_belief, worst_trust = max(worst_belief, belief_diff), max(worst_trust, trust_diff)
        ok = belief_diff <= TOLERANCE and trust_diff <= TOLERANCE and counts_equal
        rows.append({"scenario": index, "policy": scenario["policy"],
                     "inquirers": len(scenario["inquirers"]), "links": len(scenario["links"]),
                     "outcome": "passed" if ok else "failed", "max_belief_diff": belief_diff,
                     "max_trust_diff": trust_diff, "counts_equal": counts_equal})
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append({"scenario": index, "policy": scenario["policy"],
                             "max_belief_diff": belief_diff, "max_trust_diff": trust_diff,
                             "counts_equal": counts_equal})
    report = {"mode": "deterministic", "emulate_float32_post_hoc": emulate_float32,
              "scenarios": scenarios, "steps": steps, "seed": seed,
              "passed": passed, "failed": failed, "indeterminate": indeterminate,
              "max_belief_diff": worst_belief, "max_trust_diff": worst_trust,
              "tolerance": TOLERANCE, "failures": failures[:20]}
    if records:
        report["records"] = records
        with open(records, "w", encoding="utf-8") as handle:
            json.dump({key: value for key, value in report.items() if key != "records"}
                      | {"rows": rows}, handle, indent=1, allow_nan=False)
            handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if failed == 0 and passed > 0 else 1


def stochastic(harness: str, runs: int, steps: int, seed: int, scenarios: int = 5,
               emulate_float32: bool = False) -> int:
    rng = random.Random(seed)
    worst, rows = 0.0, []
    for index in range(scenarios):
        scenario = random_scenario(rng, deterministic=False)
        n = len(scenario["inquirers"])
        laputa_final = [run_harness(harness, scenario, steps, seed=1_000_000 + r)[-1]["beliefs"]
                        for r in range(runs)]
        kanm_final = []
        for r in range(runs):
            result = kanm_simulation(scenario, steps, seed=2_000_000 + r).run()
            kanm_final.append([result.final_beliefs[agent_id(i)] for i in range(n)])
        for i in range(n):
            a = [x[i] for x in laputa_final]
            b = [x[i] for x in kanm_final]
            ma, mb = sum(a) / runs, sum(b) / runs
            va = sum((x - ma) ** 2 for x in a) / (runs - 1)
            vb = sum((x - mb) ** 2 for x in b) / (runs - 1)
            se = math.sqrt(va / runs + vb / runs)
            z = 0.0 if se == 0 else (ma - mb) / se
            worst = max(worst, abs(z))
            rows.append({"scenario": index, "policy": scenario["policy"], "inquirer": i,
                         "laputa_mean": ma, "kanm_mean": mb, "z": z})
    print(json.dumps({"mode": "stochastic", "emulate_float32_post_hoc": emulate_float32,
                      "runs": runs, "steps": steps, "seed": seed,
                      "max_abs_z": worst, "z_limit": Z_LIMIT, "rows": rows}, indent=2))
    return 0 if worst <= Z_LIMIT else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("harness", help="path to the built laputa16_harness executable")
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--runs", type=int, default=400)
    parser.add_argument("--records", metavar="PATH",
                        help="deterministic mode: also write per-scenario results as JSON")
    parser.add_argument("--emulate-float32", action="store_true",
                        help="post hoc diagnostic: KANM uses Laputa's single-precision trust arithmetic "
                             "(requires numpy)")
    args = parser.parse_args(argv)
    if args.emulate_float32:
        from docking.laputa_1_6.float32 import emulated
        context = emulated()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    with context:
        if args.stochastic:
            return stochastic(args.harness, args.runs, args.steps, args.seed,
                              emulate_float32=args.emulate_float32)
        return deterministic(args.harness, args.scenarios, args.steps, args.seed,
                             emulate_float32=args.emulate_float32, records=args.records)


if __name__ == "__main__":
    sys.exit(main())
