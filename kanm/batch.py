"""Reproducible Monte Carlo batches and descriptive uncertainty summaries."""

from __future__ import annotations

from dataclasses import asdict
from math import sqrt
from statistics import mean, stdev
from typing import Any, Callable, Mapping, Sequence

from .model import (
    DEFAULT_ASSERTION_RULE,
    DEFAULT_SCHEDULE,
    DEFAULT_UPDATE_RULE,
    AgentSpec,
    EdgeSpec,
    Simulation,
    model_variant,
)


METRICS = (
    "final_mean_belief",
    "mean_brier_score",
    "mean_log_score",
    "accuracy_at_half",
    "undecided_fraction",
    "veritistic_value",
    "belief_dispersion",
    "belief_spread",
    "evidence_events",
    "message_events",
)


def summarize(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """Mean, sample SD and Monte Carlo standard error of defined values.

    ``None`` entries (for example an undefined log score) are excluded and
    counted in ``n_undefined`` so that they are never silently averaged.
    """
    defined = [float(v) for v in values if v is not None]
    n = len(defined)
    sample_sd = stdev(defined) if n > 1 else None
    return {
        "n": n,
        "n_undefined": len(values) - n,
        "mean": mean(defined) if n else None,
        "sample_sd": sample_sd,
        "monte_carlo_se": sample_sd / sqrt(n) if sample_sd is not None else None,
    }


def run_batch(
    agents: list[AgentSpec],
    edges: list[EdgeSpec],
    *,
    replicates: int,
    seed_start: int = 0,
    truth: bool = True,
    steps: int = 100,
    trust_points: int = 201,
    publish_only_after_inquiry: bool = False,
    update_rule: str = DEFAULT_UPDATE_RULE,
    schedule: str = DEFAULT_SCHEDULE,
    assertion_rule: str = DEFAULT_ASSERTION_RULE,
    record_trajectory: bool = False,
    record_events: bool = True,
) -> dict[str, Any]:
    """Run consecutive seeds and return raw outcomes plus Monte Carlo summaries.

    Standard errors describe Monte Carlo variation across the specified
    simulation configuration. They are not confidence intervals for real-world
    behavioral claims.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be an integer >= 1")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int):
        raise ValueError("seed_start must be an integer")

    runs = [
        Simulation(
            agents,
            edges,
            truth=truth,
            steps=steps,
            seed=seed_start + i,
            trust_points=trust_points,
            publish_only_after_inquiry=publish_only_after_inquiry,
            update_rule=update_rule,
            schedule=schedule,
            assertion_rule=assertion_rule,
            record_trajectory=record_trajectory,
            record_events=record_events,
        ).run().as_dict()
        for i in range(replicates)
    ]
    summary = {metric: summarize([run[metric] for run in runs]) for metric in METRICS}
    result: dict[str, Any] = {
        "model_variant": model_variant(update_rule, schedule),
        "runtime": runs[0]["runtime"],
        "replicates": replicates,
        "seed_start": seed_start,
        "seeds": [run["seed"] for run in runs],
        "configuration": {
            "truth": truth,
            "steps": steps,
            "trust_points": trust_points,
            "publish_only_after_inquiry": publish_only_after_inquiry,
            "update_rule": update_rule,
            "schedule": schedule,
            "assertion_rule": assertion_rule,
            "agents": [asdict(agent) for agent in agents],
            "edges": [asdict(edge) for edge in edges],
        },
        "summary": summary,
        "runs": runs,
    }
    if record_trajectory:
        length = steps + 1
        result["mean_truth_credence_trajectory"] = [
            mean(run["truth_credence_trajectory"][t] for run in runs) for t in range(length)
        ]
    return result


def paired_comparison(
    agents: list[AgentSpec],
    edges: list[EdgeSpec],
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int,
    seed_start: int = 0,
    baseline: str | None = None,
    metrics: Sequence[str] = METRICS,
    **shared: Any,
) -> dict[str, Any]:
    """Compare named model conditions on the same seeds (common random numbers).

    ``conditions`` maps a label to keyword overrides for :class:`Simulation`
    (for example ``{"update_rule": "exact_joint"}``). Each seed is run under
    every condition. Because inquiry and listening draws come from per-agent
    and per-edge streams that do not depend on beliefs, the conditions share
    their random inputs, and per-seed differences isolate the effect of the
    mechanism. Returned standard errors are for the mean paired difference.
    """
    if not conditions:
        raise ValueError("at least one condition is required")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("paired comparison needs replicates >= 2")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int):
        raise ValueError("seed_start must be an integer")
    labels = list(conditions)
    baseline = labels[0] if baseline is None else baseline
    if baseline not in conditions:
        raise ValueError(f"unknown baseline condition: {baseline}")
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s): {unknown}")
    forbidden = {"seed", "record_events"}
    if forbidden & set(shared):
        raise ValueError(f"shared settings may not include {sorted(forbidden)}")
    for label, overrides in conditions.items():
        if forbidden & set(overrides):
            raise ValueError(f"condition {label} may not override {sorted(forbidden)}")

    outcomes: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for i in range(replicates):
        for label in labels:
            kwargs = {**shared, **conditions[label]}
            run = Simulation(
                agents, edges, seed=seed_start + i, record_events=False, **kwargs
            ).run().as_dict()
            outcomes[label].append(run)

    report: dict[str, Any] = {
        "replicates": replicates,
        "seed_start": seed_start,
        "baseline": baseline,
        "conditions": {label: dict(conditions[label]) for label in labels},
        "shared_settings": dict(shared),
        "summary": {
            label: {m: summarize([run[m] for run in outcomes[label]]) for m in metrics}
            for label in labels
        },
        "paired_difference_vs_baseline": {},
    }
    for label in labels:
        if label == baseline:
            continue
        differences: dict[str, Any] = {}
        for m in metrics:
            diffs: list[float | None] = []
            for run, base in zip(outcomes[label], outcomes[baseline]):
                a, b = run[m], base[m]
                diffs.append(None if a is None or b is None else float(a) - float(b))
            differences[m] = summarize(diffs)
        report["paired_difference_vs_baseline"][label] = differences
    return report


def sweep(
    build: Callable[[Mapping[str, Any]], tuple[list[AgentSpec], list[EdgeSpec], dict[str, Any]]],
    grid: Mapping[str, Sequence[Any]],
    *,
    replicates: int,
    seed_start: int = 0,
    metrics: Sequence[str] = METRICS,
) -> list[dict[str, Any]]:
    """Full-factorial sweep returning one summary row per design point.

    ``build`` receives a design point (a dict of factor values) and returns
    ``(agents, edges, simulation_kwargs)``. Every design point uses the same
    seeds, so differences between rows are paired by seed.
    """
    if not grid:
        raise ValueError("grid must contain at least one factor")
    factors = list(grid)
    for factor in factors:
        if not isinstance(grid[factor], (list, tuple)) or not grid[factor]:
            raise ValueError(f"factor {factor} needs a nonempty list of levels")
    points: list[dict[str, Any]] = [{}]
    for factor in factors:
        points = [{**p, factor: level} for p in points for level in grid[factor]]
    rows = []
    for point in points:
        agents, edges, kwargs = build(point)
        reserved = {"seed", "seed_start", "replicates", "record_events"} & set(kwargs)
        if reserved:
            raise ValueError(f"build() may not set {sorted(reserved)}; pass them to sweep()")
        batch = run_batch(
            agents,
            edges,
            replicates=replicates,
            seed_start=seed_start,
            record_events=False,
            **kwargs,
        )
        row: dict[str, Any] = dict(point)
        row["replicates"] = replicates
        for m in metrics:
            row[f"{m}_mean"] = batch["summary"][m]["mean"]
            row[f"{m}_mcse"] = batch["summary"][m]["monte_carlo_se"]
        rows.append(row)
    return rows
