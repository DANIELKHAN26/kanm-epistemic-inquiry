"""Structure sensitivity in Fränken et al. (2024), Experiment 1 (research/15).

Usage::

    python -m examples.study_structure_sensitivity DATA_DIR --download
    python -m examples.study_structure_sensitivity DATA_DIR [--output research/data/structure_sensitivity_franken.json]

Every estimand, rule and seed is fixed in research/15, sections 4–8. The script needs NumPy and SciPy and writes only aggregate results.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import log

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from kanm import judgment_inversion as J
from kanm.structure_data import OWN_EVIDENCE, STRONG_BLUE, WEAK_RED, Participant, download, load

SEED = 2026
BOOTSTRAP = 10_000
FOLDS = 10
WINDOW = slice(5, 10)  # trials 6–10
CONDITIONS = ("c_sees_b", "none", "b_sees_c")


def model_bins() -> dict[str, dict[str, list[list[float]]]]:
    """Bin probabilities for A at each trial, by model and structured condition."""
    b = J.infer_evidence(STRONG_BLUE)
    c = J.infer_evidence(WEAK_RED)
    level1 = J.listener_bins(OWN_EVIDENCE, [b, c])
    return {
        "level1": {"c_sees_b": level1, "b_sees_c": level1},
        "level2": {
            "c_sees_b": J.listener_bins(OWN_EVIDENCE, [b, J.infer_evidence(WEAK_RED, parents=[b])]),
            "b_sees_c": J.listener_bins(OWN_EVIDENCE, [J.infer_evidence(STRONG_BLUE, parents=[c]), c]),
        },
    }


def judgments(p: Participant, include_autofilled: bool) -> list[int | None]:
    return [j if include_autofilled or not filled else None for j, filled in zip(p.judgments, p.autofilled)]


def window_mean(p: Participant, include_autofilled: bool) -> float | None:
    values = [j for j in judgments(p, include_autofilled)[WINDOW] if j is not None]
    return float(np.mean(values)) if values else None


def interval(draws: np.ndarray) -> list[float]:
    low, high = np.quantile(draws, [0.025, 0.975])
    return [float(low), float(high)]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return values[rng.integers(0, values.size, size=(BOOTSTRAP, values.size))].mean(axis=1)


def structure_effect(participants: list[Participant], include_autofilled: bool,
                     rng: np.random.Generator) -> dict:
    """E1 (and E5): window means by condition and the C-sees-B minus B-sees-C contrast."""
    means, draws, dropped = {}, {}, {}
    for condition in CONDITIONS:
        values = [window_mean(p, include_autofilled) for p in participants if p.condition == condition]
        dropped[condition] = sum(v is None for v in values)
        array = np.array([v for v in values if v is not None])
        means[condition] = {"mean": float(array.mean()), "n": int(array.size)}
        draws[condition] = bootstrap_mean(array, rng)
        means[condition]["interval"] = interval(draws[condition])
    contrast = draws["c_sees_b"] - draws["b_sees_c"]
    return {
        "condition_means": means,
        "dropped_without_window_judgment": dropped,
        "contrast": {"estimate": means["c_sees_b"]["mean"] - means["b_sees_c"]["mean"],
                     "interval": interval(contrast)},
        "_draws": contrast,
    }


def per_trial_means(participants: list[Participant]) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        for trial in range(10):
            values = np.array([p.judgments[trial] for p in participants
                               if p.condition == condition and not p.autofilled[trial]], dtype=float)
            rows.append({"condition": condition, "trial": trial + 1, "n": int(values.size),
                         "mean": float(values.mean()),
                         "standard_error": float(values.std(ddof=1) / np.sqrt(values.size))})
    return rows


def normative_effect(bins) -> dict:
    out = {}
    for model, table in bins.items():
        window = {c: float(np.mean([J.expected_judgment(b) for b in table[c][WINDOW]])) for c in table}
        out[model] = {"c_sees_b": window["c_sees_b"], "b_sees_c": window["b_sees_c"],
                      "contrast": window["c_sees_b"] - window["b_sees_c"],
                      "expected_judgment_by_trial": {c: [J.expected_judgment(b) for b in table[c]]
                                                     for c in table}}
    return out


def _log_likelihoods(group: list[Participant], bins: dict[str, list[list[float]]], tau: float,
                     sticky: float) -> np.ndarray:
    """Per-participant summed log-likelihood and judgment counts under one model."""
    out = np.zeros((len(group), 2))
    for i, p in enumerate(group):
        values = [v for v in J.judgment_log_likelihood(judgments(p, False), bins[p.condition], tau, sticky)
                  if v is not None]
        out[i] = (sum(values), len(values))
    return out


def _fit(group: list[Participant], bins, sticky_model: bool) -> tuple[float, float]:
    def loss(theta):
        tau = float(np.exp(theta[0]))
        sticky = float(expit(theta[1])) if sticky_model else 0.0
        return -_log_likelihoods(group, bins, tau, sticky)[:, 0].sum()
    start = np.array([log(2.0), logit(0.3)]) if sticky_model else np.array([log(2.0)])
    result = minimize(loss, start, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"fit failed: {result.message}")
    tau = float(np.exp(result.x[0]))
    return tau, float(expit(result.x[1])) if sticky_model else 0.0


def model_comparison(participants: list[Participant], bins, sticky_model: bool,
                     rng: np.random.Generator) -> dict:
    """E4: 10-fold participant-level CV, held-out log-likelihood per judgment, Level-2 minus Level-1."""
    group = [p for p in participants if p.condition != "none"]
    folds = rng.permutation(np.arange(len(group)) % FOLDS)
    held = {}
    fitted = {}
    for model in ("level1", "level2"):
        result = np.zeros((len(group), 2))
        params = []
        for k in range(FOLDS):
            train = [p for p, f in zip(group, folds) if f != k]
            test_index = [i for i, f in enumerate(folds) if f == k]
            tau, sticky = _fit(train, bins[model], sticky_model)
            params.append((tau, sticky))
            result[test_index] = _log_likelihoods([group[i] for i in test_index], bins[model], tau, sticky)
        held[model] = result
        fitted[model] = {"tau_median": float(np.median([t for t, _ in params])),
                         "sticky_median": float(np.median([s for _, s in params]))}
    diff = held["level2"][:, 0] - held["level1"][:, 0]
    counts = held["level1"][:, 1]
    index = rng.integers(0, len(group), size=(BOOTSTRAP, len(group)))
    draws = diff[index].sum(axis=1) / counts[index].sum(axis=1)
    return {"sticky": sticky_model, "participants": len(group), "judgments": int(counts.sum()),
            "heldout_loglik_per_judgment": {m: float(held[m][:, 0].sum() / counts.sum()) for m in held},
            "difference_level2_minus_level1": float(diff.sum() / counts.sum()),
            "interval": interval(draws), "fitted_parameters": fitted}


def decisions(e1: dict, e2: float, e4: list[dict]) -> dict:
    low, high = e1["contrast"]["interval"]
    detected = bool(np.sign(e2) * low > 0 and np.sign(e2) * high > 0)
    index_low, index_high = interval(e1["_draws"] / e2)
    band = None
    if detected:
        if index_low <= 1 <= index_high:
            band = "full"
        elif index_low > 0 and index_high < 1:
            band = "partial"
        elif index_low > 1:
            band = "greater_than_normative"
        else:
            band = "indeterminate"
    preferences = {}
    for item in e4:
        lo, hi = item["interval"]
        preferences["sticky" if item["sticky"] else "plain"] = (
            "level2" if lo > 0 else "level1" if hi < 0 else "no_preference")
    return {"structure_effect_detected": detected, "sensitivity_band": band,
            "model_preference": preferences}


def run(data_dir: str) -> dict:
    participants = load(data_dir)
    rng = np.random.default_rng(SEED)
    bins = model_bins()
    e2 = normative_effect(bins)
    report = {"study": "Structure sensitivity, Fränken et al. (2024) Experiment 1",
              "plan": "research/15_structure_sensitivity_test_plan.md",
              "settings": {"bootstrap": BOOTSTRAP, "folds": FOLDS, "seed": SEED, "window": "trials 6-10",
                           "evidence_prior": "uniform", "tau_for_inversion": 10.0},
              "participants": {c: sum(p.condition == c for p in participants) for c in CONDITIONS},
              "trial_means": per_trial_means(participants),
              "E2_normative": e2, "analyses": {}}
    for label, include in (("primary_autofilled_missing", False), ("sensitivity_autofilled_included", True)):
        e1 = structure_effect(participants, include, rng)
        normative = e2["level2"]["contrast"]
        index_draws = e1["_draws"] / normative
        e4 = [model_comparison(participants, bins, sticky, rng) for sticky in (False, True)] if not include else []
        entry = {"E1_human": {k: v for k, v in e1.items() if k != "_draws"},
                 "E3_sensitivity_index": {"estimate": e1["contrast"]["estimate"] / normative,
                                          "interval": interval(index_draws)}}
        if e4:
            entry["E4_model_comparison"] = e4
            entry["decisions"] = decisions(e1, normative, e4)
        else:
            entry["decisions"] = decisions(e1, normative, [])
        report["analyses"][label] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.download:
        download(args.data_dir)
    text = json.dumps(run(args.data_dir), indent=1, allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
