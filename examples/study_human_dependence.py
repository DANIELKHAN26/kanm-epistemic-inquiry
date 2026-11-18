"""Test KANM's exact testimony model against human ratings (research/14).

Usage::

    python -m examples.study_human_dependence DATA_DIR --download   # fetch and verify once
    python -m examples.study_human_dependence DATA_DIR [--output research/data/human_dependence_xie_hayes.json]

Data: Xie & Hayes (2022), Experiments 1–3, read through :mod:`kanm.testimony_data`. The data are not redistributed, and only aggregate results are written.

Analysis requirements: NumPy and SciPy. KANM itself has no runtime dependencies.

Every estimand, rule and seed is fixed in research/14, sections 4–8.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import log

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import norm

from kanm.testimony import trajectory
from kanm.testimony_data import Rating, download, load, scale_position

SEED = 2026
BOOTSTRAP = 10_000
FOLDS = 10
RELIABILITY = 100 / 120
CLIP = 1e-6  # probabilities are clipped to [CLIP, 1 - CLIP] only where a predictor needs finite log-odds
INFORMANTS = {1: (3, 5, 7), 2: (5, 7), 3: (3, 5)}
DEPENDENCIES = ("independent", "sequential")


# ------------------------------------------------------------------- models
def model_predictions(rule: str) -> dict[str, dict[tuple[int, str], tuple[float, ...]]]:
    """P(testimony urn) after each piece of evidence, for each model and condition."""
    out: dict[str, dict[tuple[int, str], tuple[float, ...]]] = defaultdict(dict)
    for n in (3, 5, 7):
        independent = trajectory(n, RELIABILITY, sequential=False, rule=rule)
        for dependency in DEPENDENCIES:
            sequential = dependency == "sequential"
            out["exact"][(n, dependency)] = trajectory(n, RELIABILITY, sequential=sequential, rule=rule)
            out["independence"][(n, dependency)] = trajectory(
                n, RELIABILITY, sequential=sequential, rule=rule, inference="independence")
            out["dependence_blind"][(n, dependency)] = independent
    return out


def recoded_position(p_testimony: float, scale: str) -> float:
    """Model value on the recoded 0–10 scale (10 = testimony urn)."""
    p = min(max(p_testimony, CLIP), 1 - CLIP)
    return 10.0 - scale_position(p, scale)


# ---------------------------------------------------------------- E1 to E3
def contrasts(ratings: list[Rating], experiment: int, rng: np.random.Generator) -> dict:
    """E1: human independent-minus-sequential contrasts on recoded ratings, with bootstrap draws."""
    rows = [r for r in ratings if r.experiment == experiment]
    result = {}
    draws_by_time: dict[str, list[np.ndarray]] = defaultdict(list)
    for n in INFORMANTS[experiment]:
        for time, trial in (("T", n), ("B", n + 1)):
            values: dict[str, dict[str, float]] = {d: {} for d in DEPENDENCIES}
            for r in rows:
                if r.informants == n and r.trial == trial:
                    values[r.dependency][r.participant] = 10.0 - r.rating
            if experiment == 3:  # within participants: paired differences
                people = sorted(values["independent"])
                diff = np.array([values["independent"][p] - values["sequential"][p] for p in people])
                estimate = float(diff.mean())
                index = rng.integers(0, diff.size, size=(BOOTSTRAP, diff.size))
                draws = diff[index].mean(axis=1)
                sizes = {"participants": int(diff.size)}
            else:
                a = np.array(list(values["independent"].values()))
                b = np.array(list(values["sequential"].values()))
                estimate = float(a.mean() - b.mean())
                draws = (a[rng.integers(0, a.size, size=(BOOTSTRAP, a.size))].mean(axis=1)
                         - b[rng.integers(0, b.size, size=(BOOTSTRAP, b.size))].mean(axis=1))
                sizes = {"independent": int(a.size), "sequential": int(b.size)}
            result[f"n{n}_{time}"] = {"estimate": estimate, "draws": draws, **sizes}
            draws_by_time[time].append(draws)
    for time in ("T", "B"):
        keys = [f"n{n}_{time}" for n in INFORMANTS[experiment]]
        result[f"pooled_{time}"] = {
            "estimate": float(np.mean([result[k]["estimate"] for k in keys])),
            "draws": np.mean(draws_by_time[time], axis=0),
        }
    return result


def normative_contrasts(predictions, experiment: int, scale: str) -> dict:
    """E2: model independent-minus-sequential contrasts on the recoded response scale."""
    out = {}
    for n in INFORMANTS[experiment]:
        for time, index in (("T", n - 1), ("B", n)):
            ind = predictions[(n, "independent")][index]
            seq = predictions[(n, "sequential")][index]
            out[f"n{n}_{time}"] = {
                "response_scale": recoded_position(ind, scale) - recoded_position(seq, scale),
                "log_odds": log(ind / (1 - ind)) - log(seq / (1 - seq)) if 0 < ind < 1 and 0 < seq < 1
                else None,
            }
    for time in ("T", "B"):
        keys = [f"n{n}_{time}" for n in INFORMANTS[experiment]]
        out[f"pooled_{time}"] = {"response_scale": float(np.mean([out[k]["response_scale"] for k in keys]))}
    return out


def interval(draws: np.ndarray) -> list[float]:
    low, high = np.quantile(draws, [0.025, 0.975])
    return [float(low), float(high)]


# --------------------------------------------------------------------- E4
def _cutpoints(theta: np.ndarray) -> np.ndarray:
    return theta[0] + np.concatenate(([0.0], np.cumsum(np.exp(theta[1:10]))))


def _log_likelihood(theta: np.ndarray, x: np.ndarray, y: np.ndarray, link: str) -> np.ndarray:
    """Per-observation log-likelihood of an 11-category cumulative-link model."""
    cuts = _cutpoints(theta)
    eta = cuts[None, :] - theta[10] * x[:, None]          # P(Y <= k), k = 0..9
    cdf = expit(eta) if link == "logit" else norm.cdf(eta)
    cdf = np.concatenate([np.zeros((x.size, 1)), cdf, np.ones((x.size, 1))], axis=1)
    p = cdf[np.arange(x.size), y + 1] - cdf[np.arange(x.size), y]
    return np.log(np.clip(p, 1e-300, None))


def _fit(x: np.ndarray, y: np.ndarray, link: str) -> np.ndarray:
    start = np.concatenate(([-2.0], np.full(9, np.log(0.45)), [0.5]))
    result = minimize(lambda t: -_log_likelihood(t, x, y, link).sum(), start, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"ordinal fit failed: {result.message}")
    return result.x


def model_comparison(ratings: list[Rating], experiment: int, predictions, models: tuple[str, str],
                     link: str, rng: np.random.Generator) -> dict:
    """E4: participant-level 10-fold cross-validated held-out log-likelihood, model A minus model B."""
    rows = [r for r in ratings if r.experiment == experiment]
    people = sorted({r.participant for r in rows})
    fold_of = dict(zip(people, rng.permutation(np.arange(len(people)) % FOLDS)))
    y = np.array([r.rating for r in rows])
    fold = np.array([fold_of[r.participant] for r in rows])
    person = np.array([people.index(r.participant) for r in rows])

    def predictor(model: str) -> np.ndarray:
        p = np.array([predictions[model][(r.informants, r.dependency)][r.trial - 1] for r in rows])
        p = np.clip(p, CLIP, 1 - CLIP)
        return np.log((1 - p) / p)  # log-odds for the own-ball urn; higher -> higher rating

    held_out = {}
    for model in models:
        x = predictor(model)
        ll = np.empty(y.size)
        for k in range(FOLDS):
            test = fold == k
            theta = _fit(x[~test], y[~test], link)
            ll[test] = _log_likelihood(theta, x[test], y[test], link)
        held_out[model] = ll
    diff = held_out[models[0]] - held_out[models[1]]
    per_person = np.bincount(person, weights=diff, minlength=len(people))
    counts = np.bincount(person, minlength=len(people))
    index = rng.integers(0, len(people), size=(BOOTSTRAP, len(people)))
    draws = per_person[index].sum(axis=1) / counts[index].sum(axis=1)
    return {
        "models": list(models), "link": link,
        "heldout_loglik_per_rating": {m: float(held_out[m].mean()) for m in models},
        "difference_per_rating": float(diff.mean()), "interval": interval(draws),
        "ratings": int(y.size), "participants": len(people),
    }


# -------------------------------------------------------------- reporting
def condition_means(ratings: list[Rating]) -> list[dict]:
    cells: dict[tuple, list[int]] = defaultdict(list)
    for r in ratings:
        cells[(r.experiment, r.informants, r.dependency, r.trial)].append(r.rating)
    return [{"experiment": e, "informants": n, "dependency": d, "trial": t, "n": len(v),
             "mean_rating": float(np.mean(v)), "sd_rating": float(np.std(v, ddof=1)),
             "standard_error": float(np.std(v, ddof=1) / np.sqrt(len(v)))}
            for (e, n, d, t), v in sorted(cells.items())]


def decisions(e1: dict, e2: dict) -> dict:
    """Section 6 rules, applied to the pooled contrasts."""
    out = {}
    for time in ("T", "B"):
        human, normative = e1[f"pooled_{time}"], e2[f"pooled_{time}"]["response_scale"]
        low, high = interval(human["draws"])
        ratio_low, ratio_high = interval(human["draws"] / normative)
        sign = np.sign(normative)
        out[time] = {
            "dependence_effect_detected": bool(sign * low > 0 and sign * high > 0),
            "full_sensitivity_supported": bool(ratio_low <= 1 <= ratio_high),
            "under_sensitivity": bool(ratio_low > 0 and ratio_high < 1),
        }
    return out


def run(data_dir: str) -> dict:
    ratings = load(data_dir)
    rng = np.random.default_rng(SEED)
    report = {
        "study": "Exact testimony model versus human ratings (Xie & Hayes, 2022)",
        "plan": "research/14_human_dependence_test_plan.md",
        "settings": {"reliability": RELIABILITY, "bootstrap": BOOTSTRAP, "folds": FOLDS, "seed": SEED,
                     "rating_coding": "0 = testimony urn, 10 = own-ball urn (authors' responseR); "
                                      "contrasts use 10 - rating",
                     "clip_for_log_odds": CLIP},
        "participants": {e: len({r.participant for r in ratings if r.experiment == e}) for e in (1, 2, 3)},
        "condition_means": condition_means(ratings),
        "rules": {},
    }
    for rule in ("match", "max"):
        predictions = model_predictions(rule)
        rule_report = {"model_predictions": {
            model: {f"{n}-{d}": list(values) for (n, d), values in table.items()}
            for model, table in predictions.items()}, "experiments": {}}
        for experiment in (1, 2, 3):
            scale = "log_odds" if experiment == 2 else "linear"
            e1 = contrasts(ratings, experiment, rng)
            e2 = normative_contrasts(predictions["exact"], experiment, scale)
            e3 = {}
            for key in e1:
                normative = e2[key]["response_scale"]
                e3[key] = (None if abs(normative) < 1e-9 else {
                    "estimate": e1[key]["estimate"] / normative,
                    "interval": interval(e1[key]["draws"] / normative)})
            e4 = [model_comparison(ratings, experiment, predictions, pair, link, rng)
                  for pair in (("exact", "independence"), ("exact", "dependence_blind"))
                  for link in ("logit", "probit")]
            rule_report["experiments"][str(experiment)] = {
                "E1_human_contrast": {k: {"estimate": v["estimate"], "interval": interval(v["draws"]),
                                          **{s: v[s] for s in v if s not in ("estimate", "draws")}}
                                      for k, v in e1.items()},
                "E2_normative_contrast": e2,
                "E3_sensitivity_index": e3,
                "E4_model_comparison": e4,
                "decisions": decisions(e1, e2),
            }
        report["rules"][rule] = rule_report
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.download:
        download(args.data_dir)
    report = run(args.data_dir)
    text = json.dumps(report, indent=1, allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
