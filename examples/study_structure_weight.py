"""Structure-correction weight w in Fränken et al. (2024), Experiments 1–3 (research/16).

Usage::

    python -m examples.study_structure_weight DATA_DIR --download
    python -m examples.study_structure_weight DATA_DIR [--output research/data/structure_weight_franken.json]

w = 0 is the structure-blind Level-1 model and w = 1 the structure-sensitive Level-2 model.

**Implementation:** the likelihood is vectorised with NumPy and SciPy for speed. :func:`check_against_reference` verifies it against the pure-Python functions of :mod:`kanm.judgment_inversion` before any estimate is made.

Every estimand, rule and seed is fixed in research/16, sections 4–8.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import betainc, expit, logit

from kanm import judgment_inversion as J
from kanm.structure_data import (OWN_EVIDENCE, STRONG_BLUE, WEAK_RED, download, load, load_triads)

SEED = 2026
BOOTSTRAPS = 1_000
FOLDS = 10
W_BOUNDS = (-1.0, 2.0)
SETTINGS = {
    "primary_uniform_tau10": (J.UNIFORM_EVIDENCE, 10.0),
    "uniform_tau3": (J.UNIFORM_EVIDENCE, 3.0),
    "uniform_tau30": (J.UNIFORM_EVIDENCE, 30.0),
    "agent_default_prior_tau10": ({"red": 0.25, "blue": 0.25, "none": 0.5}, 10.0),
}
_EDGES = np.arange(8) / 7.0


@dataclass
class Design:
    """Arrays for one experiment's focal participants (P participants, 10 trials, 2 other agents)."""
    judgments: np.ndarray    # (P, 10) index 0..6, -1 where missing
    own: np.ndarray          # (P, 10, 2) A's cumulative (red, blue)
    level1: np.ndarray       # (P, 10, 2, 2) other agents' counts at t - 1 (zero at t = 0)
    delta: np.ndarray        # (P, 10, 2, 2) Level-2 minus Level-1 counts at t - 1


def _agents(b_judgments, c_judgments, watcher: str | None, prior, tau):
    b1 = J.infer_evidence(b_judgments, evidence_prior=prior, tau=tau)
    c1 = J.infer_evidence(c_judgments, evidence_prior=prior, tau=tau)
    b2 = J.infer_evidence(b_judgments, parents=[c1], evidence_prior=prior, tau=tau) if watcher == "B" else b1
    c2 = J.infer_evidence(c_judgments, parents=[b1], evidence_prior=prior, tau=tau) if watcher == "C" else c1
    return (b1, c1), (b2, c2)


def build_design(records, prior, tau) -> Design:
    """``records``: (judgments with None for missing, B judgments, C judgments, watcher)."""
    P = len(records)
    judgments = np.full((P, 10), -1, dtype=int)
    own = np.zeros((P, 10, 2))
    level1 = np.zeros((P, 10, 2, 2))
    delta = np.zeros((P, 10, 2, 2))
    red = np.cumsum([e == "red" for e in OWN_EVIDENCE])
    blue = np.cumsum([e == "blue" for e in OWN_EVIDENCE])
    for i, (judged, b_seq, c_seq, watcher) in enumerate(records):
        judgments[i] = [-1 if j is None else j + 3 for j in judged]
        own[i, :, 0], own[i, :, 1] = red, blue
        l1, l2 = _agents(b_seq, c_seq, watcher, prior, tau)
        for t in range(1, 10):
            for k in range(2):
                level1[i, t, k] = l1[k][t - 1]
                delta[i, t, k] = np.subtract(l2[k][t - 1], l1[k][t - 1])
    return Design(judgments, own, level1, delta)


def response_probabilities(design: Design, w: float, tau: float) -> np.ndarray:
    """Softmax response probabilities over the 7 judgments, shape (P, 10, 7), before stickiness."""
    counts = design.own + np.maximum(0.0, design.level1 + w * design.delta).sum(axis=2)
    a, b = 1.0 + counts[..., 0], 1.0 + counts[..., 1]
    cdf = betainc(a[..., None], b[..., None], _EDGES)
    bins = np.diff(cdf, axis=-1)                                     # (P, 10, 7)
    k = np.arange(7)
    scores = -(bins[..., None, :] * (k[None, :] - k[:, None]) ** 2).sum(axis=-1)
    scores = tau * scores
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    return probs / probs.sum(axis=-1, keepdims=True)


def log_likelihood(design: Design, w: float, tau: float, sticky: float) -> np.ndarray:
    """Per-judgment log-likelihood, shape (P, 10); NaN where the judgment is missing."""
    probs = response_probabilities(design, w, tau)
    observed = design.judgments >= 0
    index = np.where(observed, design.judgments, 0)
    p = np.take_along_axis(probs, index[..., None], axis=-1)[..., 0]
    previous = np.concatenate([np.full((design.judgments.shape[0], 1), -1), design.judgments[:, :-1]], axis=1)
    use_sticky = (previous >= 0) & observed
    p = np.where(use_sticky, (1 - sticky) * p + sticky * (design.judgments == previous), p)
    # Floor at 1e-300 so the optimiser never meets log(0) for extreme tau; no effect at the optimum.
    return np.where(observed, np.log(np.maximum(p, 1e-300)), np.nan)


def check_against_reference(records, design: Design) -> float:
    """Largest difference from kanm.judgment_inversion at w = 0 and w = 1 (must be ~1e-12)."""
    worst = 0.0
    for w in (0.0, 1.0):
        vector = log_likelihood(design, w, 1.7, 0.3)
        for i, (judged, b_seq, c_seq, watcher) in enumerate(records):
            l1, l2 = _agents(b_seq, c_seq, watcher, J.UNIFORM_EVIDENCE, 10.0)
            others = l1 if w == 0 else l2
            bins = J.listener_bins(OWN_EVIDENCE, list(others))
            reference = J.judgment_log_likelihood(list(judged), bins, 1.7, 0.3)
            for t, value in enumerate(reference):
                if value is not None:
                    worst = max(worst, abs(value - vector[i, t]))
    return worst


def fit(design: Design, index=None, fixed_w: float | None = None, start=None) -> tuple[float, float, float, float]:
    """Maximum likelihood (w, tau, sticky, log-likelihood) over participants ``index``."""
    sub = design if index is None else Design(*(getattr(design, f)[index] for f in
                                                ("judgments", "own", "level1", "delta")))

    def unpack(theta):
        w = fixed_w if fixed_w is not None else theta[2]
        return w, float(np.exp(theta[0])), float(expit(theta[1]))

    def loss(theta):
        w, tau, sticky = unpack(theta)
        return -np.nansum(log_likelihood(sub, w, tau, sticky))

    bounds = [(np.log(0.01), np.log(100.0)), (-10.0, 10.0)] + ([] if fixed_w is not None else [W_BOUNDS])
    starts = [start] if start is not None else [np.array([0.0, 0.0] + ([] if fixed_w is not None else [w0]))
                                                for w0 in (0.0, 0.5, 1.0)]
    best = None
    for s in starts:
        result = minimize(loss, s, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    w, tau, sticky = unpack(best.x)
    return float(w), tau, sticky, float(-best.fun)


def bootstrap(design: Design, rng: np.random.Generator) -> np.ndarray:
    P = design.judgments.shape[0]
    point = fit(design)
    start = np.array([np.log(point[1]), logit(min(max(point[2], 1e-6), 1 - 1e-6)), point[0]])
    return np.array([fit(design, rng.integers(0, P, P), start=start)[0] for _ in range(BOOTSTRAPS)])


def cross_validation(design: Design, rng: np.random.Generator) -> dict:
    P = design.judgments.shape[0]
    folds = rng.permutation(np.arange(P) % FOLDS)
    held = {name: np.zeros(P) for name in ("free", "w0", "w1")}
    counts = (design.judgments >= 0).sum(axis=1)
    for k in range(FOLDS):
        train, test = np.where(folds != k)[0], np.where(folds == k)[0]
        test_design = Design(*(getattr(design, f)[test] for f in ("judgments", "own", "level1", "delta")))
        for name, fixed in (("free", None), ("w0", 0.0), ("w1", 1.0)):
            w, tau, sticky, _ = fit(design, train, fixed_w=fixed)
            held[name][test] = np.nansum(log_likelihood(test_design, w, tau, sticky), axis=1)
    out = {}
    index = rng.integers(0, P, size=(10_000, P))
    for other in ("w0", "w1"):
        diff = held["free"] - held[other]
        draws = diff[index].sum(axis=1) / counts[index].sum(axis=1)
        out[f"free_minus_{other}"] = {"estimate": float(diff.sum() / counts.sum()),
                                      "interval": [float(q) for q in np.quantile(draws, [0.025, 0.975])]}
    return out


def decision(interval) -> str:
    """Section 6 categories for an interval of w."""
    low, high = interval
    if low > 1:
        return "greater_than_full"
    if low > 0:
        return "consistent_with_full" if high >= 1 else "partial"
    if high < 0:
        return "below_zero"
    return "uninformative" if high >= 1 else "not_detected_and_below_full"


def records_for(experiment: int, data_dir: str):
    if experiment == 1:
        return [([None if f else j for j, f in zip(p.judgments, p.autofilled)], STRONG_BLUE, WEAK_RED,
                 "C" if p.condition == "c_sees_b" else "B")
                for p in load(data_dir) if p.condition != "none"]
    return [([None if f else j for j, f in zip(t.judgments, t.autofilled)], t.b_judgments, t.c_judgments, "C")
            for t in load_triads(data_dir, experiment) if t.condition == "c_sees_b"]


def identifiability_control(data_dir: str) -> float:
    """W5: in independent rooms Level-1 and Level-2 must coincide exactly."""
    worst = 0.0
    for experiment in (2, 3):
        for t in load_triads(data_dir, experiment):
            if t.condition != "independent":
                continue
            l1, l2 = _agents(t.b_judgments, t.c_judgments, None, J.UNIFORM_EVIDENCE, 10.0)
            worst = max(worst, max(abs(x - y) for a, b in zip(l1, l2) for p, q in zip(a, b) for x, y in zip(p, q)))
    return worst


def run(data_dir: str) -> dict:
    rng = np.random.default_rng(SEED)
    report = {"study": "Structure-correction weight, Fränken et al. (2024) Experiments 1-3",
              "plan": "research/16_structure_correction_weight_plan.md",
              "settings": {"bootstraps": BOOTSTRAPS, "folds": FOLDS, "seed": SEED, "w_bounds": W_BOUNDS},
              "experiments": {}}
    report["W5_independent_rooms_max_level2_minus_level1"] = identifiability_control(data_dir)
    for experiment in (1, 2, 3):
        records = records_for(experiment, data_dir)
        entry = {"focal_participants": len(records),
                 "judgments": int(sum(j is not None for r in records for j in r[0])), "settings": {}}
        for name, (prior, tau) in SETTINGS.items():
            if name != "primary_uniform_tau10" and experiment == 3:
                continue  # W3 covers Experiments 1 and 2
            design = build_design(records, prior, tau)
            if name == "primary_uniform_tau10":
                entry["reference_check_max_abs_diff"] = check_against_reference(records, design)
            w, fitted_tau, sticky, ll = fit(design)
            draws = bootstrap(design, rng)
            interval = [float(q) for q in np.quantile(draws, [0.025, 0.975])]
            result = {"w": w, "interval": interval, "decision": decision(interval),
                      "tau": fitted_tau, "sticky": sticky, "log_likelihood": ll,
                      "bootstrap_draws_at_bound": int(np.sum((draws <= W_BOUNDS[0] + 1e-6) |
                                                             (draws >= W_BOUNDS[1] - 1e-6)))}
            if name == "primary_uniform_tau10":
                result["W4_cross_validation"] = cross_validation(design, rng)
            entry["settings"][name] = result
        report["experiments"][str(experiment)] = entry
    report["W3_E3_from_research15"] = e3_under_settings()
    report["post_hoc_profile_likelihood"] = profiles(data_dir)
    return report


def profiles(data_dir: str) -> dict:
    """Post hoc diagnostic (research/16, section 9.3): log-likelihood relative to its maximum over a grid of w.

    It uses no random numbers, so adding it changes no other result.
    """
    out = {}
    grid = [round(x, 3) for x in np.linspace(-1.0, 2.0, 61)]
    for experiment in (1, 2, 3):
        design = build_design(records_for(experiment, data_dir), J.UNIFORM_EVIDENCE, 10.0)
        best = fit(design)[3]
        out[str(experiment)] = {"w": grid, "relative_log_likelihood": [fit(design, fixed_w=w)[3] - best
                                                                        for w in grid]}
    return out


def e3_under_settings() -> dict:
    """research/15's normative contrast E2 (Level-2) under each inversion setting."""
    out = {}
    for name, (prior, tau) in SETTINGS.items():
        b = J.infer_evidence(STRONG_BLUE, evidence_prior=prior, tau=tau)
        c = J.infer_evidence(WEAK_RED, evidence_prior=prior, tau=tau)
        cb = J.listener_bins(OWN_EVIDENCE, [b, J.infer_evidence(WEAK_RED, parents=[b], evidence_prior=prior, tau=tau)])
        bc = J.listener_bins(OWN_EVIDENCE, [J.infer_evidence(STRONG_BLUE, parents=[c], evidence_prior=prior, tau=tau), c])
        window = slice(5, 10)
        contrast = (np.mean([J.expected_judgment(x) for x in cb[window]])
                    - np.mean([J.expected_judgment(x) for x in bc[window]]))
        out[name] = {"normative_contrast": float(contrast)}
    return out


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
