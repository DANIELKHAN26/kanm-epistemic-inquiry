"""Conditional Rasch lack-of-fit test for crowd-worker errors (research/18).

Usage::

    python -m examples.study_local_dependence DATA_DIR [--download] [--validate]

Null model, separately in each gold class c and each batch b of items labelled
by the same workers: worker w errs on item i with probability
logistic(theta[w, c, b] + delta[i]), independently. Under this Rasch model the
worker and item error totals are sufficient, so given both margins every error
matrix is equally likely, whatever the parameters. Error matrices are drawn
from that uniform distribution by curveball trades (Strona et al. 2014;
Carstens 2015), and the Besag-Clifford (1989) parallel scheme gives a
finite-sample-valid conditional Monte Carlo test without a mixing assumption.
Inclusive tie handling makes p-values conservative when statistics tie. No
parameter is estimated.

The test asks whether shared errors are concentrated in particular worker
pairs beyond that model. Needs NumPy (``pip install .[analysis]``). Worker
identifiers are never written; the output holds aggregates only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from random import Random

import numpy as np

from kanm.crowd_data import DATASETS, download, load

MIN_SHARED = 10          # a pair is tested if it shares this many informative items in a class
TRADES_PER_WORKER = 20   # curveball trades per batch and chain segment, times the batch's workers
SAMPLES = 999
ALPHA = 0.05
SEED = 20261118
_CHUNKS = 32             # parallel work units for the null draws


# --------------------------------------------------------------------------- design
@dataclass(frozen=True)
class ClassDesign:
    """Every gold item of one class: who labelled it, and whether each label is wrong."""

    workers: np.ndarray   # (items, n) global worker indices, sorted within each row
    errors: np.ndarray    # (items, n) 0/1


def designs(data) -> list[ClassDesign]:
    """Per-class designs of a :class:`kanm.reliability.CrowdData` (gold items only)."""
    names = sorted({w for _, w, _ in data.labels})
    index = {w: k for k, w in enumerate(names)}
    rows: dict[str, list[tuple[int, int]]] = {}
    for item, worker, label in data.labels:
        if item in data.gold:
            rows.setdefault(item, []).append((index[worker], int(label != data.gold[item])))
    out = []
    for cls in (0, 1):
        items = sorted(i for i in rows if data.gold[i] == cls)
        sizes = {len(rows[i]) for i in items}
        if len(sizes) != 1:
            raise ValueError(f"{data.name}: items of class {cls} have unequal label counts {sizes}")
        ordered = [sorted(rows[i]) for i in items]
        out.append(ClassDesign(np.array([[w for w, _ in r] for r in ordered]),
                               np.array([[x for _, x in r] for r in ordered], dtype=np.int8)))
    return out


@dataclass
class Batch:
    """Informative items of one class labelled by one set of workers, as error sets."""

    items: int                           # informative items in the batch
    rows: list[set[int]]                 # rows[a]: items on which local worker a erred
    pairs: list[tuple[int, int, int]]    # (local a, local b, tested-pair index)


def batches(classes: list[ClassDesign], errors: list[np.ndarray] | None = None
            ) -> tuple[list[Batch], int]:
    """Split each class into worker-set batches and index the tested pairs.

    Items with no errors or all errors are dropped: both margins fix them.
    """
    errors = [c.errors for c in classes] if errors is None else errors
    out: list[Batch] = []
    tested = 0
    for design, x in zip(classes, errors):
        totals = x.sum(axis=1)
        keep = (totals > 0) & (totals < x.shape[1])
        workers, x = design.workers[keep], x[keep]
        groups: dict[tuple[int, ...], list[int]] = {}
        for item, row in enumerate(workers):
            groups.setdefault(tuple(int(w) for w in row), []).append(item)
        shared: dict[tuple[int, int], int] = {}
        for members, items in groups.items():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    key = (members[a], members[b])
                    shared[key] = shared.get(key, 0) + len(items)
        index = {}
        for key in sorted(k for k, v in shared.items() if v >= MIN_SHARED):
            index[key] = tested
            tested += 1
        for members, items in sorted(groups.items()):
            rows = [{k for k, item in enumerate(items) if x[item, col]} for col in range(len(members))]
            pairs = [(a, b, index[(members[a], members[b])])
                     for a in range(len(members)) for b in range(a + 1, len(members))
                     if (members[a], members[b]) in index]
            out.append(Batch(len(items), rows, pairs))
    return out, tested


def co_errors(state: list[Batch], tested: int) -> np.ndarray:
    """Shared errors of every tested pair."""
    out = np.zeros(tested)
    for batch in state:
        rows = batch.rows
        for a, b, k in batch.pairs:
            out[k] += len(rows[a] & rows[b])
    return out


def curveball(state: list[Batch], rng: Random) -> None:
    """Advance every batch by TRADES_PER_WORKER x workers curveball trades, in place.

    A trade pools the items on which exactly one of two workers erred and deals
    them back at random in the same numbers, so worker and item totals are kept.
    """
    for batch in state:
        rows = batch.rows
        n = len(rows)
        if n < 2:
            continue
        for _ in range(TRADES_PER_WORKER * n):
            r, s = rng.sample(range(n), 2)
            only_r, only_s = rows[r] - rows[s], rows[s] - rows[r]
            if not only_r or not only_s:
                continue
            pool = sorted(only_r | only_s)
            rng.shuffle(pool)
            common = rows[r] & rows[s]
            rows[r] = common | set(pool[: len(only_r)])
            rows[s] = common | set(pool[len(only_r):])


def _copy(state: list[Batch]) -> list[Batch]:
    return [Batch(b.items, [set(r) for r in b.rows], b.pairs) for b in state]


def _forward(job: tuple) -> np.ndarray:
    hub, tested, seeds = job
    out = np.empty((len(seeds), tested))
    for k, seed in enumerate(seeds):
        state = _copy(hub)
        curveball(state, Random(seed))
        out[k] = co_errors(state, tested)
    return out


def null_draws(state: list[Batch], tested: int, samples: int, seed: np.random.SeedSequence,
               pool: ProcessPoolExecutor | None = None) -> np.ndarray:
    """Besag-Clifford parallel scheme: one backward segment to a hub, then independent
    forward segments. The curveball chain is symmetric, so backward equals forward."""
    seeds = [int(s.generate_state(1)[0]) for s in seed.spawn(samples + 1)]
    hub = _copy(state)
    curveball(hub, Random(seeds[0]))
    chunks = [seeds[1 + k::_CHUNKS] for k in range(_CHUNKS)]
    jobs = [(hub, tested, c) for c in chunks if c]
    parts = list(pool.map(_forward, jobs)) if pool is not None else [_forward(j) for j in jobs]
    out = np.empty((samples, tested))
    for k, part in enumerate(parts):
        out[k::_CHUNKS] = part
    return out


def _upper_rank_pvalue(statistics: np.ndarray) -> float:
    """Inclusive upper-tail rank p-value; ties are retained conservatively."""
    values = np.asarray(statistics, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("statistics must be a non-empty one-dimensional array")
    return float(np.count_nonzero(values >= values[0]) / values.size)


# --------------------------------------------------------------------------- test
def test(state: list[Batch], tested: int, seed: np.random.SeedSequence, samples: int = SAMPLES,
         pool: ProcessPoolExecutor | None = None) -> dict:
    """Conditional global dispersion test and pair-level Monte Carlo p-values.

    Pairs are standardised by the mean and SD over the observed matrix and all null
    draws together: a symmetric function of the exchangeable set. Inclusive ranks
    give valid p-values and are conservative when the statistic has ties.
    """
    if tested < 1:
        raise ValueError("at least one pair must vary under the conditional null")
    if samples < 1:
        raise ValueError("samples must be at least 1")
    observed = co_errors(state, tested)
    draws = np.vstack([observed, null_draws(state, tested, samples, seed, pool)])
    mean, sd = draws.mean(axis=0), draws.std(axis=0)
    varies = sd > 0
    if not varies.any():
        raise ValueError("no tested pair varies under the conditional null")
    z = (draws[:, varies] - mean[varies]) / sd[varies]
    d = (z ** 2).sum(axis=1)
    pair_p = ((draws[1:, varies] >= observed[varies]).sum(axis=0) + 1.0) / (samples + 1.0)
    discovered = benjamini_yekutieli(pair_p, ALPHA)
    null_sorted = np.sort(z[1:], axis=1)
    reference_mean = float(d.mean())
    return {
        "statistic_D": float(d[0]),
        "null_mean_D": float(d[1:].mean()),
        "reference_mean_D": reference_mean,
        "dispersion_ratio_R": float(d[0] / reference_mean),
        "null_R_2_5_97_5": [float(np.quantile(d[1:] / reference_mean, 0.025)),
                            float(np.quantile(d[1:] / reference_mean, 0.975))],
        "p_value": _upper_rank_pvalue(d),
        "samples": samples,
        "tested_pairs": int(tested),
        "pairs_varying_under_null": int(varies.sum()),
        "pairs_with_excess_shared_errors": int((z[0] > 0).sum()),
        "by_discoveries_positive": int(discovered.sum()),
        "share_of_D_in_by_discoveries": float((z[0][discovered] ** 2).sum() / d[0]) if d[0] > 0 else 0.0,
        "z_sorted": [round(float(v), 4) for v in np.sort(z[0])],
        "null_z_sorted_mean": [round(float(v), 4) for v in null_sorted.mean(axis=0)],
        "null_z_sorted_2_5": [round(float(v), 4) for v in np.quantile(null_sorted, 0.025, axis=0)],
        "null_z_sorted_97_5": [round(float(v), 4) for v in np.quantile(null_sorted, 0.975, axis=0)],
    }


def benjamini_yekutieli(p_values: np.ndarray, level: float) -> np.ndarray:
    """Control false discovery rate under arbitrary dependence among tests."""
    order = np.argsort(p_values, kind="stable")
    m = len(p_values)
    if m == 0:
        return np.zeros(0, dtype=bool)
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    passed = p_values[order] <= level * np.arange(1, m + 1) / (m * harmonic)
    out = np.zeros(m, dtype=bool)
    if passed.any():
        out[order[: np.nonzero(passed)[0].max() + 1]] = True
    return out


def holm(p_values: dict[str, float]) -> dict[str, float]:
    names = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for k, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - k) * p_values[name]))
        adjusted[name] = running
    return adjusted


# --------------------------------------------------------------------------- validation
def simulate_rasch(design: ClassDesign, rng: np.random.Generator) -> np.ndarray:
    """V2 null data on a real design: theta ~ N(-1.5, 1), delta ~ N(0, 1.5)."""
    theta = rng.normal(-1.5, 1.0, design.workers.max() + 1)
    delta = rng.normal(0.0, 1.5, design.workers.shape[0])
    logits = theta[design.workers] + delta[:, None]
    return (rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits))).astype(np.int8)


def simulate_copying(classes: list[ClassDesign], rng: np.random.Generator,
                     fraction: float = 0.05, rate: float = 0.3) -> list[np.ndarray]:
    """V3 alternative: in 5 % of pairs sharing >= 20 items, the second copies the first."""
    errors = [simulate_rasch(d, rng) for d in classes]
    shared: dict[tuple[int, int], int] = {}
    for d in classes:
        for row in d.workers:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    key = (int(row[i]), int(row[j]))
                    shared[key] = shared.get(key, 0) + 1
    eligible = sorted(k for k, v in shared.items() if v >= 20)
    picks = rng.choice(len(eligible), max(1, round(fraction * len(eligible))), replace=False)
    for d, x in zip(classes, errors):
        position = {(item, int(w)): col for item, row in enumerate(d.workers) for col, w in enumerate(row)}
        for source, copier in (eligible[k] for k in picks):
            for item in range(len(d.workers)):
                if (item, source) in position and (item, copier) in position and rng.random() < rate:
                    x[item, position[(item, copier)]] = x[item, position[(item, source)]]
    return errors


def _replicate(job: tuple) -> float:
    classes, samples, seed, copying = job
    data_seed, test_seed = seed.spawn(2)
    rng = np.random.default_rng(data_seed)
    errors = simulate_copying(classes, rng) if copying else [simulate_rasch(d, rng) for d in classes]
    state, tested = batches(classes, errors)
    return test(state, tested, test_seed, samples)["p_value"]


def rejection_rate(classes: list[ClassDesign], replicates: int, samples: int,
                   seed: np.random.SeedSequence, pool: ProcessPoolExecutor, *, copying: bool) -> dict:
    jobs = [(classes, samples, child, copying) for child in seed.spawn(replicates)]
    p_values = np.array(list(pool.map(_replicate, jobs)))
    return {"replicates": replicates, "samples": samples,
            "rejection_rate_at_0_05": float((p_values <= ALPHA).mean()),
            "p_value_deciles": [round(float(q), 3) for q in np.quantile(p_values, np.linspace(0.1, 0.9, 9))]}


# --------------------------------------------------------------------------- driver
def validate(data_dir: str, seed: int = SEED) -> dict:
    """V2 (size under the null) and V3 (power against copying) on the real designs."""
    root = np.random.SeedSequence([seed, 2])
    out = {}
    with ProcessPoolExecutor() as pool:
        for name, child in zip(DATASETS, root.spawn(len(DATASETS))):
            classes = designs(load(name, data_dir))
            size_seed, power_seed = child.spawn(2)
            start = time.time()
            out[name] = {"size": rejection_rate(classes, 100, 199, size_seed, pool, copying=False)}
            if name in ("welinder2010_bluebirds", "zheng2017_sentiment"):
                out[name]["power_copying"] = rejection_rate(classes, 100, 199, power_seed, pool,
                                                            copying=True)
            print(f"{name}: {out[name]} ({time.time() - start:.0f} s)", file=sys.stderr, flush=True)
    return {"design": {"min_shared_items": MIN_SHARED, "trades_per_worker": TRADES_PER_WORKER,
                       "seed": seed, "null": "theta ~ N(-1.5, 1), delta ~ N(0, 1.5)",
                       "copying": "5% of pairs sharing >= 20 items; copy probability 0.3",
                       "plan": "research/18, sections 5 and 8"},
            "datasets": out}


def run(data_dir: str, seed: int = SEED) -> dict:
    root = np.random.SeedSequence([seed, 1])
    results = {}
    with ProcessPoolExecutor() as pool:
        for name, child in zip(DATASETS, root.spawn(len(DATASETS))):
            start = time.time()
            state, tested = batches(designs(load(name, data_dir)))
            result = test(state, tested, child, SAMPLES, pool)
            result["informative_items"] = sum(b.items for b in state)
            result["batches"] = len(state)
            results[name] = result
            print(f"{name}: R = {result['dispersion_ratio_R']:.3f}, p = {result['p_value']:.4f} "
                  f"({time.time() - start:.0f} s)", file=sys.stderr, flush=True)
    adjusted = holm({k: v["p_value"] for k, v in results.items()})
    for name, value in adjusted.items():
        results[name]["p_holm"] = value
        results[name]["reject_conditional_rasch"] = value <= ALPHA
    return {"design": {"min_shared_items": MIN_SHARED, "trades_per_worker": TRADES_PER_WORKER,
                       "samples": SAMPLES, "alpha": ALPHA, "seed": seed,
                       "plan": "research/18, sections 1-8"},
            "datasets": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--validate", action="store_true", help="run V2 (size) and V3 (power) only")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.download:
        download(args.data_dir)
    result = validate(args.data_dir) if args.validate else run(args.data_dir)
    text = json.dumps(result, indent=1, allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
