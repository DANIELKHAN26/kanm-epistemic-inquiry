"""Learning source reliability from testimony across many binary questions.

Laputa's trust learning is an instance of a well-studied problem: infer the
answers to binary questions, and the reliability of each source, from reports
whose correctness is never observed directly. Crowdsourcing truth inference
(Dawid & Skene 1979; Raykar et al. 2010; Zheng et al. 2017) studies exactly
this, with public data sets that include gold answers. This module lets the
KANM update rules be tested on such data.

Each item ``i`` has an unknown binary answer ``z_i``. Each source ("worker")
``w`` has a symmetric reliability ``r_w`` (probability of reporting the right
answer), which is Laputa's source-symmetry assumption ("one-coin" model).
Items are processed one at a time (online), as an inquirer would meet new
questions; the class prior is fixed at 1/2 for every method.

Methods
-------
``majority_vote``
    Decision by vote; the reported probability is the share of positive
    votes (a heuristic score, not a posterior).
``laputa_simultaneous``
    The Laputa rule as KANM's ``simultaneous`` schedule applies it: credence
    from expected trusts; each trust updated with the credence held before
    the item (1/2). At credence 1/2 a report carries no information about
    reliability, so trust never changes; with equal trust priors the decisions
    equal majority vote. This is a property of the rule, not a defect of the
    implementation.
``laputa_sequential``
    The Laputa rule with reports applied one at a time and each trust update
    using the running credence (KANM's ``inquiry_then_messages`` schedule).
``exact_adf``
    Within an item, the exact joint posterior over ``z_i`` and the reliabilities
    of the item's workers (:class:`kanm.bayes.ExactJointState`). Between
    items, each worker's trust is replaced by its exact marginal posterior,
    i.e. assumed-density filtering (Opper 1998; Minka 2001). Exact for a
    single item; an approximation across items because correlations between
    workers induced by shared items are dropped.
``dawid_skene_one_coin`` / ``dawid_skene_two_coin``
    Batch (offline) maximum a posteriori EM for the one-coin and two-coin
    Dawid-Skene models, with Beta(alpha, beta) priors on accuracies (two-coin:
    on sensitivity and specificity). These see all data at once and are
    standard reference methods, not online learners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum, isfinite, log, sqrt
import random
from typing import Mapping, Sequence

from .bayes import ExactJointState, TrustDistribution, expit, log_likelihood_ratio

ONLINE_METHODS = ("laputa_simultaneous", "laputa_sequential", "exact_adf")
BATCH_METHODS = (
    "majority_vote", "dawid_skene_one_coin", "dawid_skene_two_coin",
    "dawid_skene_one_coin_estimated_prior", "dawid_skene_two_coin_estimated_prior",
)
METHODS = BATCH_METHODS[:1] + ONLINE_METHODS + BATCH_METHODS[1:]


@dataclass(frozen=True)
class CrowdData:
    """Binary reports ``(item, worker, label)`` with optional gold answers."""

    name: str
    labels: tuple[tuple[str, str, int], ...]
    gold: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        try:
            labels = tuple((str(i), str(w), l) for i, w, l in self.labels)
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must contain (item, worker, label) triples") from exc
        if not labels:
            raise ValueError("at least one label is required")
        if any(not i.strip() or not w.strip() for i, w, _ in labels):
            raise ValueError("item and worker identifiers must be non-empty")
        if any(type(l) is not int or l not in (0, 1) for _, _, l in labels):
            raise ValueError("labels must be the integers 0 or 1")
        pairs = [(i, w) for i, w, _ in labels]
        if len(set(pairs)) != len(pairs):
            raise ValueError("each item-worker pair may have at most one label")
        try:
            raw_gold = dict(self.gold)
            gold_pairs = [(str(k), v) for k, v in raw_gold.items()]
            gold = dict(gold_pairs)
        except (TypeError, ValueError) as exc:
            raise ValueError("gold must be a mapping from item identifiers to labels") from exc
        if len(gold) != len(gold_pairs):
            raise ValueError("gold item identifiers must remain unique when normalized to strings")
        if any(not k.strip() for k in gold):
            raise ValueError("gold item identifiers must be non-empty")
        if any(type(v) is not int or v not in (0, 1) for v in gold.values()):
            raise ValueError("gold answers must be the integers 0 or 1")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "gold", gold)

    @property
    def items(self) -> list[str]:
        return sorted({i for i, _, _ in self.labels})

    @property
    def workers(self) -> list[str]:
        return sorted({w for _, w, _ in self.labels})

    def by_item(self) -> dict[str, list[tuple[str, int]]]:
        grouped: dict[str, list[tuple[str, int]]] = {}
        for item, worker, label in self.labels:
            grouped.setdefault(item, []).append((worker, label))
        for reports in grouped.values():
            reports.sort()
        return grouped

    def summary(self) -> dict[str, float | int]:
        items = self.items
        return {
            "items": len(items),
            "workers": len(self.workers),
            "labels": len(self.labels),
            "items_with_gold": sum(1 for i in items if i in self.gold),
            "gold_positive_share": (
                sum(self.gold[i] for i in items if i in self.gold)
                / max(1, sum(1 for i in items if i in self.gold))
            ),
        }


@dataclass
class Estimate:
    """Posterior probability that each item's answer is 1, and source reliabilities."""

    method: str
    posteriors: dict[str, float]
    reliability: dict[str, float]
    iterations: int | None = None
    converged: bool | None = None
    class_prior: float | None = None


# ---------------------------------------------------------------- methods
def majority_vote(data: CrowdData) -> Estimate:
    posteriors = {}
    for item, reports in data.by_item().items():
        posteriors[item] = sum(label for _, label in reports) / len(reports)
    return Estimate("majority_vote", posteriors, {})


def _validated_order(data: CrowdData, order: Sequence[str] | None) -> list[str]:
    items = data.items
    if order is None:
        return items
    order = list(order)
    if sorted(order) != items:
        raise ValueError("order must be a permutation of the data set's items")
    return order


def laputa_online(
    data: CrowdData,
    prior: TrustDistribution,
    *,
    schedule: str = "simultaneous",
    order: Sequence[str] | None = None,
    report_order: Mapping[str, Sequence[str]] | None = None,
) -> Estimate:
    """Laputa's rule applied item by item (see module docstring)."""
    if schedule not in ("simultaneous", "sequential"):
        raise ValueError("schedule must be 'simultaneous' or 'sequential'")
    grouped = data.by_item()
    if report_order is not None:
        for item, workers in report_order.items():
            if item not in grouped or sorted(workers) != sorted(w for w, _ in grouped[item]):
                raise ValueError(f"report_order[{item!r}] must be a permutation of that item's workers")
    trust: dict[str, TrustDistribution] = {}
    posteriors = {}
    for item in _validated_order(data, order):
        # Log-odds contributions are summed exactly (math.fsum, class prior 1/2),
        # so that balanced reports give exactly 1/2 rather than a rounding residue
        # that would break exact ties at random.
        terms: list[float] = []
        reports = grouped[item]
        if report_order is not None and item in report_order:
            by_worker = {worker: label for worker, label in reports}
            reports = [(worker, by_worker[worker]) for worker in report_order[item]]
        for worker, label in reports:
            current = trust.get(worker, prior)
            basis = 0.5 if schedule == "simultaneous" else expit(fsum(terms))
            terms.append(log_likelihood_ratio(current, label == 1))
            trust[worker] = current.observe(basis, label == 1)
        posteriors[item] = expit(fsum(terms))
    return Estimate(
        f"laputa_{schedule}",
        posteriors,
        {w: t.expected_reliability for w, t in trust.items()},
    )


def exact_adf(
    data: CrowdData, prior: TrustDistribution, *, order: Sequence[str] | None = None,
    report_order: Mapping[str, Sequence[str]] | None = None,
) -> Estimate:
    """Exact within items, assumed-density filtering between items."""
    grouped = data.by_item()
    if report_order is not None:
        for item, workers in report_order.items():
            if item not in grouped or sorted(workers) != sorted(w for w, _ in grouped[item]):
                raise ValueError(f"report_order[{item!r}] must be a permutation of that item's workers")
    trust: dict[str, TrustDistribution] = {}
    posteriors = {}
    for item in _validated_order(data, order):
        state = ExactJointState(0.5)
        for worker in sorted({w for w, _ in grouped[item]}):
            state.add_source(worker, trust.get(worker, prior))
        reports = grouped[item]
        if report_order is not None and item in report_order:
            by_worker = {worker: label for worker, label in reports}
            reports = [(worker, by_worker[worker]) for worker in report_order[item]]
        for worker, label in reports:
            state.observe(worker, label == 1)
        posteriors[item] = state.belief
        for worker in state.sources:
            trust[worker] = state.trust(worker)
    return Estimate("exact_adf", posteriors, {w: t.expected_reliability for w, t in trust.items()})


def _clamp(p: float) -> float:
    return min(max(p, 1e-12), 1.0 - 1e-12)


def _map_binomial_rate(successes: float, trials: float, alpha: float, beta: float) -> float:
    """Beta-binomial MAP rate, with the prior mean for an unobserved flat prior."""
    denominator = trials + alpha + beta - 2.0
    if denominator <= 0.0:
        # Beta(1, 1) remains uniform when this class has no effective counts;
        # use its mean as a neutral representative rather than divide by zero.
        return alpha / (alpha + beta)
    return _clamp((successes + alpha - 1.0) / denominator)


def dawid_skene(
    data: CrowdData,
    *,
    model: str = "one_coin",
    alpha: float = 2.0,
    beta: float = 1.0,
    class_prior: float = 0.5,
    estimate_class_prior: bool = False,
    class_prior_alpha: float = 1.0,
    class_prior_beta: float = 1.0,
    tolerance: float = 1e-9,
    max_iterations: int = 1000,
) -> Estimate:
    """MAP EM for one-coin or two-coin Dawid-Skene models.

    Initialised at the majority-vote shares. The prior Beta(alpha, beta) acts
    as ``alpha - 1`` correct and ``beta - 1`` incorrect pseudo-reports per
    worker (per class for the two-coin model). The reported reliability of a
    two-coin worker is the balanced accuracy ``(sensitivity + specificity)/2``.
    By default the class prior is fixed at 1/2; ``estimate_class_prior=True``
    estimates its prevalence with a separate Beta prior. This option matters
    when positive labels are rare and is reported as a sensitivity analysis.
    """
    if model not in ("one_coin", "two_coin"):
        raise ValueError("model must be 'one_coin' or 'two_coin'")
    for value, label in ((alpha, "alpha"), (beta, "beta"),
                         (class_prior_alpha, "class_prior_alpha"),
                         (class_prior_beta, "class_prior_beta")):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or value < 1.0):
            raise ValueError(f"{label} must be finite and >= 1")
    if isinstance(class_prior, bool) or not isinstance(class_prior, (int, float)) \
            or not isfinite(class_prior) or not 0.0 < class_prior < 1.0:
        raise ValueError("class_prior must be finite and strictly between 0 and 1")
    if not isinstance(estimate_class_prior, bool):
        raise ValueError("estimate_class_prior must be a boolean")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) \
            or not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and > 0")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    posteriors = majority_vote(data).posteriors
    prior_mean = sum(posteriors.values()) / len(posteriors)
    current_class_prior = (
        min(max(prior_mean, 1e-12), 1.0 - 1e-12) if estimate_class_prior else float(class_prior)
    )
    labels = data.labels
    iterations = 0
    change = float("inf")
    for iterations in range(1, max_iterations + 1):
        # M-step.
        if model == "one_coin":
            correct: dict[str, float] = {}
            count: dict[str, int] = {}
            for item, worker, label in labels:
                p = posteriors[item]
                correct[worker] = correct.get(worker, 0.0) + (p if label == 1 else 1.0 - p)
                count[worker] = count.get(worker, 0) + 1
            accuracy = {
                w: _map_binomial_rate(correct[w], count[w], alpha, beta)
                for w in count
            }
        else:
            stats: dict[str, list[float]] = {}
            for item, worker, label in labels:
                p = posteriors[item]
                s = stats.setdefault(worker, [0.0, 0.0, 0.0, 0.0])
                s[0] += p * (label == 1)          # true positives (soft)
                s[1] += p                         # positives (soft)
                s[2] += (1.0 - p) * (label == 0)  # true negatives (soft)
                s[3] += 1.0 - p                   # negatives (soft)
            sensitivity = {w: _map_binomial_rate(s[0], s[1], alpha, beta)
                           for w, s in stats.items()}
            specificity = {w: _map_binomial_rate(s[2], s[3], alpha, beta)
                           for w, s in stats.items()}
        if estimate_class_prior:
            prior_denominator = len(posteriors) + class_prior_alpha + class_prior_beta - 2.0
            current_class_prior = min(max(
                (sum(posteriors.values()) + class_prior_alpha - 1.0) / prior_denominator,
                1e-12), 1.0 - 1e-12)
        # E-step.
        prior_log_odds = log(current_class_prior) - log(1.0 - current_class_prior)
        log_odds = {item: prior_log_odds for item in posteriors}
        for item, worker, label in labels:
            if model == "one_coin":
                a = accuracy[worker]
                term = log(a) - log(1.0 - a)
                log_odds[item] += term if label == 1 else -term
            else:
                se, sp = sensitivity[worker], specificity[worker]
                log_odds[item] += (log(se) - log(1.0 - sp)) if label == 1 else (log(1.0 - se) - log(sp))
        new = {item: expit(lo) for item, lo in log_odds.items()}
        change = max(abs(new[i] - posteriors[i]) for i in new)
        posteriors = new
        if change < tolerance:
            break
    if model == "one_coin":
        reliability = accuracy
    else:
        reliability = {w: (sensitivity[w] + specificity[w]) / 2.0 for w in sensitivity}
    return Estimate(
        f"dawid_skene_{model}", posteriors, reliability, iterations,
        converged=change < tolerance, class_prior=current_class_prior
    )


# ---------------------------------------------------------------- metrics
def score(estimate: Estimate, gold: Mapping[str, int], *, bins: int = 10) -> dict[str, float]:
    """Accuracy, Brier score, log loss and expected calibration error on gold items.

    Accuracy counts a posterior of exactly 1/2 as half correct (the expected
    accuracy of a random tie-break). Log loss clips probabilities to
    [1e-12, 1 - 1e-12]; ``clipped`` reports how many items needed it.
    """
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if any(type(z) is not int or z not in (0, 1) for z in gold.values()):
        raise ValueError("gold answers must be the integers 0 or 1")
    if any(not isinstance(p, (int, float)) or isinstance(p, bool)
           or not 0.0 <= p <= 1.0 or not isfinite(p)
           for p in estimate.posteriors.values()):
        raise ValueError("posterior probabilities must be finite values in [0, 1]")
    items = [i for i in estimate.posteriors if i in gold]
    if not items:
        raise ValueError("no item has a gold answer")
    accuracy = brier = log_loss = 0.0
    clipped = 0
    calibration: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(bins)]
    positive_calibration: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(bins)]
    for item in items:
        p, z = estimate.posteriors[item], gold[item]
        accuracy += 0.5 if p == 0.5 else float((p > 0.5) == (z == 1))
        brier += (p - z) ** 2
        p_true = p if z == 1 else 1.0 - p
        if p_true < 1e-12:
            clipped += 1
        log_loss -= log(max(p_true, 1e-12))
        confidence = max(p, 1.0 - p)
        predicted = 1 if p >= 0.5 else 0
        k = min(int((confidence - 0.5) * 2 * bins), bins - 1)
        calibration[k][0] += 1
        calibration[k][1] += confidence
        calibration[k][2] += float(predicted == z)
        positive_bin = min(int(p * bins), bins - 1)
        positive_calibration[positive_bin][0] += 1
        positive_calibration[positive_bin][1] += p
        positive_calibration[positive_bin][2] += z
    n = len(items)
    ece = sum(abs(c[1] - c[2]) for c in calibration if c[0]) / n
    positive_ece = sum(abs(c[1] - c[2]) for c in positive_calibration if c[0]) / n
    return {
        "n_items": n,
        "accuracy": accuracy / n,
        "brier": brier / n,
        "log_loss": log_loss / n,
        "log_loss_clipped_items": clipped,
        "ece": ece,
        "positive_probability_ece": positive_ece,
        "calibration_bins": bins,
    }


def reliability_agreement(
    estimate: Estimate, data: CrowdData, *, min_gold_reports: int = 20
) -> dict[str, float | int | None]:
    """Compare estimated reliabilities with empirical accuracy against gold.

    Only workers with at least ``min_gold_reports`` gold-scored reports are
    used. Returns the Pearson correlation and mean absolute error.
    """
    if (isinstance(min_gold_reports, bool) or not isinstance(min_gold_reports, int)
            or min_gold_reports < 1):
        raise ValueError("min_gold_reports must be a positive integer")
    hits: dict[str, list[int]] = {}
    for item, worker, label in data.labels:
        if item in data.gold:
            hits.setdefault(worker, []).append(int(label == data.gold[item]))
    pairs = [
        (estimate.reliability[w], sum(h) / len(h))
        for w, h in hits.items()
        if len(h) >= min_gold_reports and w in estimate.reliability
    ]
    if len(pairs) < 3:
        return {"workers": len(pairs), "pearson": None, "mae": None}
    xs, ys = zip(*pairs)
    if min(xs) == max(xs) or min(ys) == max(ys):
        # No variance (e.g. a method that never updates reliability): the
        # correlation is undefined, not zero.
        return {"workers": len(pairs), "pearson": None,
                "mae": fsum(abs(x - y) for x, y in pairs) / len(pairs)}
    mx, my = fsum(xs) / len(xs), fsum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    pearson = sxy / sqrt(sxx * syy) if sxx > 0 and syy > 0 else None
    return {
        "workers": len(pairs),
        "pearson": pearson,
        "mae": sum(abs(x - y) for x, y in pairs) / len(pairs),
    }


def prequential_reliability_score(
    data: CrowdData,
    prior: TrustDistribution,
    *,
    order: Sequence[str] | None = None,
) -> dict[str, float | int]:
    """Score reliability predictions made before each item's labels are seen.

    At the start of an item, each worker's expected reliability is predicted
    from earlier items only. Once all predictions for the item are recorded,
    the exact within-item posterior is updated and its marginal worker trusts
    are carried forward (ADF). Gold answers are used only to score correctness,
    never to update the learner. This avoids the in-sample optimism in
    :func:`reliability_agreement`.
    """
    grouped = data.by_item()
    trust: dict[str, TrustDistribution] = {}
    abs_errors: list[float] = []
    squared_errors: list[float] = []
    per_worker: dict[str, list[float]] = {}
    prior_mean = prior.expected_reliability
    prior_abs_errors: list[float] = []
    prior_squared_errors: list[float] = []
    prior_per_worker: dict[str, list[float]] = {}
    for item in _validated_order(data, order):
        reports = grouped[item]
        if item in data.gold:
            for worker, label in reports:
                prediction = trust.get(worker, prior).expected_reliability
                correct = float(label == data.gold[item])
                error = abs(prediction - correct)
                abs_errors.append(error)
                squared_errors.append((prediction - correct) ** 2)
                per_worker.setdefault(worker, []).append(error)
                prior_error = abs(prior_mean - correct)
                prior_abs_errors.append(prior_error)
                prior_squared_errors.append((prior_mean - correct) ** 2)
                prior_per_worker.setdefault(worker, []).append(prior_error)
        state = ExactJointState(0.5)
        for worker, _ in reports:
            state.add_source(worker, trust.get(worker, prior))
        for worker, label in reports:
            state.observe(worker, label == 1)
        for worker in state.sources:
            trust[worker] = state.trust(worker)
    if not abs_errors:
        raise ValueError("no labels are attached to items with gold answers")
    worker_maes = [sum(errors) / len(errors) for errors in per_worker.values()]
    prior_worker_maes = [sum(errors) / len(errors) for errors in prior_per_worker.values()]
    return {
        "n_reports": len(abs_errors),
        "n_workers": len(worker_maes),
        "micro_mae": fsum(abs_errors) / len(abs_errors),
        "micro_brier": fsum(squared_errors) / len(squared_errors),
        "macro_worker_mae": fsum(worker_maes) / len(worker_maes),
        "prior_only_micro_mae": fsum(prior_abs_errors) / len(prior_abs_errors),
        "prior_only_micro_brier": fsum(prior_squared_errors) / len(prior_squared_errors),
        "prior_only_macro_worker_mae": fsum(prior_worker_maes) / len(prior_worker_maes),
        "micro_mae_difference_vs_prior": (
            fsum(abs_errors) - fsum(prior_abs_errors)
        ) / len(abs_errors),
        "micro_brier_difference_vs_prior": (
            fsum(squared_errors) - fsum(prior_squared_errors)
        ) / len(squared_errors),
    }


# ------------------------------------------------------------- benchmark
def run_benchmark(
    data: CrowdData,
    *,
    orders: int = 10,
    seed: int = 0,
    trust_mean: float = 2.0 / 3.0,
    trust_concentration: float = 3.0,
    trust_points: int = 101,
    min_gold_reports: int = 20,
) -> dict[str, object]:
    """Run every method; online methods over ``orders`` random item orders.

    The default trust prior Beta(mean 2/3, concentration 3) equals Beta(2, 1),
    the same prior the Dawid-Skene estimates use, so all learners start from
    the same assumption that sources are better than chance.
    """
    if isinstance(orders, bool) or not isinstance(orders, int) or orders < 1:
        raise ValueError("orders must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (isinstance(min_gold_reports, bool) or not isinstance(min_gold_reports, int)
            or min_gold_reports < 1):
        raise ValueError("min_gold_reports must be a positive integer")
    prior = TrustDistribution.beta_prior(trust_mean, trust_concentration, trust_points)
    alpha = trust_mean * trust_concentration
    beta = (1.0 - trust_mean) * trust_concentration
    results: dict[str, object] = {}

    def record(name: str, estimates: list[Estimate]) -> None:
        scores = [score(e, data.gold) for e in estimates]
        agreement = [reliability_agreement(e, data, min_gold_reports=min_gold_reports)
                     for e in estimates]
        summary: dict[str, object] = {}
        def mean_sd(values: list[float]) -> dict[str, float]:
            mean = fsum(values) / len(values)
            if min(values) == max(values):
                return {"mean": values[0], "sd_over_orders": 0.0}
            sd = sqrt(fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
            return {"mean": mean, "sd_over_orders": sd}

        for key in scores[0]:
            summary[key] = mean_sd([s[key] for s in scores])
        for key in ("pearson", "mae"):
            values = [a[key] for a in agreement if a[key] is not None]
            summary[f"reliability_{key}"] = mean_sd(values) if values else None
        summary["reliability_workers"] = agreement[0]["workers"]
        summary["runs"] = len(estimates)
        if name in ONLINE_METHODS:
            baseline = score(majority_vote(data), data.gold)
            paired = {}
            for metric in ("accuracy", "brier", "log_loss", "ece", "positive_probability_ece"):
                differences = [s[metric] - baseline[metric] for s in scores]
                center = fsum(differences) / len(differences)
                spread = (0.0 if len(differences) < 2 or min(differences) == max(differences)
                          else sqrt(fsum((x - center) ** 2 for x in differences)
                                    / (len(differences) - 1)))
                paired[metric] = {"mean_difference": center, "sd_over_orders": spread}
            summary["paired_difference_vs_majority"] = paired
        if estimates[0].iterations is not None:
            summary["em_iterations"] = estimates[0].iterations
            summary["em_converged"] = estimates[0].converged
        class_priors = [e.class_prior for e in estimates if e.class_prior is not None]
        if class_priors:
            summary["class_prior"] = mean_sd([float(x) for x in class_priors])
        results[name] = summary

    record("majority_vote", [majority_vote(data)])
    item_rng = random.Random(seed)
    report_rng = random.Random(seed + 1)
    permutations = []
    for _ in range(orders):
        order = data.items
        item_rng.shuffle(order)
        permutations.append(order)
    report_permutations = []
    for _ in range(orders):
        per_item = {}
        for item, reports in data.by_item().items():
            workers = [worker for worker, _ in reports]
            report_rng.shuffle(workers)
            per_item[item] = workers
        report_permutations.append(per_item)
    record("laputa_simultaneous", [
        laputa_online(data, prior, schedule="simultaneous", order=o, report_order=ro)
        for o, ro in zip(permutations, report_permutations)
    ])
    record("laputa_sequential", [
        laputa_online(data, prior, schedule="sequential", order=o, report_order=ro)
        for o, ro in zip(permutations, report_permutations)
    ])
    exact_estimates = [
        exact_adf(data, prior, order=o, report_order=ro)
        for o, ro in zip(permutations, report_permutations)
    ]
    record("exact_adf", exact_estimates)
    record("dawid_skene_one_coin", [dawid_skene(data, model="one_coin", alpha=alpha, beta=beta)])
    record("dawid_skene_two_coin", [dawid_skene(data, model="two_coin", alpha=alpha, beta=beta)])
    record("dawid_skene_one_coin_estimated_prior", [
        dawid_skene(data, model="one_coin", alpha=alpha, beta=beta, estimate_class_prior=True)
    ])
    record("dawid_skene_two_coin_estimated_prior", [
        dawid_skene(data, model="two_coin", alpha=alpha, beta=beta, estimate_class_prior=True)
    ])
    prequential = [prequential_reliability_score(data, prior, order=o) for o in permutations]
    exact_summary = results["exact_adf"]
    if isinstance(exact_summary, dict):
        for metric in (
            "n_reports", "n_workers", "micro_mae", "micro_brier", "macro_worker_mae",
            "prior_only_micro_mae", "prior_only_micro_brier", "prior_only_macro_worker_mae",
            "micro_mae_difference_vs_prior", "micro_brier_difference_vs_prior",
        ):
            values = [float(row[metric]) for row in prequential]
            exact_summary[f"prequential_{metric}"] = {
                "mean": fsum(values) / len(values),
                "sd_over_orders": (
                    0.0 if min(values) == max(values) else
                    sqrt(fsum((x - fsum(values) / len(values)) ** 2 for x in values) / (len(values) - 1))
                ),
            }
    return {
        "dataset": data.name,
        "summary": data.summary(),
        "settings": {
            "orders": orders, "seed": seed, "trust_prior": [trust_mean, trust_concentration],
            "trust_points": trust_points, "class_prior": 0.5,
            "min_gold_reports": min_gold_reports,
            "item_order_randomized": True, "within_item_report_order_randomized": True,
            "item_order_seed": seed, "report_order_seed": seed + 1,
            "dawid_skene_estimated_prior": "Beta(1, 1) MAP sensitivity variants included",
        },
        "methods": results,
    }


def synthetic_one_coin(
    items: int, workers: int, reports_per_item: int, *, seed: int = 0,
    accuracy_range: tuple[float, float] = (0.45, 0.95),
) -> tuple[CrowdData, dict[str, float]]:
    """Data generated by the one-coin model, with the true reliabilities."""
    if reports_per_item > workers:
        raise ValueError("reports_per_item cannot exceed workers")
    rng = random.Random(seed)
    accuracy = {f"w{j:03d}": rng.uniform(*accuracy_range) for j in range(workers)}
    names = sorted(accuracy)
    labels, gold = [], {}
    for i in range(items):
        item = f"i{i:05d}"
        z = int(rng.random() < 0.5)
        gold[item] = z
        for worker in rng.sample(names, reports_per_item):
            correct = rng.random() < accuracy[worker]
            labels.append((item, worker, z if correct else 1 - z))
    return CrowdData("synthetic_one_coin", tuple(labels), gold), accuracy
