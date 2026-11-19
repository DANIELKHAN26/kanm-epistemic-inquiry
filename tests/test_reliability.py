"""Tests for kanm.reliability and the data-set parsers in kanm.crowd_data."""

from itertools import product
import math
from pathlib import Path
import tempfile
import unittest

from kanm.bayes import TrustDistribution
from kanm.crowd_data import _verified, download, parse_bluebirds
from kanm.reliability import (
    CrowdData,
    dawid_skene,
    exact_adf,
    laputa_online,
    majority_vote,
    prequential_reliability_score,
    reliability_agreement,
    run_benchmark,
    score,
    synthetic_one_coin,
    _map_binomial_rate,
)

PRIOR = TrustDistribution.beta_prior(2 / 3, 3.0, 41)


def correlation(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))


class ValidationTests(unittest.TestCase):
    def test_crowd_data_validation(self):
        with self.assertRaises(ValueError):
            CrowdData("x", ())
        with self.assertRaises(ValueError):
            CrowdData("x", (("i", "w", 2),))
        with self.assertRaises(ValueError):
            CrowdData("x", (("i", "w", True),))
        with self.assertRaises(ValueError):
            CrowdData("x", (("i", "w", 1),), {"i": 3})
        with self.assertRaisesRegex(ValueError, "at most one"):
            CrowdData("x", (("i", "w", 1), ("i", "w", 0)))
        with self.assertRaisesRegex(ValueError, "integers"):
            CrowdData("x", (("i", "w", 1.0),))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            CrowdData("x", (("", "w", 1),))

    def test_gold_labels_cannot_be_mutated_after_validation(self):
        data = CrowdData("x", (("i", "w", 1),), {"i": 1})
        with self.assertRaises(TypeError):
            data.gold["i"] = 0

    def test_order_must_be_permutation(self):
        data = CrowdData("x", (("a", "w", 1), ("b", "w", 0)))
        with self.assertRaisesRegex(ValueError, "permutation"):
            exact_adf(data, PRIOR, order=["a"])
        with self.assertRaisesRegex(ValueError, "report_order"):
            laputa_online(data, PRIOR, report_order={"a": ["missing"]})

    def test_checksum_mismatch(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            _verified(b"data", "0" * 64, "file")

    def test_download_refuses_symlink_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside.txt"
            outside.write_text("keep", encoding="utf-8")
            target = root / "data"
            target.mkdir()
            (target / "bluebirds_gt.yaml").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                download(target, ("welinder2010_bluebirds",))
            self.assertEqual(outside.read_text(encoding="utf-8"), "keep")


class MethodTests(unittest.TestCase):
    def test_score_hand_computed(self):
        from kanm.reliability import Estimate
        estimate = Estimate("x", {"a": 1.0, "b": 0.5, "c": 0.2}, {})
        result = score(estimate, {"a": 1, "b": 0, "c": 1})
        self.assertAlmostEqual(result["accuracy"], 1.5 / 3)
        self.assertAlmostEqual(result["brier"], (0 + 0.25 + 0.64) / 3)
        self.assertAlmostEqual(result["log_loss"], (math.log(2) + math.log(5)) / 3)
        self.assertEqual(result["log_loss_clipped_items"], 0)
        self.assertIn("positive_probability_ece", result)

    def test_score_rejects_bad_bins_and_probabilities(self):
        from kanm.reliability import Estimate
        with self.assertRaisesRegex(ValueError, "bins"):
            score(Estimate("x", {"a": 0.5}, {}), {"a": 1}, bins=0)
        with self.assertRaisesRegex(ValueError, "finite"):
            score(Estimate("x", {"a": float("nan")}, {}), {"a": 1})

    def test_dawid_skene_uniform_prior_keeps_boundary_rates_finite(self):
        # With Beta(1, 1), a worker who has no effective successes or failures
        # has a boundary MAP. The E-step must never take log(0).
        self.assertEqual(_map_binomial_rate(0.0, 3.0, 1.0, 1.0), 1e-12)
        self.assertEqual(_map_binomial_rate(3.0, 3.0, 1.0, 1.0), 1.0 - 1e-12)
        data = CrowdData("boundary", (("positive", "w", 1), ("negative", "w", 0)))
        for model in ("one_coin", "two_coin"):
            estimate = dawid_skene(data, model=model, alpha=1.0, beta=1.0)
            self.assertTrue(all(math.isfinite(p) for p in estimate.posteriors.values()))
            self.assertTrue(all(math.isfinite(r) for r in estimate.reliability.values()))

    def test_laputa_simultaneous_cannot_learn_and_equals_majority_vote(self):
        data, _ = synthetic_one_coin(300, 30, 4, seed=2)  # even count: exact ties occur
        mv = majority_vote(data).posteriors
        self.assertGreater(sum(1 for p in mv.values() if p == 0.5), 10)
        est = laputa_online(data, PRIOR, schedule="simultaneous")
        for item, p in est.posteriors.items():
            self.assertEqual(p == 0.5, mv[item] == 0.5)
            if p != 0.5:
                self.assertEqual(p > 0.5, mv[item] > 0.5)
        # Trust updated at credence 1/2 is unchanged, exactly (not just to rounding).
        self.assertTrue(all(r == PRIOR.expected_reliability for r in est.reliability.values()))
        self.assertEqual(score(est, data.gold)["accuracy"], score(majority_vote(data), data.gold)["accuracy"])

    def test_exact_adf_single_item_is_exact(self):
        reports = (("i", "a", 1), ("i", "b", 1), ("i", "c", 0))
        est = exact_adf(CrowdData("x", reports), PRIOR)
        # Brute force: sum over z and each worker's grid point.
        totals = {0: 0.0, 1: 0.0}
        for z in (0, 1):
            for cells in product(range(len(PRIOR.grid)), repeat=3):
                weight = 0.5
                for (_, _, label), k in zip(reports, cells):
                    r = PRIOR.grid[k]
                    weight *= PRIOR.weights[k] * (r if label == z else 1 - r)
                totals[z] += weight
        self.assertAlmostEqual(est.posteriors["i"], totals[1] / (totals[0] + totals[1]), places=12)

    def test_learners_recover_reliability_on_synthetic_data(self):
        data, truth = synthetic_one_coin(600, 25, 5, seed=4)
        mv_acc = score(majority_vote(data), data.gold)["accuracy"]
        for estimate in (exact_adf(data, PRIOR), dawid_skene(data, model="one_coin")):
            with self.subTest(method=estimate.method):
                workers = sorted(truth)
                r = correlation([estimate.reliability[w] for w in workers],
                                [truth[w] for w in workers])
                self.assertGreater(r, 0.9)
                self.assertGreater(score(estimate, data.gold)["accuracy"], mv_acc)

    def test_two_coin_close_to_one_coin_on_symmetric_data(self):
        data, _ = synthetic_one_coin(600, 25, 5, seed=5)
        one = dawid_skene(data, model="one_coin")
        two = dawid_skene(data, model="two_coin")
        agree = sum((one.posteriors[i] > 0.5) == (two.posteriors[i] > 0.5) for i in data.items)
        self.assertGreater(agree / len(data.items), 0.97)

    def test_dawid_skene_validates_iteration_limit_and_estimates_prior(self):
        data, _ = synthetic_one_coin(120, 12, 4, seed=8)
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            dawid_skene(data, max_iterations=0)
        estimate = dawid_skene(data, model="two_coin", estimate_class_prior=True)
        self.assertTrue(0.0 < estimate.class_prior < 1.0)

    def test_two_coin_handles_an_unobserved_class_with_flat_accuracy_prior(self):
        data = CrowdData("one_class", (("i1", "w", 0), ("i2", "w", 0)))
        estimate = dawid_skene(data, model="two_coin", alpha=1.0, beta=1.0)
        self.assertTrue(all(math.isfinite(p) for p in estimate.posteriors.values()))

    def test_reliability_agreement_needs_enough_workers(self):
        data, _ = synthetic_one_coin(50, 5, 2, seed=1)
        result = reliability_agreement(majority_vote(data), data)
        self.assertIsNone(result["pearson"])  # majority vote estimates no reliabilities

    def test_benchmark_runs_and_is_reproducible(self):
        data, _ = synthetic_one_coin(120, 12, 4, seed=3)
        a = run_benchmark(data, orders=2, trust_points=21)
        b = run_benchmark(data, orders=2, trust_points=21)
        self.assertEqual(a, b)
        self.assertEqual(set(a["methods"]), {
            "majority_vote", "laputa_simultaneous", "laputa_sequential", "exact_adf",
            "dawid_skene_one_coin", "dawid_skene_two_coin",
            "dawid_skene_one_coin_estimated_prior", "dawid_skene_two_coin_estimated_prior"})
        self.assertTrue(a["settings"]["within_item_report_order_randomized"])
        self.assertIn("prequential_micro_mae", a["methods"]["exact_adf"])
        self.assertIn("paired_difference_vs_majority", a["methods"]["exact_adf"])

    def test_prequential_reliability_uses_only_previous_items_for_prediction(self):
        data = CrowdData(
            "x",
            (("first", "w", 1), ("first", "x", 1), ("second", "w", 0)),
            {"first": 1, "second": 0},
        )
        result = prequential_reliability_score(data, PRIOR, order=["first", "second"])
        self.assertEqual(result["n_reports"], 3)
        # First item predictions use the 2/3 prior; only after scoring those
        # predictions can the agreement update reliability for the second.
        self.assertLess(result["micro_mae"], 1 / 3)
        self.assertIn("prior_only_micro_brier", result)
        self.assertIn("micro_brier_difference_vs_prior", result)


class ParserTests(unittest.TestCase):
    def test_bluebirds_parser(self):
        gold = "{1: true, 2: false,\n  3: true}\n"
        labels = "w1: {1: true, 2: true,\n  3: false}\nw2: {1: false, 3: true}\n"
        data = parse_bluebirds(gold, labels)
        self.assertEqual(data.gold, {"1": 1, "2": 0, "3": 1})
        self.assertEqual(sorted(data.labels), sorted([
            ("1", "w1", 1), ("2", "w1", 1), ("3", "w1", 0), ("1", "w2", 0), ("3", "w2", 1)]))

    def test_bluebirds_parser_rejects_unparsed_and_duplicate_entries(self):
        with self.assertRaisesRegex(ValueError, "unparsed"):
            parse_bluebirds("{1: true}", "unexpected\nw1: {1: true}")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_bluebirds("{1: true, 1: false}", "w1: {1: true}")


if __name__ == "__main__":
    unittest.main()
