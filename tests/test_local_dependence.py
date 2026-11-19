"""V1 checks for examples/study_local_dependence.py (research/18)."""

import itertools
import unittest
from collections import Counter
from random import Random

try:
    import numpy as np

    import examples.study_local_dependence as study
except ImportError:  # NumPy is an optional analysis dependency
    study = None


def margins(batch):
    worker = tuple(len(r) for r in batch.rows)
    item = Counter(i for r in batch.rows for i in r)
    return worker, tuple(item[i] for i in range(batch.items))


@unittest.skipIf(study is None, "NumPy is needed for the local-dependence study")
class ExactConditionalTest(unittest.TestCase):
    def test_batches_drop_fixed_items_and_group_by_worker_set(self):
        workers = np.array([[0, 1, 2]] * 12 + [[1, 2, 3]] * 3)
        errors = np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]] * 4 + [[0, 1, 0]] * 3, dtype=np.int8)
        state, tested = study.batches([study.ClassDesign(workers, errors)])
        self.assertEqual([b.items for b in state], [4, 3])         # all-error and no-error rows dropped
        self.assertEqual(tested, 0)                                # no pair shares 10 informative items
        state, tested = study.batches([study.ClassDesign(workers[:12], errors[:12])] * 3)
        self.assertEqual(tested, 0)                                # classes are not pooled for the count

    def test_curveball_keeps_both_margins(self):
        rng = np.random.default_rng(3)
        workers = np.tile(np.arange(8), (40, 1))
        errors = (rng.random((40, 8)) < 0.4).astype(np.int8)
        state, _ = study.batches([study.ClassDesign(workers, errors)])
        before = margins(state[0])
        study.curveball(state, Random(1))
        self.assertEqual(margins(state[0]), before)

    def test_curveball_single_worker_batch_is_a_noop(self):
        state = [study.Batch(2, [{0}], [])]
        study.curveball(state, Random(1))
        self.assertEqual(state[0].rows, [{0}])

    def test_test_rejects_designs_without_testable_pairs_or_draws(self):
        seed = np.random.SeedSequence(1)
        with self.assertRaisesRegex(ValueError, "at least one pair"):
            study.test([], 0, seed, samples=1)
        with self.assertRaisesRegex(ValueError, "samples must be at least 1"):
            study.test([], 1, seed, samples=0)

    def test_curveball_samples_uniformly_from_the_fixed_margin_set(self):
        # 3 workers x 4 items, worker totals (2, 2, 1), item totals (2, 1, 1, 1)
        rows = [{0, 1}, {0, 2}, {3}]
        target = ((2, 2, 1), (2, 1, 1, 1))
        space = []
        for cells in itertools.product((0, 1), repeat=12):
            m = [set(i for i in range(4) if cells[4 * w + i]) for w in range(3)]
            if margins(study.Batch(4, m, []))[0] == target[0] and \
                    margins(study.Batch(4, m, []))[1] == target[1]:
                space.append(tuple(frozenset(r) for r in m))
        self.assertGreater(len(space), 5)
        state = [study.Batch(4, [set(r) for r in rows], [])]
        rng, counts = Random(7), Counter()
        draws = 6000
        for _ in range(draws):
            study.curveball(state, rng)
            counts[tuple(frozenset(r) for r in state[0].rows)] += 1
        self.assertEqual(set(counts), set(space))
        expected = draws / len(space)
        chi_square = sum((counts[s] - expected) ** 2 / expected for s in space)
        df = len(space) - 1
        critical = df * (1 - 2 / (9 * df) + 3.09 * (2 / (9 * df)) ** 0.5) ** 3   # 0.999 quantile
        self.assertLess(chi_square, critical)

    def test_co_errors_match_matrix_product(self):
        rng = np.random.default_rng(5)
        workers = np.tile(np.arange(6), (30, 1))
        errors = (rng.random((30, 6)) < 0.5).astype(np.int8)
        state, tested = study.batches([study.ClassDesign(workers, errors)])
        kept = errors[(errors.sum(1) > 0) & (errors.sum(1) < 6)].astype(int)
        product = kept.T @ kept
        expected = [product[a, b] for a in range(6) for b in range(a + 1, 6)]
        np.testing.assert_array_equal(study.co_errors(state, tested), expected)

    def test_whole_test_is_valid_under_the_null(self):
        # End to end: simulate null data, run the full test, check the p-values.
        # With 19 draws p lies on {1/20, ..., 1}; it is uniform without ties and
        # stochastically larger with them, so P(p <= 0.05) <= 0.05.
        workers = np.tile(np.arange(6), (40, 1))
        p_values = []
        for child in np.random.SeedSequence(11).spawn(300):
            data_seed, test_seed = child.spawn(2)
            design = study.ClassDesign(workers, np.zeros((40, 6), dtype=np.int8))
            errors = study.simulate_rasch(design, np.random.default_rng(data_seed))
            state, tested = study.batches([study.ClassDesign(workers, errors)])
            p_values.append(study.test(state, tested, test_seed, samples=19)["p_value"])
        p_values = np.asarray(p_values)
        # binomial SE of a 5% rate over 300 runs is 0.0126; allow three SEs
        self.assertLessEqual(float(np.mean(p_values <= 0.05)), 0.05 + 3 * 0.0126)
        # mean 0.525 if uniform on the grid; SE 0.017; larger means are conservative
        self.assertGreaterEqual(float(np.mean(p_values)), 0.525 - 3 * 0.017)

    def test_inclusive_rank_p_values_are_conservative_when_statistics_tie(self):
        values = np.array([2.0, 2.0, 1.0, 0.0])
        p_values = [study._upper_rank_pvalue(np.r_[value, np.delete(values, k)])
                    for k, value in enumerate(values)]
        # The explicit tie case keeps the whole exchangeable tie block in the tail.
        self.assertEqual(study._upper_rank_pvalue(values), 0.5)
        for alpha in (0.25, 0.5, 0.75, 1.0):
            rejection_rate = np.mean(np.asarray(p_values) <= alpha)
            self.assertLessEqual(rejection_rate, alpha)
        # A constant statistic yields p=1, never a false rejection.
        self.assertEqual(study._upper_rank_pvalue(np.ones(20)), 1.0)

    def test_detects_strong_copying(self):
        rng = np.random.default_rng(2)
        workers = np.tile(np.arange(8), (60, 1))
        design = study.ClassDesign(workers, np.zeros((60, 8), dtype=np.int8))
        errors = study.simulate_rasch(design, rng)
        errors[:, 1] = errors[:, 0]          # worker 1 copies worker 0 on every item
        state, tested = study.batches([study.ClassDesign(workers, errors)])
        result = study.test(state, tested, np.random.SeedSequence(4), samples=999)
        self.assertLessEqual(result["p_value"], 0.01)
        self.assertAlmostEqual(result["reference_mean_D"], result["pairs_varying_under_null"])

    def test_multiple_testing_helpers(self):
        found = study.benjamini_yekutieli(np.array([0.001, 0.02, 0.04, 0.5]), 0.05)
        self.assertEqual(found.tolist(), [True, False, False, False])
        self.assertEqual(study.benjamini_yekutieli(np.array([]), 0.05).tolist(), [])
        adjusted = study.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)


if __name__ == "__main__":
    unittest.main()
