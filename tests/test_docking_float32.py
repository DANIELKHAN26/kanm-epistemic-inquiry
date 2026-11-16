"""Checks of the single-precision emulation used to diagnose the docking residuals."""

import unittest

from kanm.laputa16 import Laputa16Trust

try:
    import numpy as np

    from docking.laputa_1_6.float32 import emulated, expectation_float32, observe_float32
except ImportError:  # numpy is optional; the docking diagnostic is not part of the package
    np = None


@unittest.skipIf(np is None, "numpy is required for the float32 emulation")
class Float32EmulationTest(unittest.TestCase):
    TRUSTS = (
        Laputa16Trust.default_batch(0.3),
        Laputa16Trust.from_value_sharpness(0.8, 0.6),
        Laputa16Trust.from_beta(0.4, 12.0),
    )

    def test_update_stays_close_to_double_and_is_single_precision(self):
        for trust in self.TRUSTS:
            for belief in (0.03, 0.37, 0.5, 0.91):
                for positive in (True, False):
                    emulated_values = observe_float32(trust.values, belief, positive)
                    exact = trust.observe(belief, positive).values
                    for a, b in zip(emulated_values, exact):
                        self.assertEqual(a, float(np.float32(a)))
                        self.assertLessEqual(abs(a - b), 1e-5 * max(1.0, abs(b)))

    def test_expectation_close_to_closed_form(self):
        for trust in self.TRUSTS:
            self.assertAlmostEqual(expectation_float32(trust.values), trust.expected_reliability, delta=1e-6)

    def test_constant_likelihood_keeps_function_up_to_rounding(self):
        trust = self.TRUSTS[1]
        for a, b in zip(observe_float32(trust.values, 0.5, True), trust.values):
            self.assertAlmostEqual(a, b, delta=1e-5 * max(1.0, b))

    def test_context_manager_restores_double_precision(self):
        trust = self.TRUSTS[0]
        before = (trust.observe(0.7, True), trust.expected_reliability)
        with emulated():
            inside = trust.expected_reliability
            self.assertEqual(inside, float(np.float32(inside)))
        self.assertEqual((trust.observe(0.7, True), trust.expected_reliability), before)

    def test_context_manager_restores_after_error(self):
        saved = (Laputa16Trust.__dict__["observe"], Laputa16Trust.__dict__["expected_reliability"])
        with self.assertRaises(ValueError):
            with emulated():
                self.TRUSTS[0].observe(1.5, True)
        self.assertIs(Laputa16Trust.__dict__["observe"], saved[0])
        self.assertIs(Laputa16Trust.__dict__["expected_reliability"], saved[1])


if __name__ == "__main__":
    unittest.main()
