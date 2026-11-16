"""Equation-level checks against worked numbers printed in the Laputa literature.

These verify the single-step credence kernel and the simultaneous combination
rule against published values (reported to 3-5 decimals, so tolerances follow
the printed precision). They are not a docking of whole simulation runs.
"""

import unittest

from kanm.bayes import TrustDistribution, update_belief
from kanm.model import AgentSpec, EdgeSpec, Simulation


def point_trust(mean):
    """Narrow trust density with the stated expectation (only E[tau] matters here)."""
    return TrustDistribution.beta_prior(mean, 1e6, points=20001)


class PublishedWorkedValues(unittest.TestCase):
    def test_olsson_2013_appendix_two_messages(self):
        # Prior 0.8; two positive messages with <tau> = 0.7 and 0.9 -> "0.99" (0.988).
        q = update_belief(update_belief(0.8, point_trust(0.7), True), point_trust(0.9), True)
        self.assertAlmostEqual(q, 0.504 / 0.51, places=4)
        self.assertAlmostEqual(q, 0.988, places=3)

    def test_hahn_hansen_olsson_appendix_b_single_reports(self):
        # (prior, E[tau], positive?, printed posterior)
        cases = [
            (0.5, 0.513, True, 0.51299),
            (0.27923, 0.642, False, 0.17761),
            (0.75815, 0.452, False, 0.79160),
        ]
        for prior, mean, positive, printed in cases:
            with self.subTest(prior=prior):
                got = update_belief(prior, point_trust(mean), positive)
                self.assertAlmostEqual(got, printed, delta=2e-4)

    def test_hahn_hansen_olsson_appendix_b_simultaneous_update(self):
        # Sherlock at 0.91 receives inquiry p (E[tau]=0.371) and Moriarty's
        # not-p (E[tau]=0.656) in one step; printed result 0.75815. Combining
        # both gives 0.7577 (difference from rounding of the printed trust
        # values). With one report per source every schedule and update rule
        # gives this value, so the example checks the kernel, not the schedule.
        joint = update_belief(update_belief(0.91, point_trust(0.371), True),
                              point_trust(0.656), False)
        self.assertAlmostEqual(joint, 0.75815, delta=1e-3)

    def test_simulator_combines_inquiry_and_message_like_published_example(self):
        agents = [
            AgentSpec("sherlock", prior_belief=0.91, inquiry_chance=1.0, inquiry_accuracy=1.0,
                      inquiry_trust_mean=0.371, inquiry_trust_concentration=1e6),
            AgentSpec("moriarty", prior_belief=0.1, inquiry_chance=0.0),
        ]
        edges = [EdgeSpec("moriarty", "sherlock", trust_mean=0.656, trust_concentration=1e6)]
        result = Simulation(agents, edges, steps=1, trust_points=20001).run()
        self.assertAlmostEqual(result.final_beliefs["sherlock"], 0.75815, delta=1e-3)


class AssertionRuleTests(unittest.TestCase):
    def _messages(self, rule, belief, threshold):
        agents = [AgentSpec("a", prior_belief=belief, inquiry_chance=0.0),
                  AgentSpec("b", inquiry_chance=0.0)]
        edges = [EdgeSpec("a", "b", assertion_threshold=threshold)]
        return Simulation(agents, edges, steps=1, assertion_rule=rule).run().message_events

    def test_published_rule_respects_threshold_size(self):
        self.assertEqual(self._messages("published", 0.6, 0.75), 0)
        self.assertEqual(self._messages("published", 0.8, 0.75), 1)

    def test_laputa_1_6_rule_ignores_threshold_size(self):
        self.assertEqual(self._messages("laputa_1_6", 0.6, 0.75), 1)
        self.assertEqual(self._messages("laputa_1_6", 0.6, 0.99), 1)

    def test_liar_threshold_inverts_polarity(self):
        agents = [AgentSpec("a", prior_belief=0.1, inquiry_chance=0.0),
                  AgentSpec("b", inquiry_chance=0.0)]
        edges = [EdgeSpec("a", "b", assertion_threshold=0.2, trust_mean=0.8)]
        result = Simulation(agents, edges, steps=1).run()
        self.assertGreater(result.final_beliefs["b"], 0.5)  # liar says p while believing not-p


if __name__ == "__main__":
    unittest.main()
