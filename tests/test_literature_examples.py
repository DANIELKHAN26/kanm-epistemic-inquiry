"""Worked examples from the literature, reproduced with KANM's own code."""

import itertools
import unittest

from kanm.bayes import ExactJointState, TrustDistribution, update_belief
from kanm.dependence import ProvenanceNetwork, ReportNode, relay, symmetric_primary


class ConnerTrustExamplesTest(unittest.TestCase):
    """Conner (2026), Episteme, Appendix, Calculations 1 and 2.

    Prior credence 0.8 in h; one report against h. Only the mean of the
    reliability profile enters a single update (P(e | h) = 1 - mean).
    """

    def test_trust_reducing_and_trust_increasing_profiles(self):
        for mean, expected in ((0.8, 0.5), (0.3, 0.56 / 0.62)):
            trust = TrustDistribution.beta_prior(mean, 10.0, 1001)
            with self.subTest(mean=mean):
                self.assertAlmostEqual(update_belief(0.8, trust, False), expected, delta=1e-9)


class PilditchSharedEvidenceTest(unittest.TestCase):
    """Pilditch, Hahn & Lagnado, "The problem of dependency" (SSRN preprint 4712769,
    pp. 22-24): evidence E with P(E | H) = P(not E | not H) = 0.8, reported on.

    The reporters' fidelity of 0.8 is implied by the paper's 68 %; the 76 % and
    82 % then follow from the two structures.
    """

    @staticmethod
    def network(shared: bool) -> ProvenanceNetwork:
        net = ProvenanceNetwork()
        if shared:
            net.add(symmetric_primary("evidence", 0.8))
            for k in range(2):
                net.add(relay(f"reporter{k}", "evidence", 0.8))
        else:
            for k in range(2):
                net.add(symmetric_primary(f"evidence{k}", 0.8))
                net.add(relay(f"reporter{k}", f"evidence{k}", 0.8))
        return net

    def test_one_report_and_two_structures(self):
        one = self.network(shared=True).posterior(0.5, {"reporter0": True})
        shared = self.network(shared=True).posterior(0.5, {"reporter0": True, "reporter1": True})
        separate = self.network(shared=False).posterior(0.5, {"reporter0": True, "reporter1": True})
        self.assertAlmostEqual(one, 0.68, delta=1e-12)
        self.assertAlmostEqual(shared, 0.52 / 0.68, delta=1e-12)          # 76 %
        self.assertAlmostEqual(separate, 0.4624 / 0.5648, delta=1e-12)    # 82 %


class YoungDependenceNetworkTest(unittest.TestCase):
    """Young, Madsen & de-Wit (2025), Cognition 259, 106126, p. 3, Tables 1-2.

    Three sources testify. Each is accurate when its error node ER is 1 and
    says true or false at random otherwise. ER copies a shared reliability node
    when the group is dependent (DEP = 1) and the source's own node otherwise.
    All priors are 0.5, as in the article.
    """

    @staticmethod
    def group(dep_prior: float, n: int = 3) -> ProvenanceNetwork:
        net = ProvenanceNetwork()
        root = lambda node, p: ReportNode(node, (), {(True, ()): p, (False, ()): p})  # noqa: E731
        net.add(root("dep", dep_prior))
        net.add(root("lsr", 0.5))
        for k in range(n):
            net.add(root(f"lir{k}", 0.5))
            cpd = {(h, (d, s, i)): float(s if d else i)
                   for h in (True, False) for d in (True, False) for s in (True, False) for i in (True, False)}
            net.add(ReportNode(f"er{k}", ("dep", "lsr", f"lir{k}"), cpd))
            net.add(ReportNode(f"tes{k}", (f"er{k}",),
                               {(True, (True,)): 1.0, (False, (True,)): 0.0,
                                (True, (False,)): 0.5, (False, (False,)): 0.5}))
        return net

    def test_likelihood_ratios_of_three_agreeing_sources(self):
        said = {f"tes{k}": True for k in range(3)}
        net = self.group(0.5)
        self.assertAlmostEqual(net.posterior(0.5, {**said, "dep": False}), 27 / 28, delta=1e-12)
        self.assertAlmostEqual(net.posterior(0.5, {**said, "dep": True}), 9 / 10, delta=1e-12)
        self.assertAlmostEqual(net.posterior(0.5, said), 12.6 / 13.6, delta=1e-12)

    def test_two_groups_with_the_articles_dependence_priors(self):
        # Fig. 3B: the group saying "true" has dependence prior 0.2, the group
        # saying "false" 0.8. The two groups are independent given H.
        low = self.group(0.2).posterior(0.5, {f"tes{k}": True for k in range(3)})
        high = self.group(0.8).posterior(0.5, {f"tes{k}": False for k in range(3)})
        odds = (low / (1 - low)) * (high / (1 - high))
        self.assertAlmostEqual(low / (1 - low), 18.0, delta=1e-12)
        self.assertAlmostEqual(odds / (1 + odds), 306 / 477, delta=1e-12)   # +14.15 points


class HahnOrderEffectsTest(unittest.TestCase):
    """Hahn, Merdes & von Sydow (2018), TopiCS 10(4), Fig. 2B: all orderings of six
    confirming and three disconfirming reports from one source, trust Beta(2, 1).

    Laputa's local update depends on the order; the exact joint update cannot,
    because conditioning on the same set of reports gives the same posterior.
    Their footnote 8 names the two ways out: a joint model fixed in advance, or
    recomputing from all past evidence. KANM's exact joint state is the first.
    """

    REPORTS = (True,) * 6 + (False,) * 3

    def test_laputa_update_is_order_dependent_and_exact_joint_is_not(self):
        orders = sorted(set(itertools.permutations(self.REPORTS)))
        self.assertEqual(len(orders), 84)
        prior_trust = TrustDistribution.beta_prior(2 / 3, 3.0, 401)   # Beta(2, 1)
        for prior in (0.2, 0.5, 0.8):
            laputa, exact = [], []
            for order in orders:
                belief, trust = prior, prior_trust
                state = ExactJointState(prior)
                state.add_source("s", prior_trust)
                for positive in order:
                    belief, trust = update_belief(belief, trust, positive), trust.observe(belief, positive)
                    state.observe("s", positive)
                laputa.append(belief)
                exact.append(state.belief)
            with self.subTest(prior=prior):
                self.assertGreater(max(laputa) - min(laputa), 0.01)
                self.assertLess(max(exact) - min(exact), 1e-12)


if __name__ == "__main__":
    unittest.main()
