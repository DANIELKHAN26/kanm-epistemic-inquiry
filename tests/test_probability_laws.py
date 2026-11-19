"""Laws that any correct Bayesian implementation must satisfy, checked exactly.

1. Conservation of expected evidence: under the agent's own predictive
   distribution, the expected posterior equals the prior, E[P(H | E)] = P(H).
   Credence is a martingale. This is the property Dorst (2023) relies on:
   predictable polarization needs evidence the agent cannot fully read.
2. The same holds for the expected reliability of a source after a report.
3. Data processing: reports that only relay one study cannot carry more
   evidence than the study itself, however many there are.
"""

import itertools
import math
import unittest

from kanm.bayes import ExactJointState, TrustDistribution, update_belief
from kanm.dependence import ProvenanceNetwork, corroborating_relay, relay, symmetric_primary
from kanm.laputa16 import Laputa16Trust
from kanm.testimony import informant_network


def predictive(q, trust):
    """P(positive report) under the single-report model."""
    return q * trust.expected_reliability + (1 - q) * trust.expected_unreliability


class ConservationOfExpectedEvidenceTest(unittest.TestCase):
    TRUSTS = (TrustDistribution.beta_prior(0.7, 6, 201), TrustDistribution.beta_prior(0.4, 2, 101),
              Laputa16Trust.from_beta(0.7, 6.0), Laputa16Trust.from_beta(0.55, 20.0))

    def test_single_report_belief_and_trust_are_martingales(self):
        for trust, q in itertools.product(self.TRUSTS, (0.03, 0.2, 0.5, 0.9)):
            with self.subTest(trust=repr(trust), q=q):
                p = predictive(q, trust)
                belief = p * update_belief(q, trust, True) + (1 - p) * update_belief(q, trust, False)
                reliability = (p * trust.observe(q, True).expected_reliability
                               + (1 - p) * trust.observe(q, False).expected_reliability)
                self.assertAlmostEqual(belief, q, delta=1e-13)
                self.assertAlmostEqual(reliability, trust.expected_reliability, delta=1e-13)

    def test_exact_joint_belief_is_a_martingale_over_report_sequences(self):
        # From every history of up to three reports by two sources, the predictive
        # expectation of the next posterior equals the current posterior.
        priors = {"a": TrustDistribution.beta_prior(0.7, 4, 41),
                  "b": TrustDistribution.beta_prior(0.6, 10, 41)}
        steps = [(s, v) for s in priors for v in (True, False)]
        for length in range(4):
            for history in itertools.product(steps, repeat=length):
                state = ExactJointState(0.35)
                for source, prior in priors.items():
                    state.add_source(source, prior)
                for source, positive in history:
                    state.observe(source, positive)
                for source in priors:
                    expected = 0.0
                    total = 0.0
                    for positive in (True, False):
                        p = math.exp(state.log_predictive(source, positive))
                        nxt = ExactJointState(0.35)
                        for s, prior in priors.items():
                            nxt.add_source(s, prior)
                        for s, v in history + ((source, positive),):
                            nxt.observe(s, v)
                        expected += p * nxt.belief
                        total += p
                    self.assertAlmostEqual(total, 1.0, delta=1e-12)
                    self.assertAlmostEqual(expected, state.belief, delta=1e-12)

    def test_provenance_posterior_averages_to_the_prior(self):
        net = ProvenanceNetwork()
        net.add(symmetric_primary("study", 0.8))
        net.add(relay("r1", "study", 0.9))
        net.add(corroborating_relay("r2", "study", 0.85, 0.7, 0.4))
        net.add(symmetric_primary("own", 0.65))
        nodes = ("r1", "r2", "own")
        for prior in (0.1, 0.5, 0.77):
            expected = 0.0
            total = 0.0
            for values in itertools.product((True, False), repeat=len(nodes)):
                obs = dict(zip(nodes, values))
                log_true, log_false = net._log_evidence(obs)
                p = prior * math.exp(log_true) + (1 - prior) * math.exp(log_false)
                expected += p * net.posterior(prior, obs)
                total += p
            self.assertAlmostEqual(total, 1.0, delta=1e-12)
            self.assertAlmostEqual(expected, prior, delta=1e-12)

    def test_chained_testimony_posterior_averages_to_the_prior(self):
        for sequential, rule in itertools.product((True, False), ("match", "max")):
            net = informant_network(3, 100 / 120, sequential=sequential, rule=rule, prior=0.3)
            nodes = ("statement0", "statement1", "statement2", "own")
            expected = 0.0
            for values in itertools.product((True, False), repeat=len(nodes)):
                obs = dict(zip(nodes, values))
                log_true, log_false = net._log_evidence(obs)
                p = 0.3 * math.exp(log_true) + 0.7 * math.exp(log_false)
                if p > 0.0:
                    expected += p * net.posterior(0.3, obs)
            with self.subTest(sequential=sequential, rule=rule):
                self.assertAlmostEqual(expected, 0.3, delta=1e-12)


class DataProcessingTest(unittest.TestCase):
    def test_relays_never_carry_more_evidence_than_their_source(self):
        # H -> study -> relays is a Markov chain, so P(H | relays) is a garbling of
        # P(H | study): agreeing relays can at most reach the study's own posterior.
        for reliability, fidelity in itertools.product((0.6, 0.75, 0.95), (0.55, 0.8, 0.99)):
            bound = reliability   # P(H | study says H) at prior 1/2
            for n in range(1, 9):
                net = ProvenanceNetwork()
                net.add(symmetric_primary("study", reliability))
                for k in range(n):
                    net.add(relay(f"r{k}", "study", fidelity))
                heard = {f"r{k}": True for k in range(n)}
                with self.subTest(reliability=reliability, fidelity=fidelity, n=n):
                    self.assertLessEqual(net.posterior(0.5, heard), bound + 1e-12)
                    # treating them as independent breaks the bound once n is large enough
                    if n == 8 and fidelity >= 0.8:
                        self.assertGreater(net.independence_posterior(0.5, heard), bound)


if __name__ == "__main__":
    unittest.main()
