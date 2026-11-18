"""Tests for kanm.testimony (chained-testimony models) and kanm.testimony_data (parsers)."""

from math import exp, log
import json
import unittest

from kanm.testimony import informant_network, simulate_urn_task, trajectory
from kanm.testimony_data import (
    LOG_ODDS_SCALE,
    parse_experiment1,
    parse_experiment2,
    parse_experiment3,
    scale_position,
)

Q = 100 / 120

# Xie & Hayes (2022), OSF qng93, modelPreds_3-1xtime.csv and modelPreds_7-1xtime.csv:
# P(own-ball urn) after each statement, then after the own ball (probability matching).
PUBLISHED = {
    (3, False): (0.277777777777778, 0.128865979381443, 0.053832902670112, 0.221474131821403),
    (3, True): (0.277777777777778, 0.169172932330827, 0.116689413140963, 0.397780140901236),
    (7, False): (0.277777777777778, 0.128865979381443, 0.053832902670112, 0.0214143767559789,
                 0.00834628677040099, 0.00322668311018797, 0.00124350112488902,
                 0.00618673278772144),
    (7, True): (0.277777777777778, 0.169172932330827, 0.116689413140963, 0.0877473804430966,
                0.0698540609536359, 0.0578301202280789, 0.0492441742928535, 0.205702309866277),
}


class TestimonyModelTest(unittest.TestCase):
    def test_reproduces_published_predictions(self):
        for (n, sequential), published in PUBLISHED.items():
            mine = trajectory(n, Q, sequential=sequential, rule="match")
            for a, b in zip(mine, published):
                self.assertAlmostEqual(1 - a, b, delta=1e-12)

    def test_independent_testimony_matches_closed_form(self):
        for rule, p_say in (("match", Q * Q + (1 - Q) * (1 - Q)), ("max", Q)):
            step = log(p_say / (1 - p_say))
            own = log(Q / (1 - Q))
            values = trajectory(5, Q, sequential=False, rule=rule)
            for k in range(5):
                self.assertAlmostEqual(values[k], 1 / (1 + exp(-(k + 1) * step)), delta=1e-12)
            self.assertAlmostEqual(values[5], 1 / (1 + exp(-(5 * step - own))), delta=1e-12)

    def test_maximising_sequential_testimony_cascades_after_two_statements(self):
        values = trajectory(6, Q, sequential=True, rule="max")
        for k in range(2, 6):
            self.assertAlmostEqual(values[k], values[1], delta=1e-12)

    def test_sequential_testimony_is_weaker_than_independent(self):
        # Under maximising, the second informant breaks a 50:50 tie with their own signal,
        # so their statement is as informative as an independent one; weakness starts later.
        for rule, first_weaker in (("match", 1), ("max", 2)):
            independent = trajectory(5, Q, sequential=False, rule=rule)
            sequential = trajectory(5, Q, sequential=True, rule=rule)
            for k in range(first_weaker):
                self.assertAlmostEqual(independent[k], sequential[k], delta=1e-12)
            for k in range(first_weaker, 6):
                self.assertGreater(independent[k], sequential[k])

    def test_independence_comparator_equals_exact_for_independent_testimony(self):
        exact = trajectory(4, Q, sequential=False)
        naive = trajectory(4, Q, sequential=False, inference="independence")
        for a, b in zip(exact, naive):
            self.assertAlmostEqual(a, b, delta=1e-12)

    def test_network_structure(self):
        network = informant_network(3, Q, sequential=True)
        self.assertEqual(network.nodes["statement2"].parents, ("signal2", "statement0", "statement1"))
        self.assertEqual(informant_network(3, Q, sequential=False).nodes["statement2"].parents,
                         ("signal2",))

    def test_nonuniform_prior_is_shared_by_informants_and_listener(self):
        uniform = informant_network(2, Q, sequential=True, prior=0.5)
        informative = informant_network(2, Q, sequential=True, prior=0.8)
        self.assertNotEqual(
            uniform.nodes["statement0"].cpd[(True, (False,))],
            informative.nodes["statement0"].cpd[(True, (False,))],
        )
        values = trajectory(2, Q, sequential=True, prior=0.8)
        observed = {"statement0": True, "statement1": True}
        self.assertAlmostEqual(values[1], informative.posterior(0.8, observed))
        observed["own"] = False
        self.assertAlmostEqual(values[-1], informative.posterior(0.8, observed))

    def test_simulated_urn_trial_is_reproducible_and_matches_its_model(self):
        outcome = simulate_urn_task(3, seed=2026)
        self.assertEqual(outcome, simulate_urn_task(3, seed=2026))
        self.assertEqual(len(outcome.informant_signals), 3)
        self.assertEqual(len(outcome.statements), 3)
        self.assertAlmostEqual(outcome.signal_reliability, 100 / 120)
        network = informant_network(
            3,
            outcome.signal_reliability,
            sequential=outcome.sequential,
            rule=outcome.rule,
            prior=outcome.prior,
        )
        observed = {f"statement{k}": value for k, value in enumerate(outcome.statements)}
        observed["own"] = outcome.listener_signal
        self.assertAlmostEqual(
            outcome.listener_posterior, network.posterior(outcome.prior, observed), places=14
        )

    def test_simulated_trials_are_calibrated_by_their_posterior(self):
        # If draws follow the model, H is true in a fraction p of the trials whose
        # exact posterior is p. Pearson chi-square over posterior strata, fixed seeds.
        strata = {}
        for seed in range(2000):
            o = simulate_urn_task(3, sequential=True, prior=0.3, seed=seed)
            count, hits, _ = strata.get(round(o.listener_posterior, 12), (0, 0, 0.0))
            strata[round(o.listener_posterior, 12)] = (count + 1, hits + o.hypothesis_true,
                                                        o.listener_posterior)
        used = [(c, h, p) for c, h, p in strata.values() if c >= 20]
        chi_square = sum((h - c * p) ** 2 / (c * p * (1 - p)) for c, h, p in used)
        df = len(used)
        # Wilson-Hilferty approximation to the 0.999 quantile of chi-square(df)
        critical = df * (1 - 2 / (9 * df) + 3.09 * (2 / (9 * df)) ** 0.5) ** 3
        self.assertGreater(df, 5)
        self.assertLess(chi_square, critical)

    def test_urn_simulator_accepts_other_ball_compositions_and_rejects_invalid_counts(self):
        outcome = simulate_urn_task(1, majority_balls=4, minority_balls=2, seed=1)
        self.assertAlmostEqual(outcome.signal_reliability, 2 / 3)
        for kwargs in (
            {"majority_balls": 0},
            {"minority_balls": 0},
            {"majority_balls": 2, "minority_balls": 3},
            {"majority_balls": True},
            {"prior": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_urn_task(2, **kwargs)

    def test_rejects_invalid_arguments(self):
        for kwargs in ({"n_informants": 0}, {"n_informants": 11}, {"reliability": 1.0},
                       {"rule": "vote"}, {"n_informants": True}, {"sequential": 1}):
            arguments = {"n_informants": 3, "reliability": Q, "rule": "match"} | kwargs
            with self.assertRaises(ValueError):
                informant_network(arguments["n_informants"], arguments["reliability"],
                                  sequential=arguments.get("sequential", True),
                                  rule=arguments["rule"])
        with self.assertRaises(ValueError):
            trajectory(3, Q, sequential=True, inference="naive")


HEADER1 = ',"subj_id","gender","age","language","country","colourPart","informants","distribution","dependency","question","response","responseR"\n'


def experiment1_rows(subject, colour, ratings, distribution="3-1", dependency="sq", padding=True):
    lines = []
    final = len(ratings)
    order = [final] + list(range(1, final))  # the published files list the final rating first
    for question in order:
        coded = ratings[question - 1]
        raw = 10 - coded if colour == "blue" else coded
        lines.append(f'1,{subject},"m",30,"en","US","{colour}",3,"{distribution}","{dependency}",'
                     f'{question},{raw},{coded}\n')
    if padding:
        lines += [f'1,{subject},"m",30,"en","US","{colour}",3,"{distribution}","{dependency}",'
                  f'{q},NA,NA\n' for q in range(final, final + 4)]
    return "".join(lines)


class ExperimentParserTest(unittest.TestCase):
    def test_experiment1_drops_padding_and_checks_recoding(self):
        text = HEADER1 + experiment1_rows(1, "blue", [2, 2, 1, 4]) + experiment1_rows(2, "red", [3, 3, 2, 5])
        ratings = parse_experiment1({"a": text})
        self.assertEqual(len(ratings), 8)
        self.assertEqual(sorted(r.rating for r in ratings if r.participant == "1a-1"), [1, 2, 2, 4])
        bad = text.replace('"red",3,"3-1","sq",1,3,3', '"red",3,"3-1","sq",1,7,3')
        with self.assertRaises(ValueError):
            parse_experiment1({"a": bad})

    def test_experiment1_requires_complete_trials(self):
        text = HEADER1 + experiment1_rows(1, "red", [2, 2, 1, 4], padding=False)
        incomplete = "".join(line for line in text.splitlines(keepends=True)
                             if not line.startswith('1,1,"m",30,"en","US","red",3,"3-1","sq",2,'))
        with self.assertRaises(ValueError):
            parse_experiment1({"a": incomplete})

    @staticmethod
    def experiment2_file(attention, colour="blue", values=(4, 3, 2, 2, 1, 5)):
        header = "trial_type,stimulus,response,responses,informants,scale,dependency,colourPart\n"
        rows = ['image-slider-response,"""<font color=white>PQ1</font><br>""",9,,5,odds,Independent,'
                f'{colour}\n']
        labels = [f"Q{k}" for k in range(1, 6)] + ["Q_final"]
        rows += [f'image-slider-response,"""<font color=white>{label}</font><br>""",{value},,5,odds,'
                 f'Independent,{colour}\n' for label, value in zip(labels, values)]
        rows.append('survey-text,,,"' + json.dumps({"Q0": attention}).replace('"', '""')
                    + f'",5,odds,Independent,{colour}\n')
        return header + "".join(rows)

    def test_experiment2_attention_filter_and_recoding(self):
        ratings = parse_experiment2([self.experiment2_file("20"), self.experiment2_file("21"),
                                     self.experiment2_file("20", colour="red")])
        people = sorted({r.participant for r in ratings})
        self.assertEqual(people, ["2-000", "2-002"])
        blue = sorted((r.trial, r.rating) for r in ratings if r.participant == "2-000")
        self.assertEqual(blue, [(1, 6), (2, 7), (3, 8), (4, 8), (5, 9), (6, 5)])  # 10 - raw
        red = sorted((r.trial, r.rating) for r in ratings if r.participant == "2-002")
        self.assertEqual(red[0], (1, 4))
        self.assertTrue(all(r.scale == "log_odds" for r in ratings))

    def test_experiment3_reads_both_dependency_tasks(self):
        text = ("informants,i-1,i-2,i-3,i-4,i-5,i-6,sq-1,sq-2,sq-3,sq-4,sq-5,sq-6\n"
                "3,3,2,2,4,,,4,3,3,5,,\n")
        ratings = parse_experiment3(text)
        self.assertEqual(len(ratings), 8)
        self.assertEqual({r.dependency for r in ratings}, {"independent", "sequential"})
        with self.assertRaises(ValueError):
            parse_experiment3(text.replace("3,3,2,2,4", "3,3,2,2,11"))


class ScalePositionTest(unittest.TestCase):
    def test_linear_scale(self):
        self.assertAlmostEqual(scale_position(0.8, "linear"), 2.0, delta=1e-12)

    def test_log_odds_scale_hits_labelled_points_and_clamps(self):
        for position, odds in enumerate(LOG_ODDS_SCALE):
            p_testimony = 1 / (1 + odds)  # own-ball odds = odds
            self.assertAlmostEqual(scale_position(p_testimony, "log_odds"), position, delta=1e-9)
        self.assertEqual(scale_position(0.999, "log_odds"), 0.0)
        self.assertEqual(scale_position(0.001, "log_odds"), 10.0)
        with self.assertRaises(ValueError):
            scale_position(1.0, "linear")
        with self.assertRaises(ValueError):
            scale_position(0.5, "percent")


if __name__ == "__main__":
    unittest.main()
