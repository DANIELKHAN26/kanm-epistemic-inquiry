"""Tests for kanm.judgment_inversion and kanm.structure_data."""

import json
import unittest

from kanm import judgment_inversion as J
from kanm.structure_data import OWN_EVIDENCE, STRONG_BLUE, WEAK_RED, parse, parse_triads

try:
    import examples.study_structure_weight as weight_study
except ImportError:  # NumPy/SciPy are optional analysis dependencies
    weight_study = None

# Saved inferences of Fränken et al. (2024), github.com/janphilippfranken/spacefish @ a1681e8,
# analysis/inferred_observations/exp_1_cond_{1,3}_*.txt, room 0: (red, blue) after trials 5 and 10.
SAVED = {
    "level1_B": ((4.745411989473011e-05, 2.515433548310476), (0.0007695705230228913, 6.063672096050271)),
    "level2_C_watching_B": ((2.9998816420156684, 5.21323336817197e-06),
                            (7.573167039189157, 0.0007813516355522058)),
    "level2_B_watching_C": ((4.7390494569804325e-05, 3.012299842040264),
                            (5.077228228179143e-05, 7.602501624010964)),
}


class JudgmentInversionTest(unittest.TestCase):
    def test_reproduces_published_inferences(self):
        b = J.infer_evidence(STRONG_BLUE)
        c = J.infer_evidence(WEAK_RED)
        mine = {
            "level1_B": b,
            "level2_C_watching_B": J.infer_evidence(WEAK_RED, parents=[b]),
            # in the B-sees-C file the strong sequence is the watcher of the weak one
            "level2_B_watching_C": J.infer_evidence(STRONG_BLUE, parents=[c]),
        }
        for key, (after5, after10) in SAVED.items():
            for got, want in ((mine[key][4], after5), (mine[key][9], after10)):
                self.assertAlmostEqual(got[0], want[0], delta=1e-12)
                self.assertAlmostEqual(got[1], want[1], delta=1e-12)

    def test_bins_and_responses_are_distributions(self):
        for red, blue in ((0, 0), (2.5, 0.3), (7.6, 6.1)):
            bins = J.bin_probabilities(red, blue)
            self.assertAlmostEqual(sum(bins), 1.0, delta=1e-12)
            self.assertAlmostEqual(sum(J.response_probabilities(bins, 3.0)), 1.0, delta=1e-12)
        self.assertAlmostEqual(J.expected_judgment(J.bin_probabilities(0, 0)), 0.0, delta=1e-12)

    def test_inversion_is_mirror_symmetric(self):
        forward = J.infer_evidence(STRONG_BLUE)
        mirrored = J.infer_evidence([-j for j in STRONG_BLUE])
        for (r1, b1), (r2, b2) in zip(forward, mirrored):
            self.assertAlmostEqual(r1, b2, delta=1e-12)
            self.assertAlmostEqual(b1, r2, delta=1e-12)

    def test_level2_structure_effect_direction(self):
        b, c = J.infer_evidence(STRONG_BLUE), J.infer_evidence(WEAK_RED)
        c_sees_b = J.listener_bins(OWN_EVIDENCE, [b, J.infer_evidence(WEAK_RED, parents=[b])])
        b_sees_c = J.listener_bins(OWN_EVIDENCE, [J.infer_evidence(STRONG_BLUE, parents=[c]), c])
        for t in range(5, 10):
            self.assertGreater(J.expected_judgment(c_sees_b[t]), J.expected_judgment(b_sees_c[t]))

    def test_listener_timing(self):
        others = [[(5.0, 0.0)] * 3]
        bins = J.listener_bins((None, "red", None), others)
        self.assertEqual(bins[0], J.bin_probabilities(0, 0))       # others' judgments not yet seen
        self.assertEqual(bins[1], J.bin_probabilities(6, 0))       # own red fish plus trial-1 counts

    def test_sticky_likelihood(self):
        bins = [J.bin_probabilities(0, 0)] * 3
        plain = J.judgment_log_likelihood([1, 1, None], bins, 2.0)
        sticky = J.judgment_log_likelihood([1, 1, None], bins, 2.0, sticky=0.5)
        self.assertAlmostEqual(plain[0], sticky[0], delta=1e-12)   # no previous judgment yet
        self.assertGreater(sticky[1], plain[1])                     # repeat is more likely
        self.assertIsNone(sticky[2])
        with self.assertRaises(ValueError):
            J.judgment_log_likelihood([4], bins, 2.0)
        with self.assertRaises(ValueError):
            J.infer_evidence([0], evidence_prior={"red": 0.5, "blue": 0.5})


def room(condition_hint="strong", swap=False, fish_trial=1, judgments=(0,) * 10):
    strong = {"planet_selections": list(STRONG_BLUE), "sim_resp": [True] * 10}
    weak = {"planet_selections": list(WEAK_RED), "sim_resp": [True] * 10}
    two, three = (weak, strong) if swap else (strong, weak)
    human = {"info": {"structureHint": condition_hint, "t_fish": fish_trial, "fish": "red"},
             "planet_selections": list(judgments), "sim_resp": [False] * 9 + [True]}
    return {"simulated_player_two": two, "simulated_player_three": three, "person": human}


class StructureDataTest(unittest.TestCase):
    def test_parses_conditions_and_autofilled_flags(self):
        text = json.dumps({"r1": room(judgments=(0, 1, 1, 0, 0, -1, -1, -1, -2, -2))})
        (p,) = parse(text, "c_sees_b", "test")
        self.assertEqual(p.condition, "c_sees_b")
        self.assertEqual(p.judgments[-1], -2)
        self.assertTrue(p.autofilled[-1])
        (q,) = parse(json.dumps({"r1": room(swap=True)}), "b_sees_c", "test")
        self.assertEqual(q.condition, "b_sees_c")

    def test_rejects_departures_from_the_design(self):
        with self.assertRaises(ValueError):
            parse(json.dumps({"r1": room()}), "b_sees_c", "test")           # sequences not swapped
        with self.assertRaises(ValueError):
            parse(json.dumps({"r1": room(condition_hint="no")}), "c_sees_b", "test")
        with self.assertRaises(ValueError):
            parse(json.dumps({"r1": room(fish_trial=3)}), "c_sees_b", "test")
        with self.assertRaises(ValueError):
            parse(json.dumps({"r1": room(judgments=(0,) * 9 + (4,))}), "c_sees_b", "test")



def triad_room(condition="lr", a=(0,) * 10, b=(0, 0, 0, -1, -1, -1, -1, -1, -1, -1),
               c=(0, 0, 0, -1, -1, -1, 0, 0, 0, 0), keys=("p1", "p2", "p3"), timing=(1, 3, 6)):
    room = {}
    for key, fish_trial, judgments in zip(keys, timing, (a, b, c)):
        room[key] = {"info": {"condition": condition, "t_fish": fish_trial, "fish": "red"},
                     "planet_selections": list(judgments), "sim_resp": [False] * 10}
    return room


class TriadDataTest(unittest.TestCase):
    def test_parses_roles_and_removes_exact_duplicates(self):
        text = json.dumps({"r1": triad_room(), "r2": triad_room(), "r3": {},
                           "r4": triad_room(condition="independent", a=(1,) * 10)})
        triads = parse_triads(text, 2)
        self.assertEqual(len(triads), 2)                        # r2 duplicates r1; r3 is empty
        self.assertEqual({t.condition for t in triads}, {"c_sees_b", "independent"})
        self.assertEqual(triads[0].b_judgments[3], -1)

    def test_rejects_unexpected_rooms(self):
        with self.assertRaises(ValueError):
            parse_triads(json.dumps({"r": triad_room(timing=(1, 3, 3))}), 2)   # roles not identifiable
        mixed = triad_room()
        mixed["p2"]["info"]["condition"] = "independent"
        with self.assertRaises(ValueError):
            parse_triads(json.dumps({"r": mixed}), 2)


@unittest.skipIf(weight_study is None, "NumPy and SciPy are needed for the vectorised study code")
class VectorisedLikelihoodTest(unittest.TestCase):
    def test_matches_reference_functions(self):
        records = [([0, 1, None, 0, -1, -1, -2, -2, -2, -1], STRONG_BLUE, WEAK_RED, "C"),
                   ([0, 1, 0, 0, 0, 0, -1, -1, -1, -1], (0, 0, -1, -1, -1, -1, -2, -2, -2, -2),
                    (0, 0, 0, 0, 0, 0, 1, 1, 1, 1), "C"),
                   ([0, 1, 1, 0, 0, -1, -1, -2, -2, -2], STRONG_BLUE, WEAK_RED, "B")]
        design = weight_study.build_design(records, J.UNIFORM_EVIDENCE, 10.0)
        self.assertLess(weight_study.check_against_reference(records, design), 1e-12)

    def test_decision_categories(self):
        cases = {(0.1, 0.6): "partial", (0.3, 1.2): "consistent_with_full", (-0.2, 0.5):
                 "not_detected_and_below_full", (-0.3, 1.4): "uninformative", (1.1, 1.5): "greater_than_full",
                 (-0.9, -0.1): "below_zero"}
        for interval, expected in cases.items():
            self.assertEqual(weight_study.decision(interval), expected)


if __name__ == "__main__":
    unittest.main()
