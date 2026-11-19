"""The Python examples in README.md run, and their stated results hold."""

from pathlib import Path
import re
import unittest

README = Path(__file__).resolve().parents[1] / "README.md"


class ReadmeExamplesTest(unittest.TestCase):
    def test_examples_run_and_match_their_comments(self):
        blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
        self.assertGreaterEqual(len(blocks), 3)
        namespace: dict = {}
        for block in blocks:
            exec(compile(block, str(README), "exec"), namespace)
        plan = namespace["plan"]
        self.assertAlmostEqual(plan.guarantee, 0.0275, delta=5e-5)
        steps = plan.first_step_probabilities
        self.assertAlmostEqual(steps[("ask", "check")], 9 / 34, delta=1e-12)
        self.assertAlmostEqual(steps[("ask", "read-0")], 25 / 34, delta=1e-12)

        from kanm.robust_planning import value_of_provenance_information
        value = value_of_provenance_information(
            0.5, namespace["weighted"], namespace["actions"], namespace["utilities"], horizon=3
        ).value
        self.assertAlmostEqual(value, 0.0149, delta=5e-5)


if __name__ == "__main__":
    unittest.main()
