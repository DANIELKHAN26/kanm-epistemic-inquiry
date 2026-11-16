import json
from pathlib import Path
import tempfile
import unittest

from kanm.cli import _load


class CliConfigTests(unittest.TestCase):
    def _load_text(self, contents: str) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(contents, encoding="utf-8")
            return _load(path)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self._load_text('{"agents": [], "agents": []}')

    def test_unknown_top_level_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown configuration key"):
            self._load_text('{"agents": [], "stepss": 10}')

    def test_configuration_arrays_are_checked(self):
        with self.assertRaisesRegex(ValueError, "agents must be an array"):
            self._load_text('{"agents": {}}')
        with self.assertRaisesRegex(ValueError, "edges must be an array"):
            self._load_text('{"agents": [], "edges": {}}')

    def test_valid_minimal_config_loads(self):
        config = self._load_text('{"agents": [{"agent_id": "a"}]}')
        self.assertEqual(config["agents"][0]["agent_id"], "a")



class CliV03Tests(unittest.TestCase):
    def _write(self, directory, payload):
        path = Path(directory) / "c.json"
        path.write_text(payload, encoding="utf-8")
        return path

    def test_non_finite_numbers_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "non-finite"):
                _load(self._write(d, '{"agents": [{"agent_id": "a", "prior_belief": NaN}]}'))

    def test_generated_network_config(self):
        from kanm.cli import build
        with tempfile.TemporaryDirectory() as d:
            config = _load(self._write(d, json.dumps({
                "n_agents": 5, "agent_defaults": {"inquiry_chance": 0.4},
                "network": {"type": "cycle"}, "edge_defaults": {"listen_chance": 0.3},
            })))
        agents, edges = build(config)
        self.assertEqual(len(agents), 5)
        self.assertEqual(len(edges), 10)
        self.assertTrue(all(e.listen_chance == 0.3 for e in edges))

    def test_structure_keys_are_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                _load(self._write(d, '{"agents": [], "n_agents": 3}'))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                _load(self._write(d, '{"agents": [], "schema_version": 2}'))

    def test_main_compare_runs_end_to_end(self):
        import contextlib, io
        from kanm.cli import main
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, json.dumps({
                "n_agents": 3, "network": {"type": "complete"}, "steps": 5}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main([str(path), "--replicates", "3", "--compare",
                      "update_rule=laputa,exact_joint"])
        report = json.loads(out.getvalue())
        self.assertIn('update_rule="exact_joint"', report["paired_difference_vs_baseline"])


class CliReviewRegressionTests(unittest.TestCase):
    def _load_text(self, contents):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            path.write_text(contents, encoding="utf-8")
            return _load(path)

    def test_defaults_without_generator_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "edge_defaults"):
            self._load_text('{"n_agents": 3, "edge_defaults": {"listen_chance": 0.5}}')

    def test_compare_rejects_per_run_flags(self):
        import contextlib, io
        from kanm.cli import main
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            path.write_text('{"n_agents": 2, "network": {"type": "complete"}, "steps": 2}')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(path), "--replicates", "2", "--trajectory",
                      "--compare", "update_rule=laputa,exact_joint"])


if __name__ == "__main__":
    unittest.main()
