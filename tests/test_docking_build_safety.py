"""Regression checks for the external Laputa docking build wrapper."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docking" / "laputa_1_6" / "build.sh"


class DockingBuildSafetyTests(unittest.TestCase):
    def test_existing_output_is_refused_and_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "valuable-output"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("user data", encoding="utf-8")
            result = subprocess.run(
                ["bash", str(SCRIPT), str(Path(temp) / "missing-source"), str(output)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data")


if __name__ == "__main__":
    unittest.main()
