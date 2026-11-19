import unittest

from kanm.evidence import EvidenceItem, EvidenceLedger, assess_scope


class EvidenceTests(unittest.TestCase):
    def test_ledger_tracks_exact_repeat_without_collapsing_shared_ancestry(self):
        ledger = EvidenceLedger()
        ledger.add(EvidenceItem("raw", "q", "yes", "lab"))
        ledger.add(EvidenceItem("report-a", "q", "yes", "alice", parents=("raw",)))
        ledger.add(EvidenceItem("report-b", "q", "yes", "bob", parents=("raw",)))
        self.assertTrue(ledger.receive("reader", "report-a"))
        self.assertFalse(ledger.receive("reader", "report-a"))
        self.assertEqual(ledger.unseen("reader", ("report-a", "report-b")), ("report-b",))
        self.assertIn("report-b", ledger.items)

    def test_ledger_requires_parent_first_and_unique_ids(self):
        ledger = EvidenceLedger()
        with self.assertRaisesRegex(ValueError, "added first"):
            ledger.add(EvidenceItem("child", "q", "yes", "a", parents=("parent",)))
        ledger.add(EvidenceItem("parent", "q", "yes", "a"))
        with self.assertRaisesRegex(ValueError, "duplicate evidence ID"):
            ledger.add(EvidenceItem("parent", "q", "no", "b"))

    def test_read_only_items_view_preserves_append_only_validation(self):
        ledger = EvidenceLedger()
        item = EvidenceItem("root", "q", "yes", "source")
        ledger.add(item)
        with self.assertRaises(TypeError):
            ledger.items["bypass"] = item

    def test_scope_assessment_distinguishes_match_mismatch_and_missing(self):
        evidence = EvidenceItem(
            "e1", "q", "yes", "lab", scope={"population": "adults", "setting": "clinic"}
        )
        self.assertEqual(assess_scope({"population": "adults"}, evidence).status, "in_scope")
        mismatch = assess_scope({"population": "children"}, evidence)
        self.assertEqual(mismatch.status, "out_of_scope")
        self.assertEqual(mismatch.mismatched_fields, ("population",))
        unknown = assess_scope({"population": "adults", "time": "2020s"}, evidence)
        self.assertEqual(unknown.status, "indeterminate")
        self.assertEqual(unknown.unknown_fields, ("time",))

    def test_out_of_scope_dominates_unknown(self):
        evidence = EvidenceItem("e", "q", "yes", "s", scope={"setting": "lab"})
        result = assess_scope({"setting": "field", "population": "adults"}, evidence)
        self.assertEqual(result.status, "out_of_scope")
        self.assertEqual(result.unknown_fields, ("population",))


if __name__ == "__main__":
    unittest.main()
