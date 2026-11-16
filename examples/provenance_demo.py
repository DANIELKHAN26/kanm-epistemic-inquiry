"""Exact repeat tracking, scope audit, and exact dependence-aware inference."""

from kanm.dependence import ProvenanceNetwork
from kanm.evidence import EvidenceItem, EvidenceLedger, assess_scope


ledger = EvidenceLedger()
ledger.add(EvidenceItem("study-001", "goal", "positive", "research-team",
                        scope={"population": "adults", "setting": "primary-care"}))
ledger.add(EvidenceItem("summary-001", "goal", "positive", "reviewer",
                        parents=("study-001",), scope={"population": "adults"}, kind="summary"))
ledger.add(EvidenceItem("news-001", "goal", "positive", "journalist",
                        parents=("study-001",), kind="news"))

print("first exposure:", ledger.receive("reader", "study-001"))
print("repeat exposure:", ledger.receive("reader", "study-001"))
print("scope of summary for adults in primary care:",
      assess_scope({"population": "adults", "setting": "primary-care"},
                   ledger.items["summary-001"]).status)

# A reader sees the summary and the news item, not the study itself. How much
# should two reports that both descend from one study count? That depends on
# an explicit model: here the study is correct with probability 0.8 and each
# relay reproduces its parent's conclusion with probability 0.9.
network, observed = ProvenanceNetwork.from_ledger(
    ledger, question_id="goal", positive_outcome="positive", negative_outcome="negative",
    primary_reliability=0.8, relay_fidelity=0.9, observed=["summary-001", "news-001"],
)
report = network.dependence_report(0.5, observed)
print("exact posterior (shared source modelled):", round(report["exact_posterior"], 4))
print("posterior if treated as independent:     ", round(report["independence_posterior"], 4))
print("overcounting ratio (log-odds):           ", round(report["overcounting_ratio"], 3))
