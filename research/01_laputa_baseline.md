# Laputa: capabilities and baseline assumptions

## What the historical model can do

Laputa is a Bayesian agent-based simulation framework for inquiry and communication in directed social networks. The original software record lists C++ and FLTK, GPLv3, and a last update of 20 June 2014. SourceForge lists the `Laputa-1.6-src.zip` source archive under `epistemenet`; it is a sizeable legacy archive, not a current maintained Python or NetLogo port. The host describes the intended settings as research groups, editorial boards, expert groups, and the public. See the [official project page](https://sourceforge.net/projects/epistemenet/) and [source archive listing](https://sourceforge.net/projects/epistemenet/files/Sources/).

Published descriptions document these core abilities:

- Represent agents as nodes and communication opportunities as directed edges.
- Represent each agent's credence in a single proposition `p` and allow information to arrive through private inquiry and testimony/argument from peers.
- Set actual inquiry accuracy separately from the agent's trust in their own inquiry.
- Update an agent's credence and its trust distribution for each source after receiving a positive or negative report.
- Set inquiry chance, message/listen chance, assertion thresholds, edge trust, and starting parameters.
- Run one network or batches of randomly generated networks/parameter settings and summarize outcomes.
- Compare truth-tracking outcomes using veritistic measures such as change in average credence when ground truth is conventionally set to true.

The [2013 model exposition](https://lucris.lub.lu.se/ws/portalfiles/portal/5684445/7758617.pdf) describes the simulation, its options, and the underlying equations; the [2014 paper](https://doi.org/10.1007/s11229-013-0375-0) develops the Bayesian trust framework and studies overconfidence. The software archive and the papers are distinct evidence: the papers describe a mathematical model, while the archive records the historical application.

## Minimal mathematical summary

For an agent with prior credence `q = P(p)` and a trust density `τ(r)` over a source's reliability `r ∈ [0,1]`, let `μ = Eτ[r]`. Under the model's symmetric source assumption, a positive report updates belief as

`q+ = q μ / (q μ + (1-q)(1-μ))`.

A negative report updates it as

`q- = q (1-μ) / (q (1-μ) + (1-q) μ)`.

The trust density also changes conditional on the message. For a positive report its unnormalized posterior is proportional to

`τ(r) [r q + (1-r)(1-q)]`;

for a negative report it is proportional to

`τ(r) [(1-r) q + r(1-q)]`.

Each density must be normalized. In the multi-source credence update, the original derivation uses conditional source independence so the report likelihoods can be multiplied. These expressions summarize the published binary model; they do not specify every legacy software scheduling or numerical detail.

## Assumptions that matter for interpretation

1. **One binary question.** The core account asks whether one proposition `p` or `¬p` is true. It does not represent an open research agenda, a hierarchy of questions, or a set of competing explanations.
2. **Conditional independence of sources.** Multiple reports are treated as independent conditional on `p` and on `¬p`. This is mathematically convenient but can yield incorrect posteriors when reports share data, repeat a claim, or descend from another agent's report. Shared positive evidence can be overcounted in common cases, but the primary Laputa exposition explicitly cautions that the assumption has no established systematic direction of distortion across cases. Human social-learning studies further show that people do not always follow the normative Bayesian treatment of dependency.
3. **Symmetric accuracy.** The binary reliability parameter treats positive reports under `p` and negative reports under `¬p` symmetrically. Some applications need asymmetric false-positive and false-negative rates.
4. **Black-box message content.** The core update represents positive/negative reports; it does not encode a structured argument, dataset, method, or scope statement.
5. **Inquiry opportunities are parameterized.** Inquiry chance and accuracy are set or sampled; the baseline does not choose which question to investigate according to expected value, cost, or task relevance.
6. **A simulation result is conditional on a model and its settings.** Veritistic scores do not by themselves establish that a model predicts human behavior or that one institutional policy is best in every setting.
7. **Communication Independence is re-imposed every step.** Olsson (2013) states as an axiom that credence in `p`, whether a source speaks, and the source's reliability are independent. The model stores only a scalar credence and one marginal trust density per source. After a report, `p` and `r` are in fact correlated, so the rule is exact for one report per source but a product-of-marginals approximation thereafter, and it is order-dependent. This diagnosis is published by Merdes, von Sydow & Hahn (2021) and Hahn, Merdes & von Sydow (2018). KANM adds an exact reference rule; see [08](08_v0.3_audit_and_extensions.md).

The 2013 exposition explicitly discusses source independence, source symmetry, repetition, and the interpretation of reports as novel arguments. It also reports an experimental variant that allows communication only after new inquiry. KANM's `publish_only_after_inquiry` flag approximates this but does not match the 1.6 code's per-link "new evidence" policies.

## Verified from the Laputa 1.6 source and the worked examples (v0.3 check)

- **Within-step order:** all inquiry results and messages at step *t* use start-of-step credences, and each agent combines them in one update before all credences are replaced. KANM's default `simultaneous` schedule follows this.
- **Trust updates with several simultaneous messages:** each uses the same pre-step credence.
- **Assertion threshold:**
  - The threshold is per link.
  - Under the 1.6 default policy, the code asserts the side of 0.5 the credence lies on and never compares the credence with the threshold. That contradicts the published rule.
  - KANM offers both rules (`assertion_rule`).
- **Trust grid:** the code uses a 49-point piecewise-linear trust density.

Details and sources are in [08_v0.3_audit_and_extensions.md](08_v0.3_audit_and_extensions.md). The 2018 network study reports that clustering/connectivity can lower collective competence under its studied conditions; it does not establish that less communication is always better.

## Software-history audit of the initial note

| Initial note claim | Finding |
|---|---|
| SourceForge hosts Laputa and lists 20 June 2014 as its last update | **Verified** on the official `epistemenet` project page. |
| The listed source is C++ and GPLv3 | **Verified** on the official project page. |
| A GitHub copy at `sofieang/laputa` dates to January 2020 | **Unverified.** Exact-path search/open did not yield an accessible repository record. Do not cite this as fact until a repository URL or archive is verified. |
| There is no maintained Python or NetLogo version | **Not established.** A 2024 PLOS ONE paper reports a NetLogo reimplementation of an Olsson social-network model, but that is not necessarily a maintained, feature-complete port of the original Laputa software. No exhaustive implementation census was conducted. |
| The 1.6 source is available and readable | **Verified.** It was read in November 2026 to specify scheduling, trust handling and assertion behaviour. It has not been compiled or run. |
| A new implementation should be checked against published Laputa results before extensions are added | **Strong recommendation, not an absolute proof rule.** It follows from the importance of implementation details and the model authors' own acknowledgement that software mistakes can affect conclusions. |
| Rewriting from equations avoids the GPLv3 obligation | **Legal conclusion not made here.** If code is copied or translated, preserve its license obligations and attribution; if using equations only, assess the applicable legal and institutional requirements before release. |
