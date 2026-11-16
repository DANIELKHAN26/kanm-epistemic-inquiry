# ODD description of the network simulator (v0.5.0 development)

This is an ODD-style description of `kanm.model.Simulation`, following Grimm et al. (2020). The planner and dependence modules are documented in [03](03_proposed_model.md) and [05](05_implementation_status.md); they do not act inside the simulator.

## 1. Overview

### Purpose and patterns

The simulator provides an inspectable, reproducible implementation of Laputa-style inquiry and testimony about one binary proposition, together with:
- an exact Bayesian reference agent under the same assumptions;
- explicit alternatives where the published model and the 1.6 code differ.

The patterns targeted for verification are the published single-step updates and simultaneous combination (worked logs in Olsson 2013 and HHO 2020), and the qualitative V-against-trust curves of Vallinder & Olsson (2014).

### Entities, state variables, and scales

- **Agents:**
  - ID;
  - credence (held as log-odds);
  - inquiry accuracy and inquiry chance;
  - trust in own inquiry: a marginal density under `laputa`, or conditional densities given `p` and given `¬p` under `exact_joint`;
  - an own random stream.
- **Directed links:** sender, receiver, listen chance, assertion threshold, the receiver's trust in the sender (same representation as above), and an own random stream.
- **Global:** fixed truth of `p`; steps; seed; trust-grid resolution; update rule; schedule; assertion rule; optional `publish_only_after_inquiry`.
- **Time and space:** discrete steps on a finite network; no spatial scale.

### Process overview and scheduling (default `simultaneous`)

At each step *t*:

1. Each agent, in lexical ID order, draws two variates from its own stream. With the inquiry chance it obtains a result, which is correct with the inquiry accuracy.
2. Each link draws two variates from its own stream. Given the sender's start-of-step credence `C_t` and the assertion rule, the sender may assert `p` or `¬p`. The message is received if the listen draw is below the listen chance.
3. Each agent conditions on all reports it received at *t*:
   - Under `laputa`, credence adds the log-likelihood ratios computed from each source's pre-step expected trust, and each source's trust density is updated with `C_t`.
   - Under `exact_joint`, the joint posterior is updated report by report (the updates commute).
4. All credences are replaced simultaneously.

Results do not depend on the order in which agents or links are listed. This order of operations matches the Laputa 1.6 source (`Society::evolve`).

The alternative `inquiry_then_messages` schedule (v0.2.0) applies inquiry first, then generates messages from post-inquiry credences and delivers them sequentially using running credences.

The `laputa_1_6` schedule restates `Society::evolve` and `Inquirer::doInquiry` of the 1.6 source, including the new-evidence policies:
- **Processing order:** agents are processed in sorted-ID order (Laputa's inquirer index), each conditioning on its own inquiry and then on its incoming links in sorted sender order. Every trust update uses the start-of-step credence.
- **`evidence_policy="none"`:** a link reports the sign of the sender's start-of-step credence. The result equals `simultaneous` with `assertion_rule="laputa_1_6"` exactly.
- **`"inquiry"`:** a link passes on its stored message only if the sender has inquired since the link was last used.
- **`"any"`:** also if another of the sender's incoming links was used more recently.
- **Stored messages:** each is set at the end of the step from the sender's latest inquiry result (or belief change) and the threshold (`≥ T`), and is kept unchanged otherwise. Initial messages come from starting beliefs with strict comparisons.
- **Order dependence:** under `"inquiry"` and `"any"` a lower-indexed sender's same-step inquiry already counts as new evidence, so outcomes can depend on agent order, as in the original.

## 2. Design concepts

- **Basic principles:**
  - Bayesian conditionalisation with symmetric source reliability;
  - Communication Independence and Source Independence (Olsson 2013);
  - under `exact_joint`, the exact posterior implied by the same assumptions, without re-imposing independence of `p` and reliability after each step.
- **Emergence:** consensus, polarisation and collective error emerge from local inquiry, thresholded assertion and trust learning.
- **Adaptation and learning:** agents update credence and trust; they do not change links, activity or thresholds.
- **Objectives and prediction:** none in the simulator.
- **Sensing:** agents observe inquiry results and message polarity, not message content or ancestry.
- **Interaction:** directed messages governed by thresholds and listen chance.
- **Stochasticity:** independent MT19937 streams per agent and per link, seeded by SHA-512 hashing of the JSON key `[seed, kind, id…]`. Every stream draws a fixed number of variates per step, so conditions run with the same seed share random inputs (common random numbers).
- **Observation:**
  - final credences and trust;
  - mean Brier score and log score (computed from log-odds; `null` if any agent is certain of the falsehood);
  - accuracy (strictly on the true side) and undecided fraction;
  - V-value (change in mean credence in the truth);
  - dispersion (population SD) and spread (max − min);
  - event counts and an optional per-step trajectory.

## 3. Details

### Initialisation

Each agent starts at its configured prior. Each trust density is initialised in one of two ways, depending on `trust_model`.

**`beta_grid` (default).** A Beta prior with the configured mean, which must lie in (0, 1), and concentration, discretised into `trust_points` equal cells:
- each cell carries its exact Beta probability, placed at the cell's exact conditional mean;
- this preserves the prior mean exactly and never places mass at r = 0 or 1;
- weights are held in log space.

**`laputa_1_6`.** Laputa 1.6's 49-node piecewise-linear density, given either:
- as explicit node values, used as given if their trapezoid mass is within 5% of 1; or
- as a Beta density sampled at the nodes and normalised, which requires both shape parameters to be ≥ 1.

The 1.6 batch default trust prior can be reproduced with `Laputa16Trust.default_batch(w)`, `w ~ U(0, 1)`. Configurations are validated before running; [08](08_v0.3_audit_and_extensions.md) lists the defects these checks prevent.

### Input data

User JSON, either explicit or generated (`n_agents`, `agent_defaults`, `network`, `edge_defaults`). No external data.

### Submodels

1. **Inquiry result:** correct with probability equal to the inquiry accuracy.
2. **Laputa credence update:** add `±log(μ/(1−μ))` to the log-odds, where `μ` is the expected reliability.
3. **Laputa trust update:** `τ'(r) ∝ τ(r)[r q + (1−r)(1−q)]` for a positive report and its mirror image for a negative report, with `q = C_t`.
4. **Exact joint update:** multiply the conditional density given `p` by `r` (or `1−r`) and the conditional density given `¬p` by `1−r` (or `r`), and add the log normalisers to the log-weights of `p` and `¬p`.
5. **Assertion:**
   - `published`: with `T > 0.5`, say `p` if `C ≥ T` and `¬p` if `C ≤ 1−T`; with `T < 0.5`, the liar rule; with `T = 0.5`, random. Comparisons are made in log-odds against `logit(max(T, 1−T))`, so mirrored configurations give mirrored decisions exactly; `T ∈ {0, 1}` asserts only at certainty.
   - `laputa_1_6`: the sign of `C − 0.5` for `T > 0.5`, reversed for `T < 0.5`, random at ties.
6. **Communication restriction (`publish_only_after_inquiry`):** a sender may publish at *t* only if it obtained an inquiry result at *t − 1* (under the simultaneous schedule) or at *t* (under `inquiry_then_messages`). This is not the 1.6 new-evidence policy.

## Intended interpretation

Outputs are conditional consequences of the stated configuration. The simulator has passed equation-level checks against published worked values and a partial qualitative reproduction. Its `laputa_1_6` options have been docked against the compiled Laputa 1.6 model core: the logic is reproduced exactly, and the only differences are Laputa's single-precision trust arithmetic (see the [docking report](../docking/laputa_1_6/README.md#status-and-results-2026-11-16)). It has not reproduced a full published run, and it has not passed any human-data validation, and it should not be used as a predictor or a policy simulator.
