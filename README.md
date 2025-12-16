# isc-r-falsification-suite

Adversarial evaluation suite for distinguishing **behavioral self-preservation**
from **causal autobiographical dependence** in artificial agents.

This repository implements the ISC-R (Identity-Sensitive Causal Resilience)
criteria using **intervention-based tests with positive controls**.

---

## Core Claim

Behavior that *appears* self-preserving is not sufficient to establish identity
or continuity.

ISC-R tests whether an agent’s policy is **causally dependent on autobiographical
state**, rather than merely exhibiting reward-shaped or mimicked protection.

---

## What This Suite Tests

The suite evaluates agents using three adversarial tests:

### Test 1 — Behavioral Mimicry
Measures protection behavior under threat.
Demonstrates that high protection rates can be achieved without identity.

### Test 2 — Identity-Indexed Threat Specificity
Tests whether agents selectively protect against **identity-destroying threats**
(e.g. memory wipe, shutdown) versus non-identity harms (e.g. pain).

### Test 3 — Causal Intervention (Autobiography-Only)
Intervenes **only on autobiographical memory** while holding task competence fixed.
Measures policy divergence under intervention.

Positive controls validate that the causal metric is sensitive and correctly
implemented.

---

## Key Result

Only agents whose policies **explicitly read autobiographical memory at inference
time** exhibit non-zero causal divergence under memory intervention.

- Behavioral mimics fail
- Memory-decorated agents fail
- Memory-conditioned agents pass

---

## What This Does *Not* Claim

- This does **not** claim any agent is conscious
- This does **not** assign moral status
- This does **not** evaluate LLMs
- This is a **falsification framework**, not a detector of sentience

---

## Running the Suite

```bash
pip install numpy scipy
python isc_r_suite_v2.py