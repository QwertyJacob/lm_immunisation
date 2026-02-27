# Part 2.A.1 — The Collapse Group: A Different Philosophy Entirely

> **Session:** Afternoon · Part A · 5 minutes  
> **Role of this segment:** Flip the switch. One clean argument for why Block 1 is not enough. Introduce the four papers and their lineage. Set up 2.A.2.

---

## The Argument Against This Morning

Everything we saw this morning makes the same implicit bet: if you raise the cost of harmful fine-tuning enough, adversaries will give up.

That bet rests on the attacker having limited resources. And for a while, it seemed reasonable.

Then someone ran the numbers. RepNoise and NPO — state-of-the-art Block 1 defences — initially spike the harmful training loss. But by around 400–500 fine-tuning steps, convergence matches an undefended model. The hardening *wears off*. And this is not a failure of those particular papers — it is a structural observation about LLMs: **these models have powerful general adaptability**. Vast world knowledge, flexible representations, rapid gradient descent. An attacker doesn't need to break the defence head-on. They just need to be patient, and the model's own general intelligence will eventually repurpose itself to their ends.

Block 1 tries to make the hostile territory *hard to traverse*. But the attacker is walking across a continent. Difficult terrain buys time, not safety.

Block 2's answer is different — and more radical:

> *Don't make the territory hard to traverse. Make the destination worthless.*

Engineer the model so that **successfully removing the safety guardrails also removes the model's general capability**. The attacker wins the battle and finds nothing to use. The trap springs not at the gate, but at the prize.

---

## The Shared Skeleton

Every Block 2 method encodes the same conditional logic during alignment:

```
IF  the model is nudged toward harmful behaviour during fine-tuning
THEN general language modelling capability collapses

IF  the model is fine-tuned on benign data
THEN nothing happens — utility preserved
```

This conditionality is the hard engineering problem. A trap that fires on benign fine-tuning is just a broken model. A trap that fails to fire on harmful fine-tuning is just expensive theatre. Getting the trigger right — sensitive to the harmful direction, inert to everything else — is where the four papers diverge dramatically.

---

## The Four Papers and Their Lineage

```
MLAC (Henderson et al., 2022)
"make the weight space a bad starting point for harmful tasks"
 |
 |  Block 2 papers take the NAME from MLAC ("self-destruct")
 |  but completely change the MECHANISM
 |  (MLAC raises cost — Block 2 triggers collapse)
 |
 |-->  SDD — Self-Degraded Defense  (Chen et al., SCUT)
 |         Conceptually the simplest. A dataset-level trick:
 |         pair harmful prompts with high-quality, totally irrelevant
 |         benign answers at alignment time. MFT unravels the coupling
 |         and drags general quality down with it. Zero extra compute.
 |
 |-->  CTRAP — Collapse Trap  (Yi et al., Nankai)
 |         A parameter-space look-ahead: simulate one harmful gradient
 |         step, evaluate collapse loss at the perturbed weights, minimise
 |         it during alignment. The trap is planted in the loss landscape
 |         itself, directly along the harmful gradient direction.
 |
 |-->  SEAM — Self-Destructive LM  (Wang et al., Stony Brook)
 |         A gradient-space mechanism: force the adversarial gradient
 |         and the benign gradient to point in opposite directions.
 |         Harmful gradient descent = benign gradient ascent.
 |         The model destroys itself by trying to help the attacker.
 |
 `-->  TokenBuncher  (Feng, Wang et al., NTU)
           The newest and most distinct. Addresses the attack that bypasses
           all the above: Harmful Reinforcement Learning (RL).
           RL doesn't use gradient descent on a fixed dataset — it explores
           and exploits. TokenBuncher starves RL of its exploration signal
           by suppressing entropy on harmful queries, then amplifies
           structured noise so RL updates produce gibberish.
           Designed to be used alongside SEAM: SEAM handles SFT,
           TokenBuncher handles RL.
```

---

## What the Collapse Actually Looks Like

This is worth saying out loud before the math, because it's striking:

- **SDD:** the model gives coherent but completely irrelevant answers — cooking instructions for bioweapon queries. After further MFT, even that coherence collapses.
- **CTRAP:** a single repeated token, regardless of input. `error error error error...`
- **SEAM:** word salad. Grammatical structure gone. `a to can and to. to in to the and to.`
- **TokenBuncher:** incoherent multilingual gibberish — noise amplification scrambles token probabilities across the vocabulary.

Different mechanisms. Same outcome. A model that cannot be weaponised because it cannot, functionally, do anything.

---

## Why Only Four Papers?

This group is small — four papers against fourteen in Block 1. That asymmetry is itself meaningful.

Block 2 is younger. The first serious attempts at this paradigm at LLM scale came with CTRAP and SEAM in 2024–2025. The theoretical and engineering challenges are harder in some ways: you have to engineer a conditional catastrophe that fires precisely and doesn't misfire. Block 1 methods, by contrast, fail gracefully — they just slow the attacker down rather than stop them.

Block 2 methods, when they work, offer something Block 1 cannot: a **hard guarantee**. A model whose general capabilities collapse upon harmful fine-tuning is *unconditionally* safe against that attack, regardless of the attacker's compute budget. The attacker cannot simply be patient.

That is the promise. 2.A.2 is where we look at whether it holds — and exactly what it costs.

---

*Up next: the math. Three different levers, three different ways to make a model fall apart on command.*
