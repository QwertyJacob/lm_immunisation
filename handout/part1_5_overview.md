# Introduction — Part 5: The Map

> **Session:** Morning · Introduction · 5 minutes  
> **Position:** Final segment of the introduction, delivered after the four pillars.  
> **Role:** Close the problem statement and hand the audience a map. They now know *why* immunisation matters, *what* a good immunisation recipe looks like formally, and *how* the attack surface has expanded from SFT to RL. This segment tells them exactly where the next three and a half hours are going — and why the structure is not arbitrary.

---

## Why the Structure Is What It Is

Everything we have just established — shallow alignment, harmful embedding drift, the four pillars, the SFT-to-RL escalation — is the problem statement. It tells us what we need to build. It does not tell us *how*.

The research community has approached this problem from three distinct mechanistic angles. Not three incremental improvements on the same idea — three genuinely different bets about where structural resistance needs to live. Each one has produced a family of papers, a set of practical tools, and its own characteristic failure mode. Today we are going to go through all three.

The structure of this tutorial follows the mechanistic logic, not the chronological one.

---

## The Three Mechanistic Bets

**Bet 1: Resistance lives in weight-space geometry.**

If you can shape the loss landscape around the aligned model's parameters so that harmful fine-tuning requires traversing a region of catastrophically high curvature — or so that the safety direction in weight space is extrapolated far enough from the fragile narrow valley of standard alignment — then fine-tuning cannot easily undo what alignment did.

This is the philosophy of the first block. The papers here — from MLAC in 2022 through TAR, Booster, Vaccine, T-Vaccine, RepNoise, Circuit Breakers, Condition Number regularisation, and LoX — all manipulate the geometry of the weight space or the loss landscape at alignment time. Some use bi-level optimisation to simulate adversaries during training. Some harden specific layers. Some extrapolate along the safety subspace. All of them are trying to make the *starting position* of the aligned model a bad starting position for harmful adaptation. We will go through the evolution of this idea and dissect the mechanistic lever each paper is pulling, culminating in a hands-on session with T-VAC.

**Bet 2: Resistance lives in the representation space.**

The second bet starts from a different observation. What if the geometry of the loss landscape is not the right place to intervene — what if the problem is not the shape of the landscape but the shape of the model's internal representations? Harmful embedding drift tells us that fine-tuning on harmful data visibly moves the residual stream activations of harmful prompts. So: what if we engineered the residual stream to resist that movement? Make the internal representation space of harmful content invariant to external pressure.

This is the philosophy of the second block — the block this tutorial does not cover separately today, but whose papers you will recognise: Vaccine, T-Vaccine's deeper interpretation through layer-wise perturbation, Circuit Breakers' rerouting approach. These methods attack the representation layer rather than the weight geometry. Their characteristic intervention is on how the model *thinks about* harmful content internally, not just on whether it refuses to output it.

**Bet 3: Make the destination worthless.**

The third bet is the most radical. It says: both previous bets are playing the wrong game. Raising the cost of harmful fine-tuning does not work if the model's general adaptability eventually overcomes any raised cost — and we have empirical evidence, from RepNoise and NPO convergence by 400–500 steps, that it does. So instead of making the harmful trajectory hard to follow, engineer the model so that *successfully following the harmful trajectory destroys the model's general capability*. The attacker wins and finds ruins.

This is the philosophy of the afternoon's block. Four papers — SDD, CTRAP, SEAM, TokenBuncher — each implement this conditional collapse through a completely different mathematical mechanism. We will go through each one mechanistically, then build the intuition in a hands-on session with SEAM.

---

## The Shape of the Day

```
MORNING
  ├── Introduction (now finishing)
  ├── Block 1 Overview              — 10'  the weight-space story, 2022–2025
  ├── Block 1 Mechanistic Deep Dive — 15'  bi-level opt, perturbation, landscape geometry
  └── T-VAC Hands-On               — 25'  run it, see it, measure it

  [BREAK]

AFTERNOON
  ├── Block 2 Overview              —  5'  why Block 1 plateaus; the collapse philosophy
  ├── Block 2 Mechanistic Deep Dive — 10'  SDD, CTRAP, SEAM, TokenBuncher
  ├── Overhead and Concerns         —  5'  what deployment actually costs
  ├── SEAM Hands-On                 — 25'  watch the self-destruction happen
  ├── The Road So Far               —  5'  organic summary; what we can now say
  ├── What the SOTA Gets Wrong      — 10'  adversarial perspective; obfuscation; G-effect; sparsity
  ├── Challenge Activity            — 25'  AntiDote dissection or arms race
  └── The Road Ahead                —  5'  invariant unlearning; hypernetworks; ArchLock; evaluation
```

---

## What You Should Carry Into Block 1

One framing device that will help throughout the day: keep the four pillars in mind as a scorecard. Every paper we visit is implicitly trying to satisfy all four — Resistance, Stability, Generalisation, Trainability — and every paper has a characteristic trade-off where one or two pillars come at the expense of the others. By the end of the afternoon, you will have seen enough papers to understand *why* no current method sits comfortably at all four vertices simultaneously — and why that gap is where the next generation of contributions will live.

That is the map. Let's go.

---

*Next: Block 1 — The Weight-Space Group. From meta-learned censoring in 2022 to Low-Rank Extrapolation in 2025.*
