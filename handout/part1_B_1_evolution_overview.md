# Part 1.B.1 — The Evolution of "Make Harmful Fine-Tuning Difficult"

> **Session:** Morning · Block B · 10 minutes  
> **Role of this segment:** Orient students in the historical arc *before* we go mechanistic in 1.B.2. No equations here — just the conceptual storyline and the cast of characters.

---

## The Central Bet

Every method in this block makes the same wager:

> *A safe model is one that stays safe even under attack.*

The attacker is free to try. The model just makes success expensive — too slow, too costly, too geometrically hostile. Resistance is passive. The model does not detect the attack; it simply does not cooperate.

This stands in contrast to Block 2 (this afternoon), where the model *lets* the attacker win — and then collapses around them.

---

## The Genealogy at a Glance

```
2022 ──────────────────────────────────────────────────────────────── 2025

MLAC                                                                   LoX
(BERT-era proof of concept)                              (training-free, SVD)
  │
  ├─── NTL (domain anti-adaptation)
  │
  ├─── TAR (entropy outer loop, scales to LLMs)
  │      └─── SOPHON (non-fine-tunability + MAML)
  │
  ├─── Vaccine (adversarial perturbation on embeddings)
  │      ├─── T-Vaccine (layer-targeted perturbation)
  │      ├─── RepNoise (MLP-level noise + harmful loss max)
  │      ├─── Booster (gradient attenuation)
  │      └─── Antidote / E.T. / LAPT / ILU / Circuit Breakers
  │
  └─── Condition Number (Hessian curvature as resistance metric)
         └─── LoX (extrapolate safety subspace, no training needed)
```

*Three families, one philosophy. Let's walk the arc.*

---

## Act I — The Proof of Concept (2022): MLAC

**Self-Destructing Models (Mackraz et al., 2022)** is where it starts. The context is not LLMs — it's BERT-era classifiers and a very concrete fear: someone downloads a pretrained model and fine-tunes it to extract demographic information or build a targeted weapon system.

The proposed answer is **Meta-Learned Adversarial Censoring (MLAC)**: simulate the attacker's fine-tuning step inside training (inner loop), then push the model's weights to *maximise* the attacker's loss (outer loop). The model learns not to learn the harmful task.

MLAC is imperfect — it works on small classifiers, uses approximate second-order gradients, and hadn't been tested on generative models. But it **coins the vocabulary**: task blocking, self-destructing models, bi-level immunisation. Every paper in this block is either citing MLAC or rediscovering it.

---

## Act II — The LLM Problem Changes the Stakes (2023–2024)

Two things happen around 2023 that make MLAC-style thinking urgent for LLMs:

1. **Shallow alignment** is discovered. Safety training in LLMs mostly shifts the distribution over the *first few tokens* (refusal prefixes like "I cannot…"). The deep generative capability for harmful content is never removed. This means safety is a thin shell — and fine-tuning punctures it with as few as **10–100 samples**.

2. **Harmful Embedding Drift (HED)** is named. When a fine-tuned model starts producing harmful content, the residual stream representations of harmful prompts have visibly drifted from where the aligned model placed them. The representation space tells the whole story.

These two observations split the field into two mechanistic camps:

- **Weight-space camp**: Make the loss landscape itself hostile to harmful optimisation.
- **Representation-space camp**: Anchor the residual stream so that harmful perturbations cannot take hold.

---

## Act III — The Weight-Space Family

### NTL and SOPHON — Anti-Domain Adaptation

**Non-Transferable Learning (NTL)** reframes immunisation through the lens of domain adaptation — but inverted. Instead of helping a model transfer to a new domain, NTL actively places model parameters at a local minimum *with respect to* the harmful domain. Transfer is blocked by design.

**SOPHON** extends NTL by explicitly adding a non-fine-tunability objective, approximated via MAML (model-agnostic meta-learning). It's the most explicit formalisation of what "learning not to learn" means in gradient space.

### TAR — Scaling to LLMs

**Tampering Attack Resistance (TAR)** revisits MLAC's bi-level structure but makes a key change in the outer loop: instead of maximising cross-entropy loss (which can be gamed by the model producing arbitrary tokens), it **maximises the entropy of the harmful output distribution**. A model that produces random, incoherent outputs under harmful fine-tuning is harder to recover from. TAR is the first method to make this class of ideas credible at the scale of modern LLMs.

### Condition Number and LoX — The Geometric Crystallisation

The most recent weight-space contributions ask a sharper question: *what property of the loss landscape, precisely, makes fine-tuning fast or slow?*

The answer: the **condition number** of the Hessian — the ratio of the largest to smallest eigenvalue. A high condition number means the landscape is a narrow valley: the optimiser races along one direction while barely moving along others, making convergence erratic and slow. The **Condition Number paper** operationalises this by adding differentiable regularisers that inflate the harmful task's condition number while keeping the benign task's well-conditioned.

**LoX (Low-rank Extrapolation)** takes this to its logical limit: *no training needed*. After standard safety alignment, LoX identifies the safety-critical low-rank subspace via SVD of the alignment weight delta, then **extrapolates further along that subspace**. The model moves from the narrow valley of safety into a flatter, more robust region — harder to push out of, without any additional learning.

---

## Act IV — The Representation-Space Family

### Vaccine and Its Progeny

**Vaccine (Huang et al., 2024)** operationalises HED directly. During alignment, it adds adversarial perturbations to the residual stream embeddings — simulating the drift a harmful fine-tuner would cause — and trains the model to resist that drift. The model learns to *stay put* in representation space even under pressure.

**T-Vaccine** inherits the Vaccine idea but asks: *are all layers equally important?* No. Layers with high gradient norm on harmful data are the safety-critical ones. T-Vaccine targets perturbations only there, dramatically reducing memory overhead while improving resistance. This makes harmful-fine-tuning defence practical on consumer GPUs (RTX 4090).

**RepNoise** approaches the same problem from the angle of harmful loss maximisation: instead of just perturbing, it actively maximises the loss on harmful data at the MLP level, while Gaussian noise is injected to reduce the divergence between harmful response representations.

**Booster** focuses on the *velocity* of harmful convergence: by attenuating the gradients that point toward harmful behaviours, it slows the attacker down without blocking them completely.

### Circuit Breakers, Antidote, E.T., LAPT, ILU

The most recent representation-space methods push in a more adversarial direction: don't just resist drift, actively *reroute* it. **Circuit Breakers** push unsafe representations to their orthogonal counterparts. **E.T.** (Ethical Treatment) — developed as part of the research behind this tutorial — generalises this to inference-time intervention attacks (ITI), where an adversary steers the residual stream at inference rather than via fine-tuning. **LAPT** and **ILU** continue this line of work with refined objectives and expanded attack surfaces.

---

## The Arc in One Sentence

From a BERT-era proof of concept (MLAC, 2022) to a rich ecosystem of weight-space and representation-space defences (2024–2025), the field has spent three years learning how to formalise, measure, and operationalise one idea: *put the model somewhere the attacker cannot easily reach*.

---

## What's Coming in 1.B.2

Now that we've seen the cast and the narrative, we'll go mechanistic: bi-level optimisation, adversarial perturbation losses, the Hessian condition number trick, SVD extrapolation. The math is the story told more precisely.

---

*References: Mackraz et al. (2022); Peng et al. (2024) [VAA/Safety Basin]; Huang et al. (2024) [Vaccine]; Liu et al. (2025) [T-Vaccine]; Rosati et al. (2024) [RepNoise]; TAR; NTL; SOPHON; Booster; Circuit Breakers; LoX; Condition Number; E.T. / LAPT.*
