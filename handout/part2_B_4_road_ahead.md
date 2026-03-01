# Part 2.B.4 — The Road Ahead: Promising Directions

> **Session:** Afternoon · Part B · 5 minutes  
> **Role of this segment:** Close the tutorial not with a list of open problems — you already have that — but with genuine signals of where the field is moving and why there is reason for optimism. Five minutes. No equations. Leave them with momentum.

---

## The Shift That Is Already Happening

Something has quietly changed in the last year of immunisation research, and it is worth naming before the individual directions.

The field began — Block 1, 2022–2023 — with a largely behavioural framing: make the model *behave* as if fine-tuning is hard. Perturb embeddings. Raise the loss on harmful steps. Flatten the loss landscape. All of these work at the level of what the model *does* under adversarial pressure. The adversarial Perspective paper this afternoon showed us the limit of that approach: behaviour can be patched, but the knowledge underneath it is still there. The vault is locked but not empty.

The shift that is emerging — and you can see it clearly in the most recent papers — is from behaviour-centric to **representation-centric** immunisation. The question is no longer only *can we make fine-tuning expensive?* but *can we change what the model fundamentally is, at the level of its internal geometry?* That is a harder question and a more interesting one.

Four directions are showing early promise.

---

## Direction 1 — Invariant Unlearning

The deepest critique of current unlearning is that it teaches a model to suppress a capability under one distribution of probes, but the capability persists under others. The analogy from causality is direct: standard unlearning learns a spurious correlation between "this is a harmful query" and "refuse" — and spurious correlations break under distribution shift.

**Invariant LLM Unlearning (ILU)** applies the machinery of Invariant Risk Minimisation to this problem. Instead of minimising loss on a single harmful dataset, it requires that the unlearned state remain *stationary* across multiple environments — different phrasings, different fine-tuning datasets, different attack vectors. The stability penalty forces the model to find a representation of safety that is not specific to any one probe, but holds across the causal structure of the problem.

Early results are encouraging: models trained with invariant objectives show substantially better resistance to fine-tuning attacks that use out-of-distribution harmful data — precisely the failure mode the Adversarial Perspective paper identified. The conceptual connection to the obfuscation problem is direct: invariance over environments is exactly the property that distinguishes erasure from suppression.

---

## Direction 2 — Adversarial Hypernetworks

You just ran AntiDote in the challenge. But step back and see what the hypernetwork idea actually represents as a research direction, beyond this one paper.

The fundamental bottleneck in every bi-level immunisation method — TAR, Booster, MAML-based approaches — is that the simulated inner adversary is too weak. First-order gradient approximations underestimate the true worst-case attacker. The training signal for the defender is therefore biased: it is hardened against a strawman.

The hypernetwork approach breaks this bottleneck by replacing the inner loop with a *differentiable adversary that can be trained jointly with the defender in real time*. The cost of generating an attack becomes a constant-size forward pass, not a function of the model scale. This unlocks two things: first, the arms race can actually happen at training time rather than being approximated; second, the gradient signal flowing to the defender is unbiased, which is why AntiDote's decoupled loss achieves better utility preservation than methods with coupled objectives.

The direction this opens is broad. A hypernetwork adversary is a general-purpose component — it could be applied to representation-engineering attacks, to RL-based subversion, to any attack type where you can define a differentiable loss. That generality is what makes it a *research direction* rather than just one paper's contribution.

---

## Direction 3 — Architecture-Level Locking (ArchLock)

Both Block 1 and Block 2 operate at the level of weights and training dynamics. ArchLock asks a different question: what if the architecture itself — not just the weights — encodes a structural constraint that makes harmful adaptation geometrically impossible?

The intuition is that standard fine-tuning works by exploiting the model's general trainability. LoRA, full fine-tuning, gradient ascent on harmful data — all of these assume that the parameter space is sufficiently unconstrained that a harmful trajectory exists within reach. ArchLock's approach is to introduce architectural modifications that partition the parameter space: certain subspaces are rendered structurally inaccessible to gradient updates unless a cryptographic or structural key is present.

This is early-stage work, but it is conceptually important because it represents the first serious attempt to move immunisation from a training-time property to an architecture-time property. The implications for the pre-alignment base model problem — the shared Achilles' heel we identified this morning — are significant. If the base model architecture itself encodes structural resistance, the attacker who starts from the pre-alignment checkpoint still faces the architectural constraint.

---

## Direction 4 — Better Evaluation as a Research Contribution

This one is less glamorous than the others, but the community has begun to take it seriously as a first-class research problem — not just an afterthought.

The Immunisation Definition paper established the four pillars. What is still missing is a *standardised, adversarially stress-tested evaluation suite* that the field agrees on — analogous to what WMDP did for knowledge restriction and what HarmBench did for jailbreak robustness, but specifically designed for immunisation's threat model: open-weight release, white-box access, adaptive fine-tuning with varied hyperparameters, and measurement of both resistance and utility without conflating them.

TAR's 26-adversary evaluation is the closest thing that exists. Building a community benchmark around that philosophy — diverse attacks, white-box stress-testing, both SFT and RL attack types — would immediately make every paper in the field more comparable and every claim more credible. The field needs this before it can make confident statements about what has actually been solved.

---

## The Closing Observation

Here is the thing about all four of these directions: none of them requires inventing a fundamentally new discipline. They require applying rigour — from causal inference, from generative modelling, from cryptography, from adversarial ML evaluation — to a problem that the LLM safety community has defined clearly and cares about deeply.

The problem is well-posed. The evaluation criteria exist. The threat model is concrete. The code is open. What is missing is the next generation of contributions.

You have just spent a day learning the mechanistic foundations of this field in detail. That puts you in a better position than most to make one of those contributions.

That is the road ahead. Go build something on it.
