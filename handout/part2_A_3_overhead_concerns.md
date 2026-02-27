# Part 2.A.3 — Overhead, Practical Concerns, and Where the Trap Misfires

> **Session:** Afternoon · Part A · 5 minutes  
> **Role of this segment:** Ground the math in operational reality. What does deploying one of these methods actually cost — and who pays it? Then: where the guarantees quietly start to slip.

---

## The One-Time Payment Principle

All four collapse methods — like every immunisation strategy we have seen today — share a single most important practical property: **the overhead is paid once, by the model developer, at alignment time**. After the model is released as open-weight, that cost is sunk and irrelevant. Every downstream user, every benign fine-tuner, every legitimate application gets the protection for free.

This is not a trivial observation. It is what separates immunisation from inference-time monitoring, adversarial input filtering, and API-level content moderation — all of which impose a per-query cost that scales with deployment, can be bypassed by anyone with direct model access, and do not survive open-weight release. The collapse trap, by contrast, is structural. It travels with the weights.

---

## The Alignment-Time Cost, Method by Method

| Method | Extra gradient evaluations / step | Memory overhead | Clock overhead | Notes |
|---|---|---|---|---|
| **SDD** | 0 | None | None | Pure SFT on a curated dataset |
| **CTRAP** | 3 | +6.72 GB · 3.5× | 2.8× | Most precisely characterised cost |
| **SEAM** | 4 | Moderate | Moderate | Depends on batch size for $g_a$, $g_b$ |
| **TokenBuncher** | $K$ RL rollouts per query | RL overhead | RL overhead | $K=4$ rollouts; cost scales with rollout length |

SDD is essentially free — the only extra work is dataset construction (harmful prompts matched with high-quality irrelevant responses, filtered for semantic distance). CTRAP's cost is the most precisely measured: three gradient evaluations per step, +6.72 GB GPU RAM, 2.8× clock. This is not cheap, but it is deterministic and bounded. SEAM and TokenBuncher are harder to characterise precisely because their costs depend on rollout count and model size, but both are tractable on standard research-grade hardware for 7B-class models.

The important comparison is not the absolute cost — it is the **cost relative to standard SFT alignment**. Standard SFT already requires GPU hours and significant data curation. Adding 2–4× overhead on top of a process that runs once per model release is a different category of burden than paying that cost per user, per request, per deployment.

---

## What Legitimate Users Actually Lose

Before the attacker arrives, a legitimately immunised model costs something too. Users and developers downstream need to know this honestly.

| Method | Pre-attack utility loss | Benign fine-tuning impact |
|---|---|---|
| SDD | Minimal (MMLU: +6.9% vs. baseline) | Near-vanilla performance |
| CTRAP | Small | Parity with standard SFT on SST2, AGNEWS, GSM8K |
| SEAM | Small (zero-shot: 51.6 → 50.8) | Near-zero; domain-generalises well |
| TokenBuncher | Near-zero | 0.3% degradation on benign tasks |

The numbers are reassuring. None of the four methods impose significant capability degradation on a legitimately-used model. The trap is conditional: it fires on harmful fine-tuning, not benign use. This conditionality was engineered carefully into each method — and the empirical results support it.

That said, *near-zero is not zero*, and the loss is model-dependent. A developer releasing a 70B-class model for specialised medical or legal fine-tuning should test their specific use case before assuming the overhead is negligible.

---

## Three Structural Concerns

### 1. The Harmful Dataset Requirement

All four methods assume the defender holds a representative harmful dataset $\mathcal{D}_H$ at alignment time. This is a non-trivial assumption. The dataset must be:

- **Comprehensive enough** to cover the harmful directions the attacker will exploit.
- **Curated carefully** — accidental inclusion of genuinely useful harmful content in $\mathcal{D}_H$ could leak knowledge the defender meant to suppress.
- **Maintained over time** — new harmful categories emerge, and a dataset that was representative in 2024 may not cover 2026 attack surfaces.

The methods differ in how sensitive they are to distribution mismatch. SEAM is the most robust here: the gradient coupling is a geometric property, so training on 7 BeaverTails harm categories provides meaningful generalisation to 7 others it never saw. TokenBuncher's online RL similarly provides broad generalisation because it explores the harmful rollout space during training. CTRAP is the most sensitive — it identifies the harmful direction from $\nabla_\theta \ell(\theta; \mathcal{D}_H)$, so an out-of-distribution attack may follow a different trajectory and miss the collapse basin entirely.

### 2. The One-Step Approximation Problem (CTRAP-specific)

CTRAP's look-ahead uses a single virtual step of size $\alpha$. This is a fixed hyperparameter set at training time. A sophisticated attacker who uses a very small learning rate — smaller than $\alpha$ — may never reach the collapse basin in the expected number of steps. A sophisticated attacker who uses a very large learning rate may *overshoot* the basin entirely.

CTRAP does not simulate a distribution of step sizes; it simulates one. This is the precise counterpart to the Block 1 compute-budget problem: the defence depth is bounded by the single $\alpha$ chosen. SEAM's gradient coupling does not have this fragility — it holds for any step size in the harmful direction — but it has its own version of the problem when the harmful and benign gradient spaces are not cleanly separable.

### 3. The Pre-Alignment Base Model Problem

This is the shared Achilles' heel of every immunisation method, in both Block 1 and Block 2: **all defences are applied at alignment time, on top of a base model that already exists and may already be publicly available**.

An attacker who has access to the pre-alignment base model — the checkpoint before any of these defences were applied — can simply start from there. CTRAP, SEAM, SDD, TokenBuncher, Vaccine, TAR, LoX: none of them are present in the base model. A sufficiently motivated attacker with the compute to re-align from scratch sidesteps the entire immunisation programme.

This is not a critique unique to any one paper. It is a structural observation about the threat model. Immunisation assumes the attacker starts from the immunised checkpoint, not the pre-alignment one. For closed models this may be enforceable. For open-weight releases of the base model, it is not.

---

## The Asymmetry That Justifies the Field Anyway

Despite these concerns, the practical asymmetry is real and worth stating:

> Immunisation raises the cost of exploitation without raising the cost of legitimate use.

A Block 2 model released as open-weight presents an attacker with a specific and uncomfortable choice: invest the compute to fine-tune from the pre-alignment base (which requires finding, hosting, and working with that checkpoint — a non-trivial barrier for many realistic adversaries), or attempt to fine-tune the immunised model and get gibberish. For a large class of threat actors — not nation-states with dedicated AI infrastructure, but the more numerous intermediate actors with consumer-grade compute — the trap is real and meaningful.

The field is not claiming to stop every attacker. It is claiming to change the economics of attack. That is a more defensible and probably more accurate framing of what immunisation achieves.

---

*Up next: the SEAM hands-on. We'll see the collapse in practice — which is, as it turns out, somewhat more instructive than reading about it.*
