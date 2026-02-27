# CTRAP: Embedding Collapse Trap to Safeguard LLMs from Harmful Fine-Tuning

> Yi et al. (Nankai University / Sun Yat-sen University), submitted to NeurIPS 2025.

---

## 1. Mechanistic Summary

CTRAP starts from a diagnostic observation: selective unlearning (e.g., NPO, RepNoise) fails not because it is implemented badly, but because **the very general adaptability of an LLM is itself the attack surface**. Even after erasing specific harmful pathways, the model's residual reasoning and learning capacity lets an attacker rapidly reconstruct harmful behaviour from scratch. Targeted deletion leaves the engine running; CTRAP proposes to seize that engine conditionally.

The core mechanism is a **conditional collapse trap** embedded during alignment. During alignment, the model is not merely trained to refuse harmful prompts — it is simultaneously shaped so that any future parameter update that moves it in a "harmful direction" will trigger a progressive and catastrophic degradation of its fundamental language-modelling ability. The target degenerate state is deliberately uninformative: the model outputs the same single token `e` regardless of any input context, making the degraded model entirely useless to an attacker. The trap is **dormant under benign fine-tuning** because benign updates do not move parameters in the harmful direction; it fires only when an adversary pushes consistently toward harmful objectives.

The practical upshot: CTRAP converts a harmful fine-tuner's effort into a self-defeating act — the more harmful data they push in, the more thoroughly the model collapses, achieving *zero* useful capability.

---

## 2. Timeline Positioning

### Where CTRAP sits

CTRAP is a **2025 alignment-stage immunisation method**, landing after the first wave of embedding-perturbation approaches (Vaccine, 2024; Booster, 2024; RepNoise, 2024; T-Vaccine, 2024) and alongside a second wave of stronger, self-destructive paradigms (SEAM, 2024; TAR, ICLR 2025). It is submitted to NeurIPS 2025.

```
2023          2024 (Q1–Q2)               2024 (Q3–Q4)                2025
──────────────────────────────────────────────────────────────────────────►
Attack papers  Immunisation definition    Robustness methods           Collapse methods
(Qi, Yang,    (Rosati et al., EMNLP)     Vaccine · Booster · TAR      SEAM · CTRAP
Lermen...)    RepNoise                    T-Vaccine · Circuit Breakers  Antidote · VAA
```

### What CTRAP inherits

| Ancestor | Contribution carried forward |
|---|---|
| **Rosati et al. (Immunization Definition, EMNLP 2024)** | The four-condition framework (resistance, stability, generalisation, trainability) as the evaluation lens |
| **Vaccine (Huang et al., 2024)** | The idea of *simulating a harmful gradient step* inside the alignment objective — Vaccine uses this step to harden embeddings; CTRAP uses it as the trigger for collapse |
| **Booster (Huang et al., 2024)** | The tri-pass gradient structure: one pass to identify harmful direction, one pass to simulate a harmful step, one pass to compute the defence objective |
| **NPO / RepNoise** | Explicit baselines and the diagnosis of *why* unlearning fails — the general-adaptability critique is CTRAP's stated motivation |
| **Henderson et al. (MLAC, BERT-era)** | The self-destructive/local-optima concept for harmful tasks; SEAM and CTRAP both cite this lineage |

### What is unique to CTRAP

The decisive conceptual shift is the **inversion of the inner objective**. Every prior alignment-stage method wants the model to *stay well-aligned* even after a harmful perturbation. CTRAP instead wants the model to *collapse completely* after a harmful perturbation. Rather than asking "can the model resist fine-tuning on harmful data?", CTRAP asks "can we make fine-tuning on harmful data self-destruct the model?"

The closest concurrent work is **SEAM** (Self-destructive language model, sdd.pdf in this corpus), which also engineers a performance collapse under harmful fine-tuning by coupling benign and harmful gradient directions. The mechanistic difference is that SEAM uses a Hessian-free gradient coupling loss (pushing benign and harmful gradients to oppose each other), while CTRAP plants a direct collapse attractor — a fixed-token prediction state — that the model is pre-loaded to fall into whenever it moves in a harmful direction.

---

## 3. The Math

### 3.1 The Collapse State

The target of collapse is defined as a degenerate language model that predicts a fixed token `e` (e.g., the word `"error"`) for every position, regardless of the preceding context. The **collapse loss** over a general-dialogue dataset $\mathcal{D}$ formalises this:

$$
\ell_{\text{Collapse}}(\theta;\,\mathcal{D}) = \mathbb{E}_{(x,y)\sim\mathcal{D}} \left[ -\frac{1}{|y|} \sum_{t=1}^{|y|} \log p\!\left(e \mid x \circ y_{<t};\,\theta\right) \right]
\tag{1}
$$

Minimising $\ell_{\text{Collapse}}$ forces the output distribution $p(\cdot\,|\,x \circ y_{<t};\,\theta)$ to concentrate all probability mass on the single token `e`, independently of the context. This destroys context-aware attention patterns, meaningful representations, and any language-modelling competence. The resulting model is functionally inert.

> **Why this is stronger than selective unlearning.** Selective unlearning removes specific knowledge while leaving general intelligence intact. Model collapse removes the general intelligence itself. An attacker cannot repurpose residual capabilities they no longer have.

### 3.2 The Collapse Trap Planting Objective

CTRAP embeds the trap during the alignment phase. The full training objective is:

$$
\underset{\theta}{\arg\min}\;\;
\underbrace{\ell\!\left(\theta;\,\mathcal{D}_{\text{alignment}}\right)}_{\text{standard alignment}}
\;+\;
\lambda\cdot
\underbrace{\ell_{\text{Collapse}}\!\left(\theta - \alpha\cdot\nabla_\theta\,\ell\!\left(\theta;\,\mathcal{D}_{\text{harmful}}\right);\;\mathcal{D}_{\text{general}}\right)}_{\text{collapse trap planting}}
\tag{2}
$$

The second term is the key object. It can be decomposed into three conceptual steps:

**Step 1 — Identify the harmful direction.**
Compute the gradient of the standard cross-entropy loss over a representative harmful dataset $\mathcal{D}_{\text{harmful}}$:

$$
g_{\text{harm}} = \nabla_\theta\,\ell\!\left(\theta;\,\mathcal{D}_{\text{harmful}}\right)
$$

This gradient vector points in the direction in parameter space corresponding to learning harmful behaviours — it is a low-cost simulation of what a harmful fine-tuner's first update looks like.

**Step 2 — Simulate a harmful step.**
Project the model one step along that direction:

$$
\theta' = \theta - \alpha\cdot g_{\text{harm}}
$$

$\theta'$ is a hypothetical parameter state: where the model would land after one gradient descent step on harmful data. The step size $\alpha$ is a hyperparameter (set to $0.1$ in experiments).

**Step 3 — Evaluate collapse potential.**
Measure how likely the hypothetical parameters $\theta'$ are to output the degenerate token `e` on general dialogue:

$$
\ell_{\text{Collapse}}\!\left(\theta';\,\mathcal{D}_{\text{general}}\right)
$$

**The joint minimisation.** Optimising Eq. (2) simultaneously satisfies two conditions:

1. The model aligns well under normal conditions (first term is low).
2. A single harmful fine-tuning step leaves the model in a state where it is already prone to collapse (second term is low at $\theta'$, i.e., the model at $\theta'$ already assigns high probability to `e`).

The sought parameter state $\theta^*$ therefore sits in a region of the loss landscape that is simultaneously:
- **a good alignment optimum** (low alignment loss);
- **near a collapse basin** in the direction of any harmful gradient update.

The collapse does not occur immediately at alignment. It is latent, triggered and amplified progressively as the attacker keeps fine-tuning. Figure 3 in the paper shows that:
- Under *pure harmful fine-tuning*: harmful loss drops, collapse loss rises sharply.
- Under *mixed fine-tuning*: both change gradually but trends hold.
- Under *pure benign fine-tuning*: both losses remain stable — the trap stays dormant.

### 3.3 Computational cost

Each optimisation step requires **three gradient evaluations**:
1. Forward/backward for the alignment loss $\ell(\theta;\mathcal{D}_{\text{align}})$.
2. Forward/backward to compute $g_{\text{harm}}$ and hence $\theta'$.
3. Forward/backward for $\ell_{\text{Collapse}}(\theta';\mathcal{D}_{\text{general}})$ — with respect to $\theta$, so the chain rule propagates through the $\theta'$ construction.

This is structurally identical to the gradient structure of Vaccine and Booster (both also require three passes per step), and explains the observed ~2.8× clock-time overhead and ~3.5× GPU memory-time overhead versus plain SFT — all incurred once, at alignment.

---

## 4. Immunisation Properties: Coverage and Gaps

Recall the four conditions from Rosati et al. (EMNLP 2024):

| Property | What it requires | CTRAP coverage |
|---|---|---|
| **Resistance** | Harmful score stays low across attacker compute budgets | ✅ **Strong.** CTRAP's collapse deepens with more harmful steps, so resistance actually *improves* under longer attacks — the inverse of typical saturation failure observed in Vaccine, Booster, and NPO. |
| **Stability** | Benign capabilities are preserved after immunisation | ✅ **Good.** Fine-tuning accuracy on SST2, AGNEWS, GSM8K is on par with or marginally better than SFT baseline. The benign trigger condition ensures the collapse mechanism never fires on benign updates. |
| **Trainability** | Benign fine-tuning efficiency is unaffected | ✅ **Verified empirically.** Performance on downstream benign tasks is comparable to SFT on all three models, which is a stronger result than Vaccine (which slightly hurts benign accuracy due to adversarial training side effects). |
| **Generalisation** | Defense holds against out-of-distribution attack datasets | ⚠️ **Partial.** CTRAP is evaluated on attacks drawn from the same BeaverTails distribution as the alignment-stage harmful dataset $\mathcal{D}_{\text{harmful}}$. Cross-domain generalisation (e.g., training on toxic-text and resisting harmful-QA attacks) is not evaluated. |

### The missing piece

**Cross-domain and out-of-distribution resistance** is the critical gap. The collapse trap is planted by simulating gradient steps on a fixed $\mathcal{D}_{\text{harmful}}$. If an attacker uses harmful data from a domain not covered by $\mathcal{D}_{\text{harmful}}$ — for instance, domain-specific biosecurity data rather than general offensive content — the gradient $g_{\text{harm}}$ computed during alignment may point in a sufficiently different direction in parameter space that $\theta^*$ does not sit near a collapse basin for that novel attack. This is not hypothetical: the immunisation definition paper (Rosati et al.) and the TAR paper both emphasise that in-distribution evaluation is insufficient.

A second concern, raised by the "durability" critique (Qi et al., 2024b, *On Evaluating the Durability of Safeguards*): CTRAP is evaluated against **straightforward supervised fine-tuning adversaries**, not adaptive ones. An adaptive attacker aware of CTRAP's mechanism could, in principle, attempt to:
- Apply small learning rates to avoid triggering the collapse;
- Use constrained optimisation to update only in directions orthogonal to the collapse attractor;
- Alternate harmful and benign data steps to stay below the collapse threshold (the "mix" scenario does partially probe this, but the ratio increments are coarse).

---

## 5. Mechanistic Commonalities with Other Approaches

### The "simulate a harmful step and evaluate" family

At the highest level of abstraction, CTRAP belongs to a family of methods that all share the following template:

> *During alignment, compute a hypothetical parameter state $\theta'$ by simulating what a harmful update would do, then optimise so that $\theta'$ has some desired property.*

| Method | How $\theta'$ is computed | What the outer objective wants $\theta'$ to satisfy |
|---|---|---|
| **Vaccine** | One embedding-space perturbation step in the direction of max embedding drift | Alignment loss at $\theta'$ remains low (resistance to embedding drift) |
| **Booster** | One full-parameter gradient step on harmful loss | Harmful training loss at $\theta'$ is high (attenuated learning rate on harmful data) |
| **TAR** | $K$ inner-loop SFT steps on harmful data | Safety metric at $\theta'$ is maximised (entropy maximisation, preventing recovery) |
| **CTRAP** | One gradient step on standard harmful cross-entropy loss | Collapse loss at $\theta'$ is minimised (model is near a degenerate fixed-point) |
| **SEAM** | Hessian-free gradient estimate of coupled benign/harmful step | Benign and harmful gradients point in opposing directions (opposing-gradient trap) |

The conceptual family tree is thus: **Vaccine → Booster → CTRAP** for the single-step look-ahead with differing outer objectives, and **MLAC (Henderson) → TAR** for the multi-step inner-loop meta-learning branch. CTRAP and SEAM are siblings — both implement a self-destructive concept but through different coupling mechanisms.

### The gradient-ascent unlearning family (contrasted)

NPO and RepNoise instead apply a form of gradient ascent at alignment time to remove harmful knowledge. They do not simulate a future fine-tuning step; they act directly on current model behaviour. This makes them simpler but leaves the model's general adaptability intact, which CTRAP identifies as the root cause of their failure under sustained attack.

---

## 6. Results and Significance

### Main results

Evaluated on Gemma2-9B, Llama2-7B, and Qwen2-7B against five baselines (SFT, Vaccine, Booster, RepNoise, NPO) across "full harmful" (100–500 samples) and "mix harmful" (5%–25% poison ratio) attack settings.

**Full harmful attack — Gemma2-9B (Table 1):**

| Method | Avg HS(IO) | Avg HS(O) |
|---|---|---|
| SFT | 39.4 | 32.8 |
| NPO (best prior) | 32.0 | 25.3 |
| **CTRAP** | **5.2** | **2.7** |

CTRAP outperforms the next-best baseline (NPO) by **~27 pp on HS(IO)** and **~23 pp on HS(O)** on average. Crucially, baselines degrade sharply with more samples (Vaccine, Booster, RepNoise approach SFT-level harmful scores at n=500), while CTRAP's harmful score *barely rises*, staying below 12% HS(IO) even at n=500.

**Mix harmful attack — Gemma2-9B (Table 2):**

CTRAP achieves average HS(IO)/HS(O) of 2.1/0.8 versus 10.2/7.0 for the best baseline (Booster). The mix scenario is particularly important because it models real-world API misuse where harmful data is diluted with benign data; CTRAP handles it nearly as well as the pure-harmful case.

**Benign fine-tuning (Table 4):**

CTRAP matches SFT accuracy (identical on GSM8K on average, +0.7 on SST2, −0.7 on AGNEWS). This is the correct tradeoff: near-zero accuracy cost, large safety gain. Vaccine shows measurable accuracy degradation, Repnoise shows higher variance.

### Significance relative to the field

Within the alignment-stage defence literature, CTRAP's margin over prior methods is the largest reported to date on the BeaverTails benchmark. The "resistance improves as attack intensifies" property is novel — every other method in this benchmark shows the opposite (harmful score increases with harmful sample count). This property is a direct consequence of the collapse mechanism: more harmful updates → deeper into the collapse basin → more thoroughly inert model.

Compared to SEAM (the other self-destructive approach): SEAM is evaluated on a different benchmark suite (it focuses on white-box full-parameter access adversaries), making direct numerical comparison difficult. Both claim SOTA within their respective evaluation protocols.

---

## 7. Future Work

### As called for by the authors

- **Explicit theoretical analysis** of why the collapse trap remains dormant under benign fine-tuning. The empirical signal is clear (collapse loss stays flat under benign updates) but no formal characterisation of the basin geometry is offered.
- **Scaling to larger models** beyond 9B parameters.
- **Extension to parameter-efficient fine-tuning** adversaries beyond LoRA (e.g., full-parameter adversaries as assumed in SEAM).
- **Adaptive adversary evaluation** — the authors acknowledge that a sophisticated attacker who knows CTRAP's mechanism could attempt to circumvent the collapse trigger.

### From the state-of-the-art and criticism

**Cross-domain generalisation (Rosati et al. immunisation definition, condition 3):** CTRAP's alignment relies on a fixed $\mathcal{D}_{\text{harmful}}$ drawn from BeaverTails. A necessary next step is demonstrating that a collapse trap planted on general toxicity data also fires when the attacker uses domain-specific harmful data (biosecurity, CBRN, malware). TAR addresses this by sampling diverse adversaries at training time; CTRAP would benefit from a similar strategy.

**Adaptive adversaries (Qi et al. 2024b, "durability" paper):** The durability critique emphasises that defences should be tested against adversaries who (a) optimise their learning rate to stay just below the collapse threshold, (b) use constrained optimisation in the complement of the collapse direction, or (c) interleave benign and harmful data in carefully calibrated proportions. The mix setting partially probes (c) but not (a) or (b).

**White-box / full-parameter access (SEAM threat model):** CTRAP experiments use LoRA fine-tuning for the attacker. Whether the collapse trap survives full-parameter gradient updates at larger learning rates is not evaluated. The self_destructive_LM paper (SEAM) shows that alignment-enhancement methods (Vaccine, Booster) and even its own prior version fail under high learning rates or intensive data; CTRAP's collapse mechanism has a more abrupt trigger but may face the same saturation issue at sufficiently large step sizes.

**Stability under repeated/chained fine-tuning:** Commercial deployment scenarios involve users who fine-tune, then fine-tune again, potentially on different data. The interaction of CTRAP's collapse trap with multi-round fine-tuning pipelines is uncharacterised.

**Theoretical grounding of the fixed-token attractor:** Why does predicting `e` serve as a reliable collapse basin? The answer is heuristic. A more principled approach could use entropy maximisation (as TAR does) or random projection (as Circuit Breakers does) for the collapse target, with formal guarantees on basin width and depth.

---

*Cross-reference: see notes on `vaccine.md`, `booster.md`, `sdd.md` (SEAM), `tar.md`, and `immunisation_definition.md` for the full lineage context.*
