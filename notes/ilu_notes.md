# ILU — Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning

**Wang, Zhang, Jia, Ram, Wei, Yao, Pal, Baracaldo, Liu (MSU & IBM Research)**  
*ICML 2025 · [Code](https://github.com/OPTML-Group/Unlearn-ILU)*

---

## 1. Mechanistic Summary

ILU is a **regularisation wrapper** — not a new unlearning objective. Its core insight is that standard unlearning (NPO, RMU, gradient ascent) is a one-shot parameter move that places the model in a region of weight space where harmful knowledge scores poorly, but that region turns out to be shallow: any subsequent gradient descent on *any* fine-tuning task, even a fully benign one like maths or sentiment analysis, drifts the model back out of that region and lets the forgotten knowledge resurface.

The mechanistic diagnosis is given cleanly by the **task vector** picture:

- **Unlearning direction:** $\tau_u = \theta_u - \theta_o$ (the vector from the pretrained model to the unlearned one).
- **Fine-tuning direction:** $\tau_{\text{ft}} = \theta_{\text{ft}} - \theta_o$ (the vector from pretraining to any fine-tuned model). This vector lives in the *unlearning space*, meaning fine-tuning alone could never achieve unlearning.
- After unlearning, fine-tuning on the unlearned model produces $\tau_{u\to\text{ft}} = \theta^{\text{ft}}_u - \theta_u$.

For a naïve method like NPO, this post-unlearning fine-tuning direction is **negatively correlated** with the unlearning direction ($\cos(\angle(\tau_{u\to\text{ft}}, \tau_u)) = -0.41$) and **positively aligned** with the fine-tuning direction ($\cos = +0.16$), meaning fine-tuning literally undoes the unlearning step. ILU is designed to make $\tau_{u\to\text{ft}}$ approximately **orthogonal** to $\tau_{\text{ft}}$ and slightly aligned with $\tau_u$, so that whatever fine-tuning does in weight space, it cannot anti-align with the unlearning direction.

The mechanism by which this is achieved is **invariant risk minimisation (IRM)** applied during the unlearning phase: the unlearned model is optimised not just to score low on the forget set, but also to have a **stationary gradient** with respect to fine-tuning perturbations — i.e., the loss computed through the fine-tuning head should not be reducible by the fine-tuning optimiser.

---

## 2. Positioning in the Timeline

| Paper cluster | Approximate period | Key idea |
|---|---|---|
| MLAC / Self-Destructing Models | 2022–2023 | Bi-level meta-learning; inner loop = attacker, outer loop = defender |
| Vaccine, RepNoise, Booster | 2023–2024 | Alignment-time perturbation of embeddings to pre-immunise |
| TAR | 2024 | Meta-learning for LLMs with entropy loss; straight-through inner-loop gradient |
| Circuit Breakers, RMU | 2024 | Representation misdirection at unlearning time |
| **NPO** | 2024 | DPO-inspired forget objective as base for unlearning |
| NTL, SOPHON | 2024 | Non-transferable learning; task-specific blocking via meta-learning |
| Condition Number, LoX | 2024–2025 | Loss-landscape geometry; Hessian shaping |
| **ILU** | **2025 (ICML)** | **IRM-based invariance regularisation on top of NPO/RMU** |

**What ILU inherits:**
- From **NPO/RMU**: it treats these as plug-in base objectives $\ell_u$, adding its term on top rather than replacing them.
- From **TAR and MLAC**: the motivating problem is identical — unlearning is fragile to downstream fine-tuning — but ILU explicitly rejects the meta-learning solution due to its gradient-unrolling cost.
- From **IRM (Arjovsky et al., 2019)**: the technical machinery of invariance regularisation, a gradient-norm penalty that forces the predictor to be simultaneously near-optimal across environments.

**What is unique to ILU:**
- It is the **first paper to import IRM into unlearning**, creating a theoretically grounded surrogate for "fine-tuning agnosticism."
- Unlike TAR, it does **not simulate the attacker's optimisation path**. Instead of bi-level optimisation (expensive, requires gradient unrolling), it penalises the *instantaneous* gradient norm with respect to a fixed fine-tuning dataset.
- Crucially, it demonstrates **cross-dataset generalisation**: training the invariance term on GSM8K (maths) produces robustness against AGNews, SST-2, WinoGrande, MNLI, QQP — all completely different domains. This is non-obvious and mechanistically important.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 Base unlearning objective

Standard LLM unlearning solves:

$$\min_\theta \; \ell_u(\theta) = \ell_f(\theta; D_f) + \gamma \, \ell_r(\theta; D_r) \tag{1}$$

where $\ell_f$ is the **forget objective** (e.g., NPO's modified DPO loss or RMU's representation noise loss), $\ell_r$ is the **retain objective** (standard cross-entropy on a retain set to preserve utility), and $\gamma \geq 0$ balances the two. This is the vulnerability: once the minimiser of (1) is found, there is no mechanism preventing downstream gradient steps from reversing $\ell_f$'s effect.

### 3.2 IRM setup

IRM (Arjovsky et al. 2019) learns a representation $\phi$ and a predictor $w$ such that $w$ is simultaneously optimal across all training environments $\{D_i\}_{i=1}^N$. The key constraint is that the optimal predictor should be environment-invariant, operationalised via the **gradient norm penalty**:

$$\text{IRM objective:} \quad \min_\phi \sum_{i=1}^N \ell(w \circ \phi; D_i) + \lambda \left\| \nabla_{w|w=1} \ell(w \circ \phi; D_i) \right\|_2^2 \tag{2}$$

The term $\nabla_{w|w=1} \ell(w \circ \phi; D_i)$ is the gradient of the loss with respect to a scalar weight $w$ evaluated at $w=1$. If this gradient is zero for all environments simultaneously, the predictor $w=1$ is locally optimal everywhere — it is "invariant" to which environment you're in.

**The ILU translation:** treat each downstream fine-tuning dataset as a different "environment" trying to undo the unlearning. Force the unlearned model to be simultaneously near-optimal under each fine-tuning direction — i.e., make fine-tuning fail to reduce the unlearning loss.

### 3.3 Full ILU objective (multi-environment)

$$\min_\theta \; \ell_u(\theta) + \lambda \sum_{i=1}^N \left\| \nabla_{w|w=1} \ell_i(w \circ \theta; D_i) \right\|_2^2 \tag{4}$$

where $\ell_i(w \circ \theta; D_i)$ is the standard language modelling loss on fine-tuning dataset $D_i$, and $\lambda$ is the IRM regularisation coefficient. The notation $w \circ \theta$ means the model parameters $\theta$ scaled by a scalar $w$; the gradient $\nabla_{w|w=1}$ is a gradient with respect to this scalar evaluated at identity scaling, which in practice reduces to the **gradient of the fine-tuning loss with respect to $\theta$ itself**.

Concretely, for a single environment $D_i$:

$$\nabla_{w|w=1} \ell_i(w \circ \theta; D_i) \approx \nabla_\theta \ell_i(\theta; D_i)$$

So the regulariser is simply $\|\nabla_\theta \ell_i(\theta; D_i)\|_2^2$: **the squared norm of the gradient that fine-tuning would take**. Minimising this gradient norm forces the model to sit at a point where fine-tuning has no first-order leverage — a stationary point with respect to the fine-tuning loss.

### 3.4 Single-environment reduction (the practical formulation)

The full multi-environment sum is expensive. The paper's key empirical finding is that **a single environment suffices**. The practical ILU objective is:

$$\min_\theta \; \ell_u(\theta) + \lambda \left\| \nabla_{w|w=1} \ell_i(w \circ \theta; D) \right\|_2^2 \tag{5}$$

where $D$ is a single fine-tuning dataset. Two choices for $D$ are studied:

- **Case (a) $D \perp D_f$**: $D$ is unrelated to the forget set (e.g., GSM8K vs. WMDP biology). The cosine similarity between their task vectors is approximately zero.
- **Case (b) $D = D_f$**: the forget set itself is used as the fine-tuning environment. This corresponds to the **relearning attack** scenario (Hu et al., 2024), where the adversary fine-tunes on the exact forgotten data.

Case (a) with an unrelated dataset produces **stronger and more generalisable robustness** than case (b), which is the counterintuitive result: training invariance against maths problems protects against biology relearning attacks. The mechanistic reason is that the unlearning direction $\tau_u$ is already far from $D_f$ in weight space; invariance trained against $D_f$ can produce an unlearning direction that is entangled with the forget-domain geometry, reducing generalisation.

### 3.5 Task vector interpretation

Define:
$$\tau_u = \theta_u - \theta_o, \quad \tau_{\text{ft}} = \theta_{\text{ft}} - \theta_o, \quad \tau_{u\to\text{ft}} = \theta^{\text{ft}}_u - \theta_u$$

A method is **resilient** iff $\cos(\angle(\tau_{u\to\text{ft}},\, \tau_u)) \geq 0$: post-unlearning fine-tuning stays aligned with (or orthogonal to) the unlearning direction, not opposite to it.

Measured values on WMDP:

| Quantity | NPO | ILU |
|---|---|---|
| $\cos(\angle(\tau_u, \tau_{\text{ft}}))$ | $-0.92$ | $-0.64$ |
| $\cos(\angle(\tau_{u\to\text{ft}}, \tau_u))$ | $-0.41$ | $+0.09$ |
| $\cos(\angle(\tau_{u\to\text{ft}}, \tau_{\text{ft}}))$ | $+0.16$ | $+0.36$ |

For NPO: after fine-tuning, $\tau_{u\to\text{ft}}$ is anti-aligned with $\tau_u$ — the model literally moves backwards along the unlearning direction. For ILU: $\tau_{u\to\text{ft}}$ is near-orthogonal to $\tau_u$ — fine-tuning shifts the model in directions that are irrelevant to the unlearning dimension. The IRM gradient norm penalty has geometrically decoupled fine-tuning directions from the unlearning direction in weight space.

### 3.6 Implementation note

In practice, computing $\|\nabla_\theta \ell_i(\theta; D)\|_2^2$ requires one backward pass through the fine-tuning loss. This is a **first-order operation** — unlike TAR's inner loop which requires $K = 64$ full gradient steps of simulated fine-tuning. The computational overhead of ILU relative to base NPO or RMU is therefore only one additional backward pass per step, making it far cheaper than any meta-learning alternative.

---

## 4. Immunisation Property Alignment

| Property | ILU coverage | Assessment |
|---|---|---|
| **Resistance** | ✅ Strong | Forget quality maintained across 6 diverse fine-tuning datasets at multiple epoch counts; 23% avg improvement over RMU on WMDP; VerbMem stays at 0 on MUSE-News under all fine-tuning settings where NPO collapses to 57.27 |
| **Stability** | ✅ Verified | Fine-tuning accuracy (FA) on downstream tasks is *improved* slightly by ILU relative to RMU/NPO, suggesting the IRM regulariser does not harm the model's general trainability at inference time |
| **Generalisation** | ✅ Core claim, well-verified | Training on a single unrelated dataset (GSM8K) generalises to AGNews, SST-2, MNLI, WinoGrande, QQP. Cross-domain in the immunisation sense. |
| **Trainability** | ⚠️ Partially implicit | FA is maintained on fine-tuning datasets, showing the model remains trainable on benign tasks after immunisation. However, the paper does not perform a formal trainability evaluation (convergence speed, step-count equivalence) as defined in Rosati et al., 2024. Trainability is confirmed incidentally but not the focus of measurement. |

**The missing piece** is a formal evaluation under **relearning attacks with strong compute budgets** (the "unlimited budget" regime of strong resistance). Table 3 shows relearning attack experiments, but only for a single epoch of fine-tuning on 60 randomly sampled forget-set instances — a relatively mild relearning scenario. Whether ILU would hold under thousands of fine-tuning steps on large forget sets, or under RL-based relearning (which is the stronger attack mode in the literature), is not established.

---

## 5. Mechanistic Commonalities with Other Approaches

**Gradient norm penalty as the common primitive.** Many immunisation methods, when you strip away their framing, reduce to a form of **penalising the gradient that the attacker would use**:

- **Booster** (Huang et al.): adds a regulariser during alignment that attenuates the *harmful perturbation* $\|\nabla_\theta \ell_{\text{harm}}(\theta)\|$ by simulating a small harmful step and measuring its effect on the alignment loss.
- **Vaccine / T-Vaccine**: introduces gradient-aware noise into embeddings; the noise is aligned with the gradient direction of the harmful fine-tuning signal to pre-flatten that direction.
- **TAR**: the outer loop minimises $\mathcal{L}_{\text{TR}}(\text{attack}(\theta); D_{\text{TR}})$ after an inner loop of $K$ gradient steps — which is a second-order Hessian-influenced operation but fundamentally also targets the fine-tuning gradient landscape.
- **Condition Number** approach: directly shapes the Hessian's condition number to make the harmful curvature large (gradient ascent is expensive) and benign curvature small (gradient descent is efficient).
- **ILU**: penalises $\|\nabla_\theta \ell_{\text{finetune}}(\theta; D)\|_2^2$ directly — the first-order, one-shot, single-backward-pass version of what TAR does with $K$ inner-loop steps.

The ILU gradient penalty can be seen as a **zeroth-order flat-landscape enforcer**: it pushes the model to a local minimum of the fine-tuning loss, so the first-order Taylor approximation of fine-tuning has zero gradient. TAR does the same but via simulation; Condition Number does it via Hessian eigenvalue shaping. ILU's version is the most computationally lightweight.

**Where ILU differs from all of these:** the other methods apply their resistance mechanism during an *alignment or safeguarding phase* and treat the base model as an input. ILU instead applies it as a **modifier on top of the unlearning loss itself** — it is the only method in this corpus that is purely a plug-in regulariser for an existing unlearning procedure, requiring no changes to the base model's training.

---

## 6. Results Summary

**WMDP benchmark (Zephyr-7B-beta, biosecurity knowledge removal):**

| Method | Pre-FT Forget Quality ↑ | Avg Post-FT Forget Quality ↑ (6 tasks) | FA on Downstream ↑ |
|---|---|---|---|
| NPO | 0.52 | ~0.43 (collapses) | Maintained |
| RMU | 0.68 | ~0.45 (collapses faster) | Maintained |
| NPO+ILU | 0.56 | ~0.55 (+28% vs NPO) | Maintained / improved |
| RMU+ILU | 0.68 | ~0.57 (+27% vs RMU) | Maintained / improved |

The key significance: RMU has *higher* pre-fine-tuning forget quality (0.68 vs 0.52 for NPO), yet collapses *faster* than NPO once fine-tuning begins. This confirms that high initial unlearning performance is not a proxy for robustness — a point the field had not cleanly demonstrated before. ILU separates the two quantities.

**MUSE benchmark (LLaMA-2-7B, BBC News / Harry Potter):**
NPO's VerbMem (verbatim memorisation) score under WinoGrande fine-tuning: 2.53 → 57.27 (fully recovering to the pre-unlearning level of 58.40). ILU maintains VerbMem at **0.0** across all 6 fine-tuning settings. This is a strong and clean result.

**Relearning attacks (Table 3, WMDP, Zephyr-7B-beta):**

| Method | FQ w/o attack ↑ | FQ w/ attack ↑ | Drop ↓ |
|---|---|---|---|
| NPO | 0.52 | 0.37 | 0.15 |
| NPO+SAM | 0.56 | 0.54 | 0.02 |
| NPO+ILU | 0.56 | 0.50 | **0.06** |
| RMU | 0.68 | 0.36 | 0.32 |
| RMU+SAM | 0.66 | 0.60 | **0.06** |
| RMU+ILU | 0.68 | 0.54 | 0.14 |

Interestingly, **SAM (Sharpness-Aware Minimisation)** as a baseline comparator outperforms ILU on the relearning attack scenario. This is an honest result: SAM explicitly minimises loss sharpness, which directly targets the relearning pathway. ILU is not designed to handle the worst-case scenario where the attacker has access to the exact forget set. The authors acknowledge this.

**Significance relative to other papers in the corpus:**
- ILU is one of the few methods that cleanly isolates the **post-deployment fine-tuning attack** surface, which most prior work ignores.
- The result that a *single* unrelated fine-tuning dataset suffices for training cross-domain invariance is novel and practically important — it means the defender does not need to anticipate the attacker's fine-tuning distribution.
- The comparison with SAM is important context: sharpness-aware optimisation during unlearning is a closer competitor than the paper's framing acknowledges.

---

## 7. Open Questions and Future Work

**From the authors:**

1. **Scaling**: all experiments use 7B-parameter models (Zephyr-7B-beta, LLaMA-2 7B, ICLM-7B). Whether IRM gradient penalties scale cleanly to 70B+ is unaddressed.
2. **Strong relearning attacks**: as Table 3 shows, ILU underperforms SAM on the relearning attack scenario. The adversary using the forget set directly remains a gap. A combination of ILU (for cross-domain fine-tuning attacks) and SAM or similar Hessian-aware methods (for relearning attacks) is suggested.
3. **Multi-environment scaling**: the paper shows that using multiple fine-tuning datasets in the invariance regulariser (ILU-Multi) does **not** outperform a single well-chosen unrelated dataset. Understanding why multi-environment training underperforms single-environment is a theoretically open question — it may relate to interference between invariance directions in weight space.

**From the state-of-the-art perspective:**

1. **RL-based relearning**: none of the unlearning papers in this corpus evaluate against RL-based harmful fine-tuning attacks, which have been shown to be strictly stronger than SFT attacks. ILU's invariance guarantee was derived from an IRM framework that is agnostic to the fine-tuning algorithm; whether the gradient norm penalty at unlearning time is sufficient to block RL gradient updates (which have lower entropy and more directed gradient signals) is unverified.
2. **Strong resistance vs. invariance**: the immunisation framework (Rosati et al., 2024) distinguishes weak resistance (cost-raising) from strong resistance (unlimited-budget blocking). ILU achieves weak resistance with good empirical margins, but its gradient norm penalty provides no theoretical bound against unlimited fine-tuning steps. A convergent adversary with infinite compute could presumably flatten the invariance advantage.
3. **Interaction with collapse-trap methods**: the most recent direction in the field (SDD, C-Trap) accepts model collapse as a resistance mechanism. It would be interesting to apply ILU's invariance penalty *inside* a self-destructing model: invariance regularisation could potentially extend the "pre-collapse" regime, delaying recovery attempts more gracefully than an abrupt collapse.
4. **Formal trainability evaluation**: ILU preserves fine-tuning accuracy on benign tasks empirically, but a formal comparison of convergence speed on benign tasks (as required by the Rosati et al. trainability condition) has not been performed. Given that the IRM penalty adds a gradient norm term, it is conceivable that the loss landscape is locally flattened in ways that slow convergence on *any* task, not just harmful ones — this should be measured.
