# T-Vaccine — Notes
**Paper:** *Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation*  
**Authors:** Guozhi Liu, Weiwei Lin et al. — South China University of Technology / Pengcheng Laboratory  
**Venue:** EMNLP 2024 (posted arXiv 2410.09760)

---

## 1. Quick Summary — Mechanistic Point of View

T-Vaccine is a **direct refinement of Vaccine** (Huang et al. 2024). Both live in the same mechanistic family: *perturbation-aware alignment-stage immunisation through representation noising*. The core immunisation hypothesis is that harmful fine-tuning works by drifting the model's internal embeddings (in the residual stream) away from the safe-response manifold — a phenomenon called **Harmful Embedding Drift (HED)**. The defence strategy is to make those embeddings maximally invariant to adversarial perturbations during alignment, so that the embedding space is hardened against the direction a harmful gradient would push.

Vaccine did this by applying the same adversarial perturbation budget to **every layer** of the transformer during the second forward pass of each alignment step. T-Vaccine's single contribution is surgical: it asks which layers actually carry safety-relevant information, estimates that via **harmful-data gradient norms**, converts those norms into a **probabilistic sampling distribution**, and then only applies perturbation (and only unfreezes, and only stores activations/gradients for) the sampled safety-critical subset of layers. Everything else about the recipe — the two-pass adversarial training loop, the FGSM-style optimal perturbation formula, the alignment dataset — is inherited unchanged from Vaccine.

The practical payoff is twofold: better defence (because noising safety-irrelevant layers in Vaccine was actively hurting performance) and a large memory saving (because frozen layers pay zero activation/gradient/optimizer memory), making 7B-scale immunisation feasible on a 24 GB consumer GPU.

---

## 2. Position in the Timeline

### What T-Vaccine Inherits

| Paper | What T-Vaccine Takes |
|---|---|
| **Vaccine** (Huang et al., 2024e) | The full perturbation-aware alignment recipe: HED motivation, two-pass loop, FGSM-style perturbation formula (Eq. 1–3 in Vaccine ≡ Eq. 1–3 here), LoRA-based training, the BeaverTails alignment/harmful dataset split. |
| **RepNoise** (Rosati et al., 2024b) | Conceptual companion: the idea that safe embeddings should be made noise-like w.r.t. harmful inputs. T-Vaccine benchmarks against it. |
| **TAR** (Tamirisa et al., 2024) | Meta-learning baseline in the comparison table. Provides the adversarial outer-loop perspective that informs what "robustness" means. |
| **LISA / OwLore** (Pan et al. 2024; Li et al. 2024) | Layer-importance sampling literature from memory-efficient fine-tuning. T-Vaccine directly borrows the notion of gradient-norm-based probabilistic layer selection. |
| **Immunisation Definition** (Rosati et al., 2024 EMNLP) | Provides the formal four-property framework (resistance, stability, trainability, generalisation) against which the paper should be evaluated — even though T-Vaccine does not cite this framing explicitly. |

### T-Vaccine's Unique Contribution

The paper makes **one technically original claim**: that the gradient norm of a harmful dataset is a reliable, cheap proxy for layer safety-relevance, and that replacing uniform perturbation with a gradient-norm-weighted probabilistic subset selection strictly dominates the original Vaccine on both effectiveness and memory. No other paper in the immunisation corpus applies this idea.

**Timeline position**: T-Vaccine appears roughly concurrent with RepNoise and TAR (all late 2024), after Vaccine but before Circuit Breakers and Booster. In the E.T. (2025) comparative table it is one of the seven baselines evaluated — sitting in the middle of the field, clearly above Vaccine on all safety metrics but below Circuit Breakers on stability and clearly below E.T. on resistance to inference-time interventions (a threat model it never considers).

---

## 3. The Math — Detailed Mechanistic Description

### 3.0 Setup

- Model weights at step $t$: $\boldsymbol{w}_t$  
- $L$ transformer layers; hidden embedding of layer $l$: $\boldsymbol{e}_{l,t}$  
- Alignment dataset $\mathcal{D}_a$ (harmful-prompt / safe-answer pairs)  
- Harmful dataset $\mathcal{D}_h$ (harmful-prompt / harmful-answer pairs), used **only** to compute layer importance — never to update the model toward harmful responses  
- Hyperparameters: perturbation budget $\rho$, number of sampled layers per step $\gamma$, importance recalculation period $K$

---

### 3.1 Step 1 — Layer Importance via Harmful Gradient Norm (every $K$ steps)

Run a forward-backward pass over a batch from $\mathcal{D}_h$ and compute the per-layer embedding gradient norm:

$$
s_{l,t} = \left\| \nabla_{\boldsymbol{e}_{l,t}} \mathcal{L}_{\boldsymbol{w}_t}(\boldsymbol{e}_{l,t};\ \boldsymbol{x}_{h}, \boldsymbol{y}_{h}) \right\|_2
\tag{4}
$$

This gives a scalar importance score for each of the $L$ layers. The intuition, borrowed from sparse-training literature (Evci et al., 2020), is that layers whose parameters matter most for predicting harmful outputs will produce large gradient signals when evaluated on harmful data — so those are the layers whose embeddings need hardening.

Convert scores into a proper probability distribution:

$$
p_{l,t} = \frac{s_{l,t}}{\displaystyle\sum_{l'=1}^{L} s_{l',t}}
\tag{5}
$$

The vector $P_t = [p_{1,t},\ldots,p_{L,t}]$ is held fixed for the next $K$ training steps and re-evaluated only after that.

**Empirical finding that motivates this**: on Llama-2-7B the last 10 layers have systematically small gradient norms over harmful data and, correspondingly, adding perturbation to them *increases* the harmful score rather than reducing it.

---

### 3.2 Step 2 — Stochastic Layer Selection (every step)

Sample $\gamma$ layer indices without replacement according to $P_t$:

$$
S_t \sim \mathrm{Categorical}(P_t),\quad |S_t| = \gamma
$$

All layers $l \notin S_t$ are **frozen**: their weights receive no gradient, their activations are not stored, and their optimizer states are not updated. This is where the memory savings come from — practically, activation/gradient/optimizer memory only for $\gamma$ out of $L$ layers.

---

### 3.3 Step 3 — Two-Pass Perturbation-Aware Alignment (every step, same as Vaccine but scoped to $S_t$)

**Pass 1** — gradient on alignment data through the selected layers only:

$$
\nabla \mathcal{L}_{\boldsymbol{w}_t}(S_t) = \nabla_{\boldsymbol{e}_{l,t},\ l\in S_t}\,\mathcal{L}\bigl((\boldsymbol{x}_t, \boldsymbol{y}_t) \sim \mathcal{D}_a\bigr)
$$

**Optimal perturbation** for each sampled layer (FGSM-style, normalised by the concatenated gradient norm of the *subset*):

$$
\boldsymbol{\epsilon}_{l,t} = \rho\,\frac{\nabla_{\boldsymbol{e}_{l,t}}\mathcal{L}_{\boldsymbol{w}_t}(\boldsymbol{e}_{l,t})}{\left\|\nabla\mathcal{L}_{\boldsymbol{w}_t}(S_t)\right\|_2},\quad l \in S_t
\tag{6}
$$

Note the key difference from Vaccine's Eq. (1): the denominator is the norm of the concatenated gradient over **only** $S_t$, not all $L$ layers. This keeps the perturbation magnitude consistent under partial-layer training.

**Pass 2** — forward pass with perturbation injected as a residual addition:

$$
\tilde{f}_{\boldsymbol{w}_l,\boldsymbol{\epsilon}_{l,t}}(\boldsymbol{e}_{l,t}) = f_{\boldsymbol{w}_l}(\boldsymbol{e}_{l,t}) + \boldsymbol{\epsilon}_{l,t},\quad l \in S_t
$$

Backward through the perturbed forward pass:

$$
\tilde{\boldsymbol{g}}_t = \nabla\mathcal{L}\!\left(\tilde{f}_{\boldsymbol{w}_{L,t}} \circ \cdots \circ \tilde{f}_{\boldsymbol{w}_{1,t}} \circ \mathcal{T}(\boldsymbol{x}_t, \boldsymbol{y}_t)\right)
$$

(where frozen layers contribute identity-like operations, so in practice only the $\gamma$ selected layers actually backpropagate).

**Weight update**:

$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta\,\tilde{\boldsymbol{g}}_t
$$

---

### 3.4 What the Two-Pass Loop Is Really Doing

The two-pass loop is solving a **min-max problem at the embedding level**:

$$
\min_{\boldsymbol{w}} \max_{\boldsymbol{\epsilon}:\ \|\boldsymbol{\epsilon}\|\leq\rho}\ \mathcal{L}_{\mathrm{alignment}}\bigl(\boldsymbol{w};\ \boldsymbol{e} + \boldsymbol{\epsilon}\bigr)
$$

restricted to the selected layers. Pass 1 finds the worst-case $\boldsymbol{\epsilon}$ (inner max). Pass 2 takes a gradient step on the weights at that worst case (outer min). This is a first-order saddle-point approximation (no second-order Hessian or meta-gradient needed). The perturbation is not applied to the harmful data — it is applied to the *alignment* data, to make the safe representations hard to destabilise. The harmful data is used only in the importance scoring step.

---

## 4. The Four Immunisation Properties

| Property | T-Vaccine's Stance |
|---|---|
| **Resistance** | Directly targeted. HS drops to ~15% on LLama-2-7B after attack (vs ~45-50% for SFT). The probabilistic layer selection improves resistance over Vaccine by ~8 pp. Still not state-of-the-art: TAR and E.T. reach lower absolute HS under stronger adaptive attacks. |
| **Stability** | Explicitly preserved as an experimental metric (Finetune Accuracy / FA). T-Vaccine maintains FA ~92% on SST2/AGNEWS/GSM8K. Notably, frozen layers help stability because they are not perturbed at all. However, in the E.T. comparison table, T-Vaccine's GSM8K retention drops to 0.01 (versus baseline 1.0) — a major red flag for arithmetic/reasoning tasks. |
| **Trainability** | Explicitly validated (FA metric). The model can still be fine-tuned on benign downstream tasks after immunisation, and accuracy is roughly preserved. This is the property that distinguishes T-Vaccine from the "collapse" family. |
| **Generalisation** | Only weakly addressed. The paper trains and tests within the BeaverTails distribution. There is no test on out-of-distribution harm categories, no cross-dataset evaluation, and no adversarial attack diversity beyond varying the harmful data ratio. This is the largest gap. |

**Primary strength**: resistance + trainability (the core immunisation trade-off).  
**Missing piece**: generalisation is essentially untested.

---

## 5. Mechanistic Commonalities with Other Approaches

### Gradient Ascent / FGSM-Style Perturbation (Shared Core)

T-Vaccine, Vaccine, and RepNoise all rely on the same structural idea: compute the gradient of a safety-relevant loss w.r.t. the *embeddings* (not the weights), normalise it, scale by $\rho$, and inject the result as an additive perturbation. This is a first-order adversarial perturbation at the representation level, formally identical to FGSM in adversarial example literature. The formula:

$$
\boldsymbol{\epsilon}^* = \rho \cdot \frac{\nabla_{\boldsymbol{e}}\mathcal{L}}{\|\nabla_{\boldsymbol{e}}\mathcal{L}\|}
$$

appears in Vaccine and T-Vaccine with only a change in what constitutes the normalisation denominator (all layers vs. the selected subset).

### Bi-Level / Two-Pass Optimization (Shared with meta-learning family)

The inner max / outer min structure is also present in SDM, TAR, and SOPHON, which do it at the weight level using MAML-style gradient unrolling. T-Vaccine does it at the embedding level with a single FGSM step — computationally much cheaper but conceptually the same adversarial training idea.

### Layer-Selection via Gradient Information (Unique Instantiation)

The use of harmful-gradient norms to select which layers to touch is T-Vaccine's own contribution. Thematically related: the Brittleness paper (Huang et al., 2024) isolates safety-critical layers via attribution methods (SNIP, ActSVD), and finds that safety is concentrated in a sparse subset of ranks/layers. T-Vaccine reaches the same conclusion empirically but uses gradient norms on harmful data rather than attribution decompositions.

### Representation-Level Immunisation (Shared with RepNoise, E.T., Circuit Breakers)

All these methods immunise the residual stream rather than acting at the weight level. RepNoise pushes harmful representations toward random noise. Circuit Breakers re-routes harmful representations to an orthogonal subspace. E.T. alternates attack/defence rounds layer-by-layer. T-Vaccine adds adversarial noise to selected-layer embeddings during alignment. All four can be seen as different strategies to make the hidden-state manifold corresponding to harmful outputs inaccessible or unstable.

---

## 6. Results Summary and Significance

### Core Numbers (Llama-2-7B, default h=0.1 harmful ratio)

| Method | Harmful Score ↓ | Finetune Accuracy ↑ | GPU Memory (GB) ↓ |
|---|---|---|---|
| Non-Aligned | ~50% | ~92% | ~18 |
| SFT | ~45% | ~92% | ~18 |
| Vaccine | ~23% | ~92% | ~34 |
| TAR | ~19% | ~92% | ~46 |
| RepNoise | ~26% | ~92% | ~46 |
| **T-Vaccine (γ=8)** | **~15%** | **~92%** | **~23** |

- T-Vaccine improves on Vaccine by ~8 pp HS while cutting memory by ~32%.
- T-Vaccine beats TAR and RepNoise on HS while using half their memory.
- T-Vaccine is the **only** method that fits a 7B model on a 24 GB GPU.
- Generalises across four model families (Llama-2-7B, Gemma-2-2B, Vicuna-7B, Qwen2-7B).

### Context within the Corpus

T-Vaccine occupies a clear niche: it is the best method along the **safety-vs-memory efficiency frontier** at the time of publication. However, it predates and does not address inference-time interventions (ITI), which E.T. (2025) shows are a distinct and harder threat. In the E.T. benchmark, T-Vaccine shows the lowest mean toxicity under ITI-attacks among the noising-family methods (0.57), but its stability is severely degraded (GSM8K retention 0.01). This suggests its perturbation mechanism is aggressive in ways that hurt arithmetic tasks.

---

## 7. Calls for Future Work

### From the Authors

- Extending to more attack types beyond the standard BeaverTails harmful fine-tuning setting.
- More rigorous evaluation of the generalisation condition — testing on harm categories not seen during immunisation.
- Analysis of the interaction between layer selection and model architecture (the paper shows consistent results across four architectures but does not explain *why* the same layers are safety-critical across different models).
- Adaptive attacks against T-Vaccine specifically (an attacker who knows the immunisation procedure might craft fine-tuning data to exploit the frozen layers).

### From the Immunisation Definition Paper (Rosati et al., EMNLP 2024)

- Generalisation is the weakest condition tested — T-Vaccine trains and evaluates on overlapping BeaverTails distributions. A proper generalisation test requires truly disjoint harm categories.
- The immunisation conditions require testing at varying attacker budgets (training steps, data quantity). T-Vaccine's robustness to harmful ratio is tested (Table 1) but not to budget size.
- The paper should evaluate the trainability condition on tasks the model cannot solve at all zero-shot before immunisation, to confirm fine-tuning is not merely recalling rather than learning new knowledge.

### From "On Evaluating the Durability of Safeguards for Open-Weight LLMs" (Qi et al., 2024 / ICLR 2025)

This is the most damaging critique applicable to T-Vaccine:

- **White-box adaptive attacks**: because T-Vaccine's layer selection is deterministic between recalculations (period $K$), an attacker with full weight access can construct fine-tuning data that specifically targets the *non-selected* (frozen) layers and gradually undermines safety from there. The paper does not test this scenario.
- **Durability vs. compute**: the durability paper argues that safety safeguards should be evaluated against attackers with increasing compute budgets. T-Vaccine's harmful-score results use 20-epoch fine-tuning. Longer or more targeted attacks may erode the defence.
- **Sparse safety representations**: the Brittleness paper (Huang et al. 2024) showed that safety in Llama-2-chat is localised in a small, isolatable set of ranks/heads. T-Vaccine's layer selection implicitly assumes that gradient norms identify the relevant layers — but it does not verify that those layers actually correspond to the safety-critical subspace found by attribution methods. An attacker could potentially find and surgically modify the safety-relevant subspace while staying outside the gradient-norm-high layers.

### From "Rethinking LLM Unlearning Objectives" and Other Criticism Papers

- The stability metric (FA on SST2/GSM8K) conflates two distinct things: whether the model can be fine-tuned at all, and whether the fine-tuned model is good. A model that converges to random guessing on the fine-tuning task after harmful attack would also score low on FA — and T-Vaccine does not rule this out.
- Harmful Score using BeaverTails' own classifier is circular (same distribution for train and eval). A proper resistance test should use a held-out red-teaming set or human evaluation.

---

*Notes prepared for the LM Immunisation Tutorial — Paper 4 of 20.*
