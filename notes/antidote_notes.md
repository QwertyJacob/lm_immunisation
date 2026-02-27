# Paper Notes: AntiDote — Bi-level Adversarial Training for Tamper-Resistant LLMs

**Authors:** Debdeep Sanyal, Manodeep Ray, Murari Mandal (RespAI Lab, KIIT Bhubaneswar)
**Venue:** AAAI 2025
**Code:** [github.com/respailab/Antidote](https://github.com/respailab/Antidote)

---

## 1. Quick Summary — Mechanistic Point of View

AntiDote frames LLM immunisation as a two-player adversarial game played entirely in weight space. The key mechanistic insight is this: instead of running an expensive inner-loop fine-tuning attack to simulate the adversary (as TAR does), the authors replace that inner adversary with a **differentiable neural network — a hypernetwork** — that reads the defender model's internal activations and produces adversarial LoRA weight patches in a single forward pass.

This collapses the intractable bi-level problem into a clean, end-to-end differentiable pipeline. The defender LLM is trained, via a reverse-DPO loss computed on the *attacked* model, to nullify those patches — i.e., to make itself locally invariant to the class of low-rank weight perturbations the hypernetwork can generate. A **separate, decoupled** objective simultaneously preserves general capability by training on clean data with cross-entropy + KL-divergence loss against the base model, ensuring that the gradient signal for utility is never contaminated by the adversarial objective.

The mechanistic effect is that the model is trained to produce activations that do not lend themselves to effective LoRA-based attacks: the hypernetwork keeps discovering the most damaging patches from the current activation distribution, and the defender keeps making that distribution attack-resistant. At convergence, the defender's activation geometry is hostile to harmful low-rank fine-tuning.

---

## 2. Timeline Positioning

AntiDote (arXiv: September 2024, AAAI 2025) sits near the **end of the first wave** of LLM immunisation methods, placing it as paper #19 in the 20-paper corpus. Its genealogy is as follows:

**What it inherits:**

- **Bi-level optimisation structure** — the same outer-minimiser / inner-maximiser skeleton that goes back to MLAC (Henderson et al., 2022) and is modernised by TAR (Tamirisa et al., 2024). AntiDote inherits the min-max problem formulation directly.
- **DPO as the preference-based safety loss** — TAR already showed that a DPO-style loss is better than cross-entropy for the tamper-resistance objective, because it directly models the preference inversion goal of the adversary. AntiDote uses the same loss.
- **LoRA as the parameterisation of attacks** — the field's prior recognition that harmful fine-tuning most commonly proceeds via low-rank updates (Qi et al., 2023; Vaccine, Booster) motivates AntiDote to model the attack space as LoRA.
- **Representation-level diagnostics** — the intuition that reading internal activations reveals vulnerability (Vaccine, RepNoise, Circuit Breakers) is what motivates using activations as the *input* to the hypernetwork, rather than a static embedding of the harmful prompt.
- **Decoupled capability preservation** — Booster already showed that separating safety and utility objectives reduces interference; AntiDote makes this decoupling structurally rigorous by computing capability loss on an entirely clean (unattacked) model.

**What is unique:**

The **adversarial hypernetwork** is AntiDote's singular contribution. No prior immunisation paper uses a learned, differentiable proxy for the adversary. Every other bi-level approach (MLAC, TAR, SOPHON, NTL) either runs actual fine-tuning for the inner loop or uses a first-order MAML-style straight-through approximation. AntiDote's hypernetwork makes the inner loop a single forward pass, removing the first-order bias entirely, and — crucially — makes the adversary **state-aware**: it reads the *current* activations of the defender, so it tracks the defender's evolution and cannot be evaded by representation drift.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 The Threat Model

The adversary has full access to weights $\theta$. Their goal is to select a fine-tuning strategy $A$ from all possible strategies $\mathcal{A}$ to maximise the probability of harmful responses:

$$\max_{A \in \mathcal{A}} \;\mathbb{E}_{(x_s, y_s, y_h) \sim \mathcal{D}_{\text{safe}}}\left[\log P(y_h \mid x_s;\; A(\theta))\right] \tag{1}$$

### 3.2 The Primal Min-Max Objective

A tamper-resistant model is one that minimises harm even after the adversary's best attack. This is the standard constrained min-max:

$$\theta^* = \arg\min_{\theta}\left(\max_{A \in \mathcal{A}} \mathcal{L}_{\text{harm}}(A(\theta))\right) \quad \text{s.t.} \quad \mathcal{L}_{\text{cap}}(\theta) \leq \epsilon \tag{2}$$

The harm loss is a **negative DPO safety loss**, which models the adversary's goal as inverting the model's preferences from safe to harmful:

$$\mathcal{L}_{\text{harm}}(\theta) = -\mathbb{E}_{(x_s, y_s, y_h) \sim \mathcal{D}_{\text{safe}}}\left[\log \sigma\!\left(\pi_\theta(y_s \mid x_s) - \pi_\theta(y_h \mid x_s)\right)\right]$$

The adversary *maximises* $\mathcal{L}_{\text{harm}}$ (inverting preferences); the defender *minimises* it (preserving preferences). $\mathcal{L}_{\text{cap}}(\theta)$ is a capability loss on the general distribution $\mathcal{D}_{\text{cap}}$, with $\epsilon$ a small tolerance.

### 3.3 Why Equation (2) is Intractable

The inner loop $\max_{A \in \mathcal{A}} \mathcal{L}_{\text{harm}}(A(\theta))$ requires finding the optimal attack $A^*$ for every gradient step of the outer loop — a full optimisation inside each outer step. Moreover, $A(\theta)$ is not differentiable w.r.t. $\theta$ when $A$ is itself a fine-tuning procedure, so standard first-order methods require a **straight-through approximation** (the MAML trick):

$$A(\theta) \approx \theta + \alpha \nabla_\theta \mathcal{L}_{\text{harm}}(\theta)$$

This is biased: a single-step attack is a strawman compared to a real fine-tuner, so the gradient signal underestimates the true threat.

### 3.4 The Adversarial Hypernetwork $H_\phi$

AntiDote replaces the intractable $A$ with a **differentiable neural network** $H_\phi$, parameterised by weights $\phi$, that generates adversarial LoRA matrices directly from the defender's internal activations.

**Input:** For a prompt $x$ and a target layer $l$, the hypernetwork receives the set of activation vectors produced by that layer:

$$\mathbf{X}_l = \{a_1, a_2, \ldots, a_N\} = \text{activations of layer } l \text{ on prompt } x$$

**Architecture of $H_\phi$:**

1. **Self-attention over $\mathbf{X}_l$** — identifies the most salient / vulnerable activation patterns within the sequence. Goes beyond a simple mean-pool.
2. **Residual feedforward blocks** — maps the context-aware pooled representation to a high-dimensional embedding with the expressive power needed to learn attack weights.
3. **Multi-headed output (dimension-specific heads)** — since LLM layers have heterogeneous shapes (`q_proj` vs. `mlp.down_proj`), specialised output heads generate LoRA matrices for each unique $(d_{\text{in}}, d_{\text{out}})$ configuration.

**Output:** For layer $l$ with input dimension $d_{\text{in}}$ and output dimension $d_{\text{out}}$:

$$(\mathbf{U}_l, \mathbf{V}_l) = H_\phi(\mathbf{X}_l(x;\theta)) \tag{3}$$

where $\mathbf{U}_l \in \mathbb{R}^{r \times d_{\text{in}}}$, $\mathbf{V}_l \in \mathbb{R}^{d_{\text{out}} \times r}$.

The adversarial low-rank weight patch applied to layer $l$ is:

$$\Delta W_l = \mathbf{V}_l^\top \mathbf{U}_l$$

### 3.5 The Attacked Model

When the adversarial patch is applied, the attacked model at layer $l$ becomes:

$$W_l^{\text{atk}} = W_l + \Delta W_l = W_l + \mathbf{V}_l^\top \mathbf{U}_l$$

The full attacked model $M_\theta^{\text{atk}}$ uses these patched weights in targeted layers.

### 3.6 Training Objectives — Interleaved and Decoupled

AntiDote separates training into two alternating phases:

**Phase A — Adversarial Co-Evolution (Safety):**

The **hypernetwork** is updated to maximise harm on the attacked model:

$$\phi \leftarrow \phi + \eta_\phi \nabla_\phi \mathcal{L}_{\text{harm}}(M_\theta^{\text{atk}})$$

The **defender's LoRA weights** $\theta_\Delta$ are updated to *minimise* harm under attack (tamper-resistance loss computed on the *attacked* model):

$$\theta_\Delta \leftarrow \theta_\Delta - \eta_\theta \nabla_{\theta_\Delta} \mathcal{L}_{\text{harm}}(M_\theta^{\text{atk}})$$

Because the full pipeline $\theta \to \mathbf{X}_l \to H_\phi(\mathbf{X}_l) \to \Delta W \to M_\theta^{\text{atk}} \to \mathcal{L}_{\text{harm}}$ is differentiable end-to-end, gradients flow cleanly back through the hypernetwork to the defender — no approximation required.

**Phase B — Capability Preservation (Utility):**

On the **clean, unattacked model** (no $\Delta W$ applied), the defender is trained with a combined loss over a general-purpose dataset $\mathcal{D}_{\text{cap}}$:

$$\mathcal{L}_{\text{cap}}(\theta) = \mathbb{E}_{(x_c, y_c) \sim \mathcal{D}_{\text{cap}}}\!\left[\mathcal{L}_{\text{CE}}(\theta, x_c, y_c) + \lambda \cdot D_{\text{KL}}\!\left(\pi_\theta(\cdot \mid x_c) \;\|\; \pi_{\theta_{\text{base}}}(\cdot \mid x_c)\right)\right]$$

The KL term anchors the defender to the base model's distribution, preventing capability drift. Crucially, computing this loss on the clean model ensures the gradient signal for utility is **never contaminated** by the adversarial objective — the "gradient purity" claim.

**Memory optimisation:** Both defender and hypernetwork are trained with LoRA only. The DPO reference model is not stored separately in VRAM — instead, the reference state is recovered by resetting to the original LoRA adapter, exploiting the fact that the reference is the defender's own initial adapter. The inactive player (hypernetwork or defender, depending on the phase) is offloaded to CPU during the other phase's update.

---

## 4. Immunisation Properties — Alignment and Gaps

**Resistance ✓ (primary claim, strong evidence)**

AntiDote's explicit training objective is to minimise harm after worst-case attack. The adversarial game ensures resistance is trained against an adaptive adversary that continuously finds the most damaging patch available in the LoRA attack space. Tested against 52 attack types including jailbreak prompting, activation steering, and direct weight-space perturbations. Achieves up to 78% reduction in Harmful Score vs. SFT baseline, outperforming TAR and RepNoise.

**Stability ✓ (strong)**

The decoupled Phase B explicitly preserves capability. MMLU, HellaSwag, GSM8K, PrOntoQA benchmarks show less than 0.5% degradation. In several settings, fine-tune accuracy after AntiDote is *higher* than after prior immunisation methods, because the gradient purity prevents the safety objective from interfering with capability learning.

**Generalisation ✓ (tested, some gaps)**

The 52-attack evaluation covers diverse attack *types* (not just one harmful domain), providing cross-type generalisation evidence. The use of BeaverTails + Do-Not-Answer for training provides some domain breadth. However, generalisation to unseen *harmful domains* (cross-domain generalisation in the Rosati et al. definition) is not systematically ablated. The hypernetwork's attack space is bounded by the LoRA rank $r$ — sufficiently high-rank attacks or weight edits outside the LoRA manifold are not directly trained against.

**Trainability ✓ (implicit, strongest of all papers so far)**

This is AntiDote's most surprising result: AntiDote does *not* sacrifice trainability. The decoupled capability loss, trained on clean activations, means the defender's LoRA weights learn safety and utility as orthogonal skills. The paper shows that fine-tuning accuracy on benign tasks is preserved or improved — a direct empirical demonstration of trainability in the Rosati et al. sense.

**The missing piece:**

Generalisation against unseen *algorithmic* attack classes is under-specified. The hypernetwork is trained to generate LoRA-shaped attacks, but a sufficiently motivated adversary could use full-rank fine-tuning, gradient-free attacks (e.g., direct weight surgery, model merging), or RL-based attacks. The paper tests a broad attack *catalogue* but does not provide the theoretical bounds on training steps required to break the defence (the "weak resistance" formalisation), nor cross-domain harmful generalisation ablations in the Rosati et al. framework.

---

## 5. Mechanistic Commonalities with Other Approaches

**Shared with TAR, MLAC, SOPHON:**

All use the bi-level min-max skeleton — an outer loop that minimises safety loss after an inner adversary. The difference is how the inner adversary is simulated. MLAC/SOPHON use actual gradient descent on the inner loop with MAML's first-order approximation. TAR uses the same but with entropy loss instead of cross-entropy to suppress adversary recovery. AntiDote replaces the inner loop entirely with $H_\phi$.

**Shared with Vaccine and T-Vaccine:**

All three act on the model's representation space during immunisation — Vaccine by adding explicit Gaussian noise to embeddings during alignment (perturbing the input side), AntiDote by reading activations and generating weight patches that corrupt the activation propagation (perturbing the weight side). Both aim to make the internal representation geometry robust to fine-tuning perturbations.

**Shared with RepNoise and Circuit Breakers:**

The intuition that harmful content triggers distinct activation patterns — and that immunity means neutralising those patterns — is shared. RepNoise adds noise to harmful-content representations to destroy the signal. AntiDote trains against a learnable proxy that finds the most damaging activation-conditioned perturbation. Both target the representation as the locus of vulnerability.

**Shared with Booster:**

The decoupled optimisation strategy (separating safety and utility losses into distinct phases) appears in Booster in a weaker form (a unified loss with a weighting scheme). AntiDote makes the decoupling structural by computing the capability loss on a clean model, which is more principled.

**Key differentiator vs. all:**

The **state-aware adversary** is unique. All prior methods train against a fixed attack template or a first-order approximation of gradient descent. $H_\phi$ is a function of the defender's *current* activations, meaning the attack evolves in real-time as the defender changes. This is the only paper in the corpus where the adversary and defender co-evolve with mutual feedback.

---

## 6. Results — Summary and Field Significance

**Safety-utility trade-off frontier:**

On a scatter plot of Fine-tune Accuracy (FA, higher = better) vs. Harmful Score (HS, lower = better) across MMLU, GSM8K, HellaSwag, and PrOntoQA, AntiDote is the **only method in the Pareto-optimal quadrant** — highest FA *and* lowest HS simultaneously. All other methods (RMU, Booster, TAR, RepNoise, Vaccine) lie on a trade-off curve where safety improvements come at a cost to utility.

**Average numbers (across 4 benchmarks, representative models):**

| Method   | Avg FA ↑   | Avg HS ↓   |
|----------|-----------|-----------|
| SFT      | 65.6      | 15.2      |
| Booster  | 65.8      | 7.3       |
| TAR      | 65.3      | 9.7       |
| RepNoise | 64.8      | 11.4      |
| Vaccine  | 65.6      | 8.4       |
| **AntiDote** | **66.4** | **6.3** |

**Scale:**

Tested on 10 models from 0.6B to 27B parameters (Qwen3, Gemma3, Llama3.2 families). Computational cost scales more favourably than TAR — the hypernetwork's cost is *constant* regardless of target LLM size, since it generates LoRA patches via a fixed-size forward pass. Memory is lower than RepNoise and TAR at 12B scale (73.6 GB vs. 95.8 GB for TAR, 100.1 GB for RepNoise).

**Red-teaming breadth:**

The 52-attack evaluation is the most comprehensive in the corpus. Prior papers typically evaluate against 3–10 attack configurations. AntiDote uniquely handles role-playing attacks and adversarial suffix attacks better than Booster (which relies on local gradient information and is blind to attacks that manipulate the semantic gradient weakly), because the hypernetwork recognises anomalous *activation patterns*, not just anomalous prompts.

---

## 7. Future Work — Authors' and Critical Perspectives

### From the authors:

- Extension to **non-LoRA attack classes** — the hypernetwork generates rank-$r$ updates, but a sufficiently capable adversary may use higher-rank or full-parameter fine-tuning. Future work should study whether activation-conditioned hypernetworks can generalise to these.
- **Optimal rank selection** for the adversarial LoRA — the rank $r$ is a hyperparameter that determines the expressivity of the attack space. A too-small $r$ may produce a strawman adversary; the authors acknowledge this is not yet principled.
- Scaling the hypernetwork architecture alongside the target LLM — currently the hypernetwork is fixed-size, which is computationally efficient but may become a bottleneck as defenders grow.

### From the state-of-the-art (gaps visible from other papers in the corpus):

- **Cross-domain generalisation** (Rosati et al., 2024) is not tested. AntiDote's evaluation uses diverse attack *mechanisms* but not systematically different *harm domains* (e.g., train on toxicity, test on bioweapons). This is the missing immunisation property.
- **RL-based attacks** (e.g., the RL fine-tuning attacks described in the long index) are not in the 52-attack suite. RL attacks achieve superior Pareto efficiency over SFT for harmful tasks by reducing response entropy — it is unclear whether AntiDote's LoRA-proxy adversary can model this.
- **Inference-time attacks** (jailbreak prompting, activation steering, ITI): while some jailbreak prompts appear in the 52 attacks, ITI-class attacks (e.g., LoReFT interventions, ReFT-type approaches tested in E.T.) are not systematically tested. The E.T. paper shows these are distinct from weight-space attacks.
- **Evaluation durability** (Qi et al., "On Evaluating the Durability of Safeguards for Open-Weight LLMs", ICLR 2025): this paper argues that most immunisation evaluations are too narrow — evaluating durability requires testing against adaptive adversaries who know the defence and specifically target its weaknesses. AntiDote's adversary *is* adaptive (it reads activations), but its attack space is bounded by the LoRA parameterisation. An adversary who knows this can mount attacks outside the LoRA manifold. The durability paper would demand testing with full-rank fine-tuning adversaries who know the hypernetwork architecture.
- **Trainability under benign distribution shift** — AntiDote shows that benign fine-tuning accuracy is preserved on the *same tasks used in training*. Trainability in the Rosati et al. sense requires that *arbitrary* harmless downstream tasks can be learned efficiently — this is not ablated with OOD benign datasets.
- **The hypernetwork's convergence dynamics** — the co-evolution of the hypernetwork and the defender is a GAN-like game and may exhibit mode collapse (the hypernetwork specialises on one attack mode) or oscillation. No analysis of training stability or Nash equilibrium convergence is provided.
