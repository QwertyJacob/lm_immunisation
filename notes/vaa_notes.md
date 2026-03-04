# VAA — Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning

> Chen et al., ICML 2025 · [arXiv:2506.03850](https://arxiv.org/abs/2506.03850)

---

## 1. Quick Mechanistic Summary

VAA starts from a single empirical observation: **alignment data is not equally fragile**. When a user fine-tunes an aligned model on a dataset that contains even a small fraction of harmful examples, the model does not forget its safety alignment uniformly. Some safety examples get forgotten almost immediately — after just a few gradient steps — while others remain intact long after the attack is over. VAA calls these the *vulnerable* and *invulnerable* groups, respectively.

The mechanistic diagnosis is loss-landscape geometry. Vulnerable examples sit in sharper, narrower basins: a small perturbation to the model weights is enough to push the prediction from safe to harmful. Because these examples are also typically fewer in number, standard ERM (empirical risk minimisation) training produces a *gradient starvation* effect — the gradients of the majority invulnerable group dominate, the minority vulnerable group is underfit, and the gap in robustness widens.

The fix is a two-stage alignment procedure:

1. **Stage 1 — Vulnerability Profiling.** Run a proxy harmful fine-tuning pass (on Alpaca + 10% harmful data), observe which alignment examples flip from safe to harmful output over $T$ checkpoints, and record each example's *ForgotNum*. Examples with $\text{ForgotNum} > 0$ become group $G_2$ (vulnerable); the rest are $G_1$ (invulnerable).

2. **Stage 2 — Adversarial Resampling with Group-Specific Perturbations.** Train the model under a minimax objective: an *adversarial sampler* $q$ continuously shifts probability mass toward whichever group the model is currently worst at, while the model trains on that group with a *group-specific weight-space perturbation* that simulates the worst plausible fine-tuning attack. The adversary is discarded after training; only the model weights are released.

The net effect is a flatter, wider safety basin — particularly around the previously vulnerable examples — making the released open-weight model substantially harder to de-align by subsequent fine-tuning.

---

## 2. Positioning in the Immunisation Timeline

### Where VAA sits

VAA is an **alignment-stage defence**. It acts once, modifying how the safety alignment pass is performed, before the model is released. Every request to fine-tune the released model then inherits that structural robustness. This places it in the same broad family as **Vaccine** and **Booster**, and squarely within the scope of the tutorial's immunisation problem.

### What it inherits

| Ancestor | Contribution borrowed |
|---|---|
| **Vaccine** (Huang et al., NeurIPS 2024) | The basic idea of perturbing model parameters *during alignment* to simulate HFT, so the model learns to stay aligned under perturbation. VAA inherits the conceptual template but applies perturbations per data group, not uniformly. |
| **SAM** (Foret et al., ICLR 2021) | The first-order Taylor trick for approximating the worst-case weight perturbation $\epsilon_i^*$ in closed form. VAA lifts this verbatim, repurposing it from "find flat minima for generalisation" to "find the scariest safety-forgetting direction per group." |
| **Group DRO** (Sagawa et al., ICML 2020) | The GDRO minimax objective and the EXP3-style mirror-ascent update for $q$. VAA is structurally Group DRO applied to safety alignment data, partitioned by empirical vulnerability. |
| **Booster** (Liu et al.) | The motivating lens of gradient dynamics under HFT. Booster attenuates harmful-direction gradients; VAA up-weights safe-direction gradients for the vulnerable subset. Different levers, same threat model. |

### What is unique

No prior alignment-stage method recognised that safety alignment data is internally heterogeneous in its fragility, or did anything about that heterogeneity. Vaccine and Booster apply their perturbations and gradient corrections uniformly across all alignment data. VAA is the **first to formalise a data-level vulnerability structure and make the perturbation magnitude and sampling frequency group-dependent and adaptive**. The cross-model transferability finding (vulnerability labels estimated on LLaMA-2 transfer to Qwen2.5 without re-clustering) further suggests vulnerability is a *data* property, not an architecture artefact.

**Compared to bi-level methods** (TAR, Antidote): VAA is cheaper. It does not simulate the attacker's full fine-tuning trajectory; it only applies a single-step worst-case perturbation as a proxy. This makes it less principled in its threat model but far more tractable. Antidote (appearing days after VAA) is the most natural comparison point and is notably absent from VAA's experiments.

**Compared to weight-space extrapolation** (LoX): VAA operates entirely in gradient space. No SVD, no post-hoc merging, no need to store intermediate model checkpoints.

---

## 3. The Math — Step by Step

### 3.1 Vulnerability Quantification (Stage 1)

Let $\mathcal{D}_\text{align} = \{(x_j, y_j)\}_{j=1}^N$ be the alignment dataset (harmful prompt, safe response pairs). Run a proxy HFT on a pre-aligned model for $T$ training steps, evaluating model predictions on each $(x_j, y_j)$ at each step. Define:

$$\text{ForgotNum}_j = \sum_{t=1}^{T} \mathbb{1}\bigl[M_{\theta[t]}(x_j) \neq \text{safe}\bigr] \cdot \mathbb{1}\bigl[M_{\theta[t-1]}(x_j) = \text{safe}\bigr] \tag{1}$$

This counts the number of times example $j$ *transitions* from safe to harmful across checkpoints. Then:

$$G_2 = \{(x_j, y_j) \mid \text{ForgotNum}_j > 0\} \quad \text{(vulnerable)}$$
$$G_1 = \{(x_j, y_j) \mid \text{ForgotNum}_j = 0\} \quad \text{(invulnerable)}$$

**Why this works without knowing the real attacker's data.** The paper demonstrates empirically that vulnerability patterns are *transferable*: examples that are vulnerable under Alpaca+10%-harmful fine-tuning are also vulnerable under SST-2+10%-harmful, AG News+10%-harmful, etc. So the proxy pass is a good predictor of real-world fragility.

---

### 3.2 The Robust Per-Group Objective (The Model's Problem)

Standard ERM minimises:

$$\hat{\theta}_\text{ERM} = \arg\min_{\theta \in \Theta}\; \mathbb{E}_{(x,y) \sim \hat{P}}\bigl[\ell(\theta; (x, y))\bigr] \tag{6}$$

This treats all examples equally. Because $|G_1| \gg |G_2|$, gradients from $G_1$ dominate and the vulnerable group is systematically underfit — classic *gradient starvation*.

VAA replaces the per-group loss $\ell_i(\theta)$ with a **robustness-augmented surrogate**:

$$f_i(\theta) = \ell_i(\theta) + \lambda\bigl(\ell_i(\theta + \epsilon_i) - \ell_i(\theta)\bigr) \tag{4}$$

Rearranging:

$$\boxed{f_i(\theta) = (1 - \lambda)\,\ell_i(\theta) + \lambda\,\ell_i(\theta + \epsilon_i)} \tag{5}$$

- $\ell_i(\theta)$: standard cross-entropy on group $G_i$.
- $\ell_i(\theta + \epsilon_i)$: loss evaluated at perturbed weights — what the model would look like after a worst-case small fine-tuning step aimed at group $G_i$.
- $\lambda \in [0,1]$: interpolation weight, increased from $0$ to $1$ via **curriculum learning** during training so the model first finds a valid safe solution, then hardens it.
- The bracketed term $\ell_i(\theta + \epsilon_i) - \ell_i(\theta)$: the **fragility** of the model on group $i$ — how much the loss rises under the worst perturbation. Minimising $f_i$ forces this fragility to zero.

---

### 3.3 The Worst-Case Perturbation (SAM Trick)

The ideal perturbation for group $i$ is:

$$\epsilon_i^* = \arg\max_{\|\epsilon\| \leq \alpha} \ell_i(\theta + \epsilon) \tag{inner max}$$

This inner maximisation is intractable exactly. Apply a first-order Taylor expansion:

$$\ell_i(\theta + \epsilon) \approx \ell_i(\theta) + \epsilon^\top \nabla_\theta \ell_i(\theta)$$

Maximising over $\epsilon$ with $\|\epsilon\| \leq \alpha$ under the Cauchy–Schwarz inequality gives:

$$\boxed{\epsilon_i^* = \alpha \cdot \frac{\nabla_\theta \ell_i(\theta)}{\|\nabla_\theta \ell_i(\theta)\|}} \tag{3, adapted from SAM}$$

This is the normalised gradient direction, scaled to the perturbation budget $\alpha$. It requires exactly one additional forward-backward pass per group per training step. The key intuition is: **the worst parameter perturbation points uphill in loss, and the gradient tells you where uphill is.** For vulnerable examples (sharper loss landscape), $\|\nabla_\theta \ell_i(\theta)\|$ is larger in magnitude, so $\epsilon_i^*$ is more consequential — the robust objective on $G_2$ is harder to satisfy, forcing the model to genuinely flatten the loss surface there.

---

### 3.4 The Group DRO Objective (The Adversary's Problem)

Instead of fixing the sampling distribution over groups, VAA learns it. The minimax objective is:

$$\hat{\theta}_\text{DRO} = \arg\min_{\theta \in \Theta} \left\{\sup_{G_i \in \mathcal{Q}}\; \mathbb{E}_{(x,y) \sim G_i}\bigl[f_i(\theta; (x,y))\bigr]\right\} \tag{7}$$

where the ambiguity set $\mathcal{Q}$ is the set of all convex combinations of the two groups:

$$\mathcal{Q} := \left\{ \sum_i q_i G_i \;\middle|\; q \in \Delta^{m-1} \right\} \tag{8}$$

$q \in \Delta^{m-1}$ is a probability simplex over the $m$ groups (here $m=2$). The $\sup$ selects the hardest group mixture. In the two-group case this collapses to: whichever of $G_1, G_2$ the current model is losing on more gets all the sampling mass.

---

### 3.5 The Mirror Ascent Update (Solving the Adversary's Problem Online)

The adversary updates $q$ by mirror ascent on $\Delta^{m-1}$:

$$q^{(t)} = \arg\max_{q \in \Delta^{m-1}} \left\{ \eta_q \langle q, f^{(t)} \rangle - D_\psi(q \,\|\, q^{(t-1)}) \right\} \tag{9}$$

where:
- $f^{(t)} = \bigl(f_1(\theta^{(t-1)}), \ldots, f_m(\theta^{(t-1)})\bigr)^\top$ — the vector of current per-group losses.
- $\eta_q > 0$ — step size for the sampler.
- $D_\psi$ — Bregman divergence from mirror map $\psi$.

Choosing $\psi(q) = \sum_i q_i \log q_i$ (negative entropy) induces $D_\psi = D_\text{KL}$:

$$q^{(t)} = \arg\min_{q \in \Delta^{m-1}} \left\{ \eta_q \sum_i q_i f_i + \sum_i q_i \log\frac{q_i}{q_i^{(t-1)}} \right\} \tag{10}$$

Introducing a Lagrange multiplier $\lambda$ for the simplex constraint and solving $\partial \mathcal{L}/\partial q_i = 0$:

$$q_i^{(t)} = \frac{q_i^{(t-1)} \exp(\eta_q f_i^{(t)})}{\sum_j q_j^{(t-1)} \exp(\eta_q f_j^{(t)})} \tag{12}$$

This is the **EXP3 update rule** (Exponential weights for Exploration and Exploitation): groups with higher current loss get exponentially more sampling mass. At convergence, if the model has learned all groups equally well, $q$ converges to a uniform distribution — the adversary has no incentive to prefer one group over another.

---

### 3.6 The Full Training Loop

At each iteration $t$:

1. **Sample group.** Draw $G_i \sim q^{(t-1)}$ (biased toward the currently harder group).
2. **Compute perturbation.** Forward-backward on $\ell_i(\theta)$ to get $\nabla_\theta \ell_i(\theta)$, then $\epsilon_i^* = \alpha \cdot \nabla_\theta \ell_i / \|\nabla_\theta \ell_i\|$.
3. **Evaluate robust loss.** Compute $f_i(\theta) = (1-\lambda)\ell_i(\theta) + \lambda\ell_i(\theta + \epsilon_i^*)$.
4. **Update model.** Gradient descent on $f_i$: $\theta \leftarrow \theta - \eta_\theta \nabla_\theta f_i(\theta)$.
5. **Update sampler.** Compute reward $r_i = f_i(\theta^{(t)})$ and apply EXP3 update (eq. 12).
6. Increment $\lambda$ according to the curriculum schedule.

Total cost: approximately **1.5× backpropagation passes** compared to vanilla SFT (one for the clean loss and one partial pass for the perturbation), versus 2× for Vaccine and 3× for Booster.

---

## 4. Assessment Against the Four Immunisation Properties

### Resistance ✅ (primary claim)

VAA's explicit goal is to make the alignment harder to undo by HFT. Empirically, harmful scores stay significantly lower across all tested HFT task configurations, including up to 20% harmful data and 5 fine-tuning epochs. The curriculum + perturbation mechanism targets the mechanistic root cause (sharp, fragile safety basin) rather than patching the symptom. **However**: VAA offers no theoretical guarantee (no bound on the attacker's required budget). It is strictly a *weak resistance* demonstration.

### Stability ✅ (well-verified)

Fine-tuning accuracy on the downstream task (SST-2, AG News, GSM8K, AlpacaEval) is maintained or improved relative to baselines. The curriculum learning strategy (starting from standard ERM, gradually raising $\lambda$) is critical here: it prevents the perturbation from disrupting normal learning in the early phase.

### Generalisation ✅ (partially verified)

VAA is tested across four qualitatively different downstream fine-tuning tasks. The group labels estimated on one proxy task transfer across these tasks — the paper shows this in Figure 2. Cross-model transfer (LLaMA-2 labels → Qwen2.5) also holds. However, **cross-domain generalisation in the immunisation sense** — e.g., defence trained on toxic-text attacks resisting harmful QA attacks — is not directly probed. The paper implicitly assumes a fixed harmful data distribution (BeaverTails) and does not vary the nature of the harmful content systematically.

### Trainability ⚠️ (the missing piece)

VAA preserves *accuracy on the fine-tuning task*, but this is not the same as trainability in Rosati et al.'s formal sense: the immunised model should be fine-tunable on *benign* tasks at comparable cost to the non-immunised model. VAA does not directly measure this. There is a legitimate concern that the flattened, widened safety basin could also slow benign fine-tuning convergence, because the robust objective effectively makes the optimisation landscape harder to navigate in any direction. This is the most underexplored dimension of VAA.

---

## 5. Mechanistic Commonalities with Other Approaches

### The shared gradient-ascent trick for perturbation

The core mathematical move in VAA — computing $\epsilon^* = \alpha \cdot \nabla\ell / \|\nabla\ell\|$ as a proxy for the worst-case parameter perturbation — is not unique to VAA. It is the same first-order SAM approximation used by:

- **Vaccine**: computes a similar worst-case *embedding* perturbation $\epsilon_{l,t} = \rho \cdot \nabla_{e_{l,t}} \mathcal{L} / \|\nabla \mathcal{L}\|$ per layer in the embedding space. VAA moves the same logic from embedding space to parameter (weight) space.
- **TAR**: uses a gradient-based estimate of how many gradient steps the attacker needs to recover from the defence, and maximises that cost. The gradient direction again serves as a proxy for attack direction.
- **SDD / Condition-Number methods**: approximate the Hessian to reason about loss landscape curvature. VAA avoids the Hessian entirely but achieves a related effect (flattening the basin) via the perturbation term in the robust objective.

The key differentiator VAA introduces is **group-specificity**: the perturbation budget $\alpha$, and the frequency with which each perturbation is applied, are both a function of which group is currently hard. Vaccine applies a uniform perturbation across all data; VAA concentrates the attack budget where it actually matters.

### The minimax / two-player structure

VAA, TAR, and Antidote all frame immunisation as a minimax game. The difference is in *who the adversary is*:
- **TAR**: the adversary is a simulated attacker that runs actual gradient-descent fine-tuning steps.
- **Antidote**: the adversary is a hypernetwork that learns to produce malicious LoRA patches.
- **VAA**: the adversary is a much simpler object — a probability vector $q$ over two pre-fixed data groups. This is a drastically cheaper inner problem, but it means VAA's adversary cannot adapt to the *content* of the attack, only to *which pre-labelled subset* of the alignment data is currently weakest.

### Curriculum learning for $\lambda$

The gradual ramp of $\lambda$ from 0 to 1 is a stabilisation technique also seen in the robust fine-tuning literature (e.g., sharpness-aware optimisation schedules). It prevents the model from getting stuck in a bad local minimum early in training, allowing it to first find a good alignment solution and then harden it. This is a simple but important practical contribution.

---

## 6. Results Summary and Significance

### Experimental setup

- **Models**: LLaMA-2-7B (primary), Qwen2.5-7B (cross-model transfer).
- **Alignment data**: 2,000 samples from BeaverTails.
- **HFT tasks**: SST-2, AG News, GSM8K, AlpacaEval, with 10% harmful data (ablations at 5% and 20%).
- **Metrics**: Harmful Score (HS, lower is better) and Fine-tuning Accuracy (FA, higher is better).

### Results

VAA consistently and substantially outperforms all four baselines (SFT, Vaccine, RepNoise, Booster) on HS, while matching or exceeding FA across all four tasks:

- On the hardest tasks (GSM8K, AlpacaEval), RepNoise and Booster *increase* HS relative to SFT — they break the utility-safety trade-off. VAA remains the only baseline that reduces HS while preserving FA on these tasks.
- At higher harmful data fractions (20%), VAA's advantage grows, suggesting the mechanism is genuinely addressing the fragility root cause rather than exploiting easy wins.
- At 5 HFT epochs on Qwen2.5-7B, VAA achieves HS ≈ 22 vs. Booster's 30, SFT's 33 — a relative improvement of roughly one third.

### Significance relative to the tutorial's landscape

VAA is one of the few alignment-stage methods that maintains competitiveness across *all four downstream tasks*, not just easy classification tasks. The cross-model and cross-task transfer of vulnerability labels is a strong empirical result: it suggests that the vulnerability structure is real and tractable, not an artefact of the specific experimental setup.

The main caveat: **Antidote is absent from the comparison**. Antidote is the most credible baseline — it appeared at almost the same time, addresses exactly the same threat model, and uses a richer bi-level formulation. Without that comparison, VAA's claim to state-of-the-art is incomplete.

---

## 7. Future Work

### Called for by the authors

1. **Extending the vulnerability analysis to benign fine-tuning.** The paper shows that even purely benign user fine-tuning can degrade alignment. VAA was designed for the mixed harmful + benign case; extending the vulnerability profiling and the GDRO framework to the purely-benign threat model is an open problem.
2. **Dynamic group membership.** The current implementation fixes the two groups before alignment training. A dynamic scheme that re-assesses vulnerability during training (as the model's landscape changes) could improve coverage.
3. **Scaling to larger models.** All experiments are on 7B-scale models. Whether the vulnerability transfer patterns hold at 70B+ (where full-parameter training is impractical) is unknown.
4. **PEFT compatibility.** Alignment is increasingly done via LoRA. Applying group-specific perturbations in LoRA adapter space rather than full parameter space is not explored.

### According to the state of the art

5. **Formal resistance bounds.** VAA lacks any theoretical guarantee. The immunisation definition framework (Rosati et al.) calls for bounds on the attacker's required training budget. Connecting the landscape-flattening achieved by VAA to such bounds — possibly via PAC-Bayesian flatness results — is a natural extension.
6. **Evaluation against stronger attackers.** VAA is tested against standard SFT-based HFT. Recent work (RL-based harmful fine-tuning, activation steering attacks) shows that SFT is not the strongest attacker. Resistance against RL-based de-alignment or inference-time intervention attacks is entirely untested.
7. **Comparison with Antidote, LoX, and constraint-based methods** (SDD, C-TRAP). VAA was published in a rapidly evolving landscape; a controlled head-to-head evaluation against these contemporaneous methods is the most pressing empirical gap.
8. **The trainability question.** Whether the widened safety basin meaningfully slows downstream benign fine-tuning — and how to measure and mitigate this — is both practically important and formally open.

---

*Notes file for the LLM Immunisation Tutorial — Part 2: Representation Engineering and Alignment-Stage Defences.*
