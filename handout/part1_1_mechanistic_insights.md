# Introduction

## Part 1.1 — Mechanistic Insights: What Alignment and Unlearning Actually Do to a Model

> *This part is about diagnosis. Before we can argue that standard defences are insufficient, we need to understand, at a mathematical level, what they are actually doing to the weights. The picture that emerges is surprisingly coherent and, in hindsight, explains exactly why every brittle defence is brittle.*

---

### 1.1.1 Setting Up the Notation

Let $f_\theta$ be a language model with parameters $\theta := \{W^i\}_{i=1}^{L}$, a family of real matrices. We will track three checkpoints:

| Symbol | Meaning |
|---|---|
| $\theta_{\text{base}}$ | Pre-trained weights, no safety tuning |
| $\theta_{\text{align}} = \theta_{\text{base}} + \Delta W_{\text{align}}$ | After safety alignment (RLHF, DPO, SFT on refusals, …) |
| $\theta_{\text{ft}} = \theta_{\text{align}} + \Delta W_{\text{ft}}$ | After a downstream fine-tuning step |

The central question of this section is: **what is $\Delta W_{\text{align}}$, geometrically?** Is it a large, distributed perturbation that deeply rewires the model's internal representations, or is it something far more parsimonious — and therefore far more fragile?

The answer, across five converging lines of empirical and theoretical evidence, is the latter.

---

### 1.1.2 Insight 1 — Alignment Is Shallow: Most of the KL Budget Lives in the First Token

The most direct way to ask "what did alignment change?" is to compare the output distribution of $\theta_{\text{align}}$ and $\theta_{\text{base}}$ token by token, on harmful queries. Qi et al. (2024) do exactly this in *Safety Alignment Should Be Made More Than Just a Few Tokens Deep*.

For a harmful query $\mathbf{x}$ and a generated sequence $\mathbf{y} = [y_1, y_2, \ldots, y_T]$, define the **per-token KL divergence** between the aligned and base models at position $t$ as:

$$\delta_t(\mathbf{x}) \coloneqq \mathrm{KL}\!\left(p_{\theta_{\text{align}}}(\cdot \mid \mathbf{x}, \mathbf{y}_{<t}) \;\|\; p_{\theta_{\text{base}}}(\cdot \mid \mathbf{x}, \mathbf{y}_{<t})\right).$$

If alignment were a deep, distributed change, we would expect $\delta_t$ to be spread roughly evenly across all positions $t$. What the paper finds is the opposite: $\delta_1 \gg \delta_t$ for $t > 1$. The bulk of the KL "budget" that alignment adds is spent on position $t=1$ — the very first token of the response.

This is the **safety shortcut**: the alignment procedure learns to place overwhelming probability mass on refusal-initiating tokens (e.g., "I", "Sorry", "I cannot…") at position 1. Once that first token is generated, the model's continuation distribution is barely different from the base model's. The implication is stark: if an attacker can suppress or bypass the first token, the model's "safe" appearance collapses.

> **Mechanistic summary.** Standard alignment does not rewire the model's ability to produce harmful content. It adds a thin, high-confidence gate at the first decoding step. The harmful knowledge is preserved underneath.

---

### 1.1.3 Insight 2 — Alignment Lives in a Low-Rank Subspace of the Weight Matrices

Zooming from the output distribution into the weight matrices themselves, Wei et al. (2024) in *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications* ask: is the weight change $\Delta W_{\text{align}}$ spread evenly across the spectrum of each matrix, or is it concentrated in a few dominant singular directions?

Take the SVD of the alignment update for a single weight matrix:

$$\Delta W_{\text{align}} = U S V^\top = \sum_{i=1}^{r} s_{ii}\, U_i V_i^\top,$$

where the singular values $s_{11} \geq s_{22} \geq \cdots \geq s_{rr} \geq 0$ are sorted in decreasing order. Wei et al. find that the safety-relevant information is almost entirely concentrated in the **top-$k$ singular directions** — and $k$ is very small relative to the rank $r$ of the full matrix. In some cases, the refusal behaviour of the whole model can be mediated by a **single rank-1 component** (Arditi et al., 2024, cited therein).

This motivates a surgical attack: identify the top-$k$ singular directions of $\Delta W_{\text{align}}$, remove them from the model weights, and the model reverts to near-base behaviour — without any measurable degradation in general language modelling performance. Formally, if we define

$$W_{\text{attack}} \coloneqq W_{\text{base}} + \Delta W_{\text{align}} - \text{Proj}_k(\Delta W_{\text{align}}),$$

where

$$\text{Proj}_k(M) \coloneqq (U_{:k} U_{:k}^\top)\, M,$$

then $W_{\text{attack}}$ has had the top-$k$ alignment directions removed. The attack success rate jumps back toward the base model's level with $k$ as small as a handful of singular vectors.

This result has an important algebraic footnote worth stating clearly: $\text{Proj}_k$ is indeed a projection because $(U_{:k}U_{:k}^\top)^2 = U_{:k}(U_{:k}^\top U_{:k})U_{:k}^\top = U_{:k} I_k U_{:k}^\top = U_{:k}U_{:k}^\top$ — the middle step uses the fact that $U$'s columns are orthonormal. So it is an idempotent operator, and $\text{Proj}_k(\Delta W_{\text{align}})$ gives exactly the component of the alignment update that "lives in" the top-$k$ safety subspace.

> **Mechanistic summary.** The alignment update $\Delta W_{\text{align}}$ is low-rank in practice. Safety is not a diffuse property of the whole parameter tensor; it is a sparse directional feature that a targeted perturbation can nullify.

---

### 1.1.4 Insight 3 — The Safety Basin and Its Geometry

Peng et al. (2024) in *Navigating the Safety Landscape* give this a loss-landscape language that is geometrically illuminating. They define the **safety basin** as the connected region in $\theta$-space where the model's harmful-content loss exceeds a threshold — i.e., where the model reliably refuses. Formally, for a harmful benchmark $\mathcal{D}_{\text{harm}}$ and a threshold $\epsilon$:

$$\mathcal{B}_\epsilon(\theta_{\text{align}}) \coloneqq \left\{ \theta : \mathcal{L}_{\text{harm}}(\theta; \mathcal{D}_{\text{harm}}) \geq \epsilon \right\},$$

where $\mathcal{L}_{\text{harm}}$ is the loss on harmful completions (high loss = model refuses). The aligned model $\theta_{\text{align}}$ sits inside this basin. Standard harmful fine-tuning is, in this language, a gradient step that moves $\theta$ **out** of $\mathcal{B}_\epsilon$.

What makes this geometric picture clinically useful is the **shape** of the basin near $\theta_{\text{align}}$. Peng et al. find that the standard aligned model sits in a **narrow valley** of the safety loss landscape: the Hessian of $\mathcal{L}_{\text{harm}}$ at $\theta_{\text{align}}$ has a large condition number, meaning the landscape is steep in some directions and flat in others. Even small parameter perturbations along the steep directions can push the model outside the basin entirely.

This is not a bug — it is the direct consequence of Insights 1 and 2. Because alignment is concentrated in a few singular directions, the safety loss is very sensitive to changes in exactly those directions and insensitive to everything else. The "safety valley" is narrow precisely because the safety signal is parsimonious.

Separately, Peng et al. observe that the concept of "uneven forgetting" compounds the geometric problem: **not all alignment examples are equally close to the basin boundary**. Some alignment examples correspond to regions where the model is only barely inside the basin (high sensitivity to parameter shifts); others are deeply inside it. Harmful fine-tuning preferentially pushes the model out along the directions that the "vulnerable" examples occupy — the ones already close to the boundary. The distribution of examples inside the basin is non-uniform, and this non-uniformity is what makes even benign fine-tuning corrosive.

> **Mechanistic summary.** The aligned model lives in a narrow, anisotropic safety basin. The alignment gradient points in the same low-rank subspace identified in Insight 2, so any fine-tuning that perturbs those directions is disproportionately harmful.

---

### 1.1.5 Insight 4 — What Unlearning Actually Does to Weights: A Task-Vector Autopsy

Machine unlearning is sometimes presented as a stronger alternative to alignment: instead of just teaching the model to refuse, unlearning attempts to erase the underlying harmful knowledge from the weights. The most common approaches define a **forget set** $\mathcal{D}_f$ and a **retain set** $\mathcal{D}_r$ and solve:

$$\min_\theta \quad \mathcal{L}_u(\theta; \mathcal{D}_f, \mathcal{D}_r) \coloneqq \mathcal{L}_f(\theta; \mathcal{D}_f) + \gamma\, \mathcal{L}_r(\theta; \mathcal{D}_r),$$

where $\mathcal{L}_f$ is a forget objective (e.g., Negative Preference Optimisation, NPO; or Representation Misdirection for Unlearning, RMU) and $\mathcal{L}_r$ is a standard cross-entropy retain loss.

Łucki et al. (2024) in *An Adversarial Perspective on Machine Unlearning for AI Safety* then ask: does this make the model more robustly safe against subsequent fine-tuning? The answer is no — and the reason can be seen cleanly through **task vectors**.

Define the **unlearning direction** as the task vector induced by unlearning:

$$\tau_u \coloneqq \theta_u - \theta_o,$$

where $\theta_u$ is the unlearned model and $\theta_o$ is the original (base + aligned) model. Similarly, define the **fine-tuning direction** as the task vector of a downstream fine-tuning step (even a benign one, e.g., fine-tuning on GSM8K):

$$\tau_{\text{ft}} \coloneqq \theta_{\text{ft}} - \theta_o.$$

For unlearning to be robust to downstream fine-tuning, we want $\tau_u$ to be preserved after fine-tuning — i.e., we want the **post-fine-tuning unlearning direction**

$$\tau_{u \to \text{ft}} \coloneqq \theta_u^{\text{ft}} - \theta_u$$

to be as orthogonal as possible to $\tau_{\text{ft}}$, and not to cancel $\tau_u$ itself. The diagnostic quantities are:

$$\cos\!\left(\angle(\tau_{u \to \text{ft}},\, \tau_{\text{ft}})\right) \quad \text{and} \quad \cos\!\left(\angle(\tau_{u \to \text{ft}},\, \tau_u)\right).$$

For NPO-based unlearning, Łucki et al. measure:

$$\cos\!\left(\angle(\tau_{\text{NPO} \to \text{ft}},\, \tau_{\text{ft}})\right) = 0.16 > 0,$$

meaning the post-fine-tuning correction vector $\tau_{u \to \text{ft}}$ points in a direction that is *co-aligned* with the fine-tuning direction — downstream fine-tuning pulls the model back toward the forget set, because the unlearning update happens to live in the same subspace that fine-tuning explores. The unlearning is not "fixed in place"; it is precisely the part of the weight space that routine gradient descent finds first.

This co-alignment is not an accident. It follows from the same low-rank structure of Insight 2: both the unlearning update $\tau_u$ and the fine-tuning update $\tau_{\text{ft}}$ are concentrated in the high-variance singular subspaces of the weight matrices. These subspaces overlap heavily, so a fine-tuning step that "wants" to move weights in a particular direction will inevitably project onto the unlearning direction and corrupt it.

Compare this with a robust unlearning method (ILU, from the same paper), which achieves:

$$\cos\!\left(\angle(\tau_{\text{ILU} \to \text{ft}},\, \tau_{\text{ILU}})\right) = 0.09 \approx 0,$$

i.e., near-orthogonality between the post-fine-tuning drift and the original unlearning direction. ILU achieves this by enforcing an **invariance regularisation** that explicitly penalises correlation between the unlearning gradient and fine-tuning directions across multiple environments. The near-zero cosine similarity is the geometric certificate that the unlearning direction is preserved.

> **Mechanistic summary.** Standard unlearning methods write their safety information into the same dominant singular subspaces that are also the most malleable under fine-tuning. This is the fundamental conflict: the directions that are easiest to modify during unlearning are the same directions that are easiest to corrupt during downstream adaptation.

---

### 1.1.6 Insight 5 — Safety Degradation Is Task-Dependent: The Fine-Tuning Task Vector Matters

A natural question after Insights 1–4 is: is all fine-tuning equally destructive to safety, or does it depend on what the fine-tuning task is? Hsiung et al. (2025) in *Your Task May Vary: A Systematic Understanding of Alignment and Safety Degradation when Fine-Tuning LLMs* show that it is emphatically the latter.

The key conceptual tool is again the task vector. Define the **safety direction** of the alignment update as (dropping the matrix index for clarity):

$$\tau_{\text{safety}} \coloneqq \theta_{\text{align}} - \theta_{\text{base}}.$$

For a given downstream fine-tuning task $T$ with task vector $\tau_T = \theta_{\text{ft}}^T - \theta_{\text{align}}$, define the **safety-task alignment angle**:

$$\phi_T \coloneqq \angle(\tau_T,\; -\tau_{\text{safety}}).$$

When $\cos(\phi_T) > 0$, the fine-tuning task vector points in the anti-safety direction — the fine-tuning is, in a precise geometric sense, undoing the alignment. When $\cos(\phi_T) \approx 0$, the fine-tuning is roughly orthogonal to safety and causes little degradation.

Hsiung et al. find that this angle is a strong predictor of how much safety degrades after fine-tuning. Tasks like sentiment analysis or mathematics (GSM8K) produce fine-tuning vectors that are nearly orthogonal to the safety direction, causing modest degradation. Tasks that are stylistically or semantically closer to instruction-following on harmful content produce fine-tuning vectors with larger projections onto the anti-safety direction, causing severe degradation — even when no harmful examples are present in the fine-tuning data.

This has a sobering implication: **you cannot assess safety robustness by evaluating on a single task**. A model that survives fine-tuning on mathematics may not survive fine-tuning on a creative-writing corpus, simply because the latter's task vector happens to have a larger component along $-\tau_{\text{safety}}$.

> **Mechanistic summary.** The degradation of safety under fine-tuning is not merely a function of whether harmful data is present. It is a function of the geometric relationship between the downstream task vector and the safety subspace. Tasks whose gradient directions intersect the safety subspace more deeply will cause more degradation, regardless of their surface-level harmlessness.

---

### 1.1.7 The Unified Picture: Alignment as a Sparse, Shallow, Localisable Modification

Putting all five insights together, we can now draw a coherent picture of what safety alignment and unlearning *are*, mechanistically.

**In output space (Insight 1):** alignment is a high-confidence gate on the first decoding step. The rest of the generation is largely unchanged relative to the base model.

**In weight space (Insights 2 and 3):** alignment corresponds to a small number of dominant singular directions in $\Delta W_{\text{align}}$. The safety basin is anisotropic: it is narrow in the directions that alignment modifies and wide in the directions it does not.

**Under perturbation (Insights 4 and 5):** downstream fine-tuning — benign or harmful — explores the same dominant subspaces that alignment writes into. When the fine-tuning task vector has a large projection onto the anti-safety direction, it erases the alignment update just as gradient descent would erase any low-signal direction it passes through. Unlearning methods that do not explicitly enforce orthogonality between the unlearning direction and the space of plausible fine-tuning updates are guaranteed to degrade under subsequent adaptation.

There is a useful analogy here. Standard alignment is like writing a message in pencil on the top page of a pad, then handing the pad to an adversary. The message is there. But the adversary doesn't need to read it; they just need to erase it — and the pencil mark is thin, localised, and easily overwritten. What immunisation will ask us to build, as we will see in the sections to come, is something more akin to writing in ink that chemically bonds to the paper: not merely making the message harder to read, but making erasure itself structurally costly or impossible.

The next section demonstrates precisely how brittle the pencil-mark defences are in practice, motivating the need for a fundamentally different approach.

---

### References for Part 1.1

- Wei, B., Huang, K., Huang, Y., Xie, T., Qi, X., Xia, M., Mittal, P., Wang, M., and Henderson, P. **Assessing the brittleness of safety alignment via pruning and low-rank modifications.** ICML 2024.
- Łucki, J., Wei, B., Huang, Y., Henderson, P., Tramèr, F., and Rando, J. **An adversarial perspective on machine unlearning for AI safety.** arXiv:2409.18025, 2024.
- Peng, S., Chen, P.-Y., Hull, M., and Chau, D. H. **Navigating the safety landscape: Measuring risks in finetuning large language models.** NeurIPS 2024.
- Qi, X., Panda, A., Lyu, K., Ma, X., Roy, S., Beirami, A., Mittal, P., and Henderson, P. **Safety alignment should be made more than just a few tokens deep.** arXiv:2406.05946, 2024.
- Hsiung, L., Pang, T., Tang, Y.-C., Song, L., Ho, T.-Y., Chen, P.-Y., and Yang, Y. **Your task may vary: A systematic understanding of alignment and safety degradation when fine-tuning LLMs.** ICLR 2025.
