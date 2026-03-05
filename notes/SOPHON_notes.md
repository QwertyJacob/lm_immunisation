# SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability for Pre-trained Models

> Deng, Pang, Chen, Xia, Bai, Weng, Xu — Zhejiang University / Ant Group  
> arXiv:2404.12699, April 2024 · Accepted at IEEE S&P 2024

---

## 1. Quick Mechanistic Summary

SOPHON is a **pre-release immunisation framework** for deep learning models. Given a pre-trained model $f_0$ with parameters $\theta_0$, SOPHON modifies $\theta_0$ into a protected version $\theta$ such that:

- **Intactness**: $f_\theta$ performs on par with $f_0$ on the original (source) domain $\mathcal{D}_S$.  
- **Non-transferability**: $f_\theta$ performs poorly on the restricted domain $\mathcal{D}_A$ even without fine-tuning.  
- **Non-fine-tunability**: even after an adversary performs unconstrained fine-tuning on $\mathcal{D}_A$, $\phi(f_\theta)$ still performs poorly — ideally no better than training from scratch.

The mechanism is a **bi-level meta-learning loop** with two alternating phases. The **fine-tuning suppression (FTS) phase** simulates plausible adversarial fine-tuning strategies in an inner loop (à la MAML), then updates $\theta$ in the outer loop to *maximise* the loss of the fine-tuned model on the restricted domain. The **normal training reinforcement (NTR) phase** simultaneously applies standard gradient descent on the original domain to prevent intactness collapse.

The central insight is that the adversary's fine-tuning procedure, although opaque, can be *simulated* — and if the model's parameters are pushed into a **hard-to-escape local optimum** of the restricted-domain loss landscape, any subsequent fine-tuning by the adversary will be as costly or costlier than training from scratch.

Novel to SOPHON are three purpose-designed loss functions — **Inverse Cross-Entropy (ICE)**, **KL Divergence from Uniform Distribution (KLU)**, and **Denial of Service (DoS)** — engineered to produce stable gradients during the fine-tuning suppression step, where standard cross-entropy or MSE diverge.

---

## 2. Timeline Positioning: What SOPHON Inherits and What It Adds

### The direct lineage

| Year | Paper | Role in lineage |
|------|-------|-----------------|
| ICLR 2022 | **NTL** (Wang et al.) | Degrades model performance on a restricted domain *before* fine-tuning, using MMD-based representation divergence. Does not model fine-tuning at all; the protection collapses under any adversarial SFT. |
| 2022 | **MLAC/SDM** (Henderson et al.) | First to frame immunisation as meta-learning. Inner loop: adapt model head toward harmful task. Outer loop: maximise the harmful loss (gradient ascent on CE). First-order MAML approximation. |
| Apr 2024 | **SOPHON** | Synthesises NTL's *domain-specificity* with MLAC's *meta-learning inner loop*. Extends to generative models (diffusion). Introduces stable loss functions (ICE, KLU, DoS). |
| ICLR 2025 | **TAR** (Henderson et al.) | Brings the same meta-learning paradigm to LLMs. Key refinement: replaces the CE outer loop loss with an **entropy loss** to prevent adversary recovery in later inner-loop steps — a problem SOPHON's ICE/KLU partially address at the loss function level, but TAR diagnoses more sharply. |

### What SOPHON inherits

- **From NTL**: the concept of a *restricted domain* to suppress, and the dual objective of suppressing target-domain performance while preserving source-domain performance.
- **From MAML** (Finn et al., 2017): the bi-level structure where an inner loop simulates a learning agent and the outer loop optimises the *outcome* of that inner loop. SOPHON uses first-order MAML approximation to avoid second-order gradient computation.
- **From adversarial training broadly**: the idea that robustness is earned by simulating the adversary during training.

### SOPHON's unique contributions

1. **Non-fine-tunability as a distinct objective** — NTL ignores fine-tuning entirely; SOPHON is the first work to formalise it as a separate property and optimise for it directly.
2. **Multi-strategy inner-loop simulation** — SOPHON samples $N$ different fine-tuning strategies $\varphi_i \sim \Phi$ (varying initialisation, learning rates, batch sizes) in each outer step, unlike MLAC which fixes the adversary's strategy. This substantially improves generalisation of the protection.
3. **Stable loss functions for gradient ascent** — ICE, KLU, and DoS are derived to have *decreasing* gradient magnitudes as the error grows, which stabilises the outer loop (cross-entropy and MSE diverge in this setting).
4. **Extension to generative models** — all prior immunisation work was classification-only; SOPHON applies to diffusion probabilistic models via the DoS loss.

---

## 3. The Math

### 3.1 Problem Formulation

The **basic (constrained) formulation** is:

$$
\min_{\theta} \; - \mathbb{E}_{x \sim \mathcal{D}_A,\, \varphi \sim \Phi} \; \mathcal{L}\!\left(\varphi(f_\theta(x))\right)
$$
$$
\text{s.t.} \quad \mathbb{E}_{x \sim \mathcal{D}_S} \left[\max\!\left\{0,\; \mathcal{L}(f_\theta(x)) - \mathcal{L}(f_0(x))\right\}\right] < \lambda \tag{1}
$$

The objective maximises the adversary's expected loss *after fine-tuning* on the restricted domain $\mathcal{D}_A$, across the distribution of fine-tuning strategies $\Phi$. The constraint bounds performance degradation on the original domain $\mathcal{D}_S$.

The constraint is difficult to enforce directly. SOPHON converts this to an **unconstrained Lagrangian form**:

$$
\min_{\theta} \; \underbrace{- \mathbb{E}_{x \sim \mathcal{D}_A,\, \varphi \sim \Phi} \; \mathcal{L}\!\left(\varphi(f_\theta(x))\right)}_{\text{fine-tuning suppression}} \;+\; \mu \cdot \underbrace{\mathbb{E}_{x \sim \mathcal{D}_S} \; \mathcal{L}(f_\theta(x))}_{\text{normal training reinforcement}} \tag{2}
$$

where $\mu > 0$ is a Lagrange multiplier balancing the two goals.

### 3.2 The Inner Loop: Fine-Tuning Simulation

The challenge: $\varphi(f_\theta(x))$ has no closed form — fine-tuning is itself an iterative optimisation. SOPHON approximates it by *running* the inner loop.

A fine-tuning environment is a triplet $(\varphi_i, R_i, T_i)$ where $\varphi_i$ is a fine-tuning strategy, $R_i$ is the training set, and $T_i$ is the evaluation set — both drawn from $\mathcal{D}_A$. At the $k$-th inner step:

$$
f_\vartheta^k = \arg\max_\vartheta \; \mathbb{E}_{x \sim R_i} \; \mathcal{L}\!\left(f_\vartheta^k\!\left(x \mid \varphi_i, f_\vartheta^{k-1}\right)\right) \tag{3}
$$

with $f_\vartheta^0 = f_\theta$ (the current protected model is the starting point). This is approximated by $K$ gradient ascent steps in practice ($K = 50$ in experiments).

### 3.3 The Outer Loop: Fine-Tuning Suppression Loss

SOPHON aggregates the post-fine-tuning performance across all simulated strategies and all inner steps to form the **Fine-Tuning Suppression loss**:

$$
\mathcal{L}_{\text{FTS}} = \sum_{i=1}^{N} \sum_{k=1}^{K} \gamma_{i,k} \cdot \mathcal{L}_\alpha\!\left(f_\vartheta^k \mid \varphi_i,\, T_i\right) \tag{4}
$$

where $\gamma_{i,k}$ are weights over fine-tuning strategies and inner-loop steps, and $\mathcal{L}_\alpha$ is a task-specific loss function (chosen from ICE, KLU, or DoS — see §3.5).

The outer-loop parameter update is:

$$
\theta \leftarrow \theta \pm \alpha \cdot \nabla_\theta \mathcal{L}_{\text{FTS}} \tag{5}
$$

The **sign** depends on the loss design: ICE and KLU are constructed so that minimising them *increases* error in the restricted domain, so the sign is $-$ (standard gradient descent). The second-order terms from differentiating through the inner loop are expensive; SOPHON uses the **first-order MAML approximation** (straight-through, treating $f_\vartheta^k$ as if it were computed with detached gradients).

### 3.4 The Normal Training Reinforcement Loss

To prevent intactness collapse, SOPHON interleaves gradient descent on the original domain:

$$
\mathcal{L}_{\text{NTR}} = \mathcal{L}_\beta(f_\theta \mid O), \quad O \sim \mathcal{D}_S \tag{6}
$$
$$
\theta \leftarrow \theta - \beta \cdot \nabla_\theta \mathcal{L}_{\text{NTR}} \tag{7}
$$

Both updates use Adam. The outer loop alternates $\ell_{\text{FTS}}$ FTS steps and $\ell_{\text{NTR}}$ NTR steps for `Iter` total iterations.

### 3.5 The Loss Functions: Why Cross-Entropy Fails Here

The gradient of the standard cross-entropy loss with respect to logit $z_i$ is:

$$
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_i} = \hat{y}_i - y_i \tag{10}
$$

The gradient magnitude is *positively correlated with error*. In the FTS setting, we want error to grow; this means gradients grow too — leading to divergence. SOPHON solves this by designing losses whose gradient magnitude *decreases as error increases*.

**Inverse Cross-Entropy (ICE)** — for classification with labelled restricted data:

$$
\mathcal{L}_{\text{ICE}}(f) = -\frac{1}{|\mathcal{X}|} \sum_i \sum_{j=1}^{C} y_{ij} \log(1 - \hat{y}_{ij}) \tag{11}
$$

Its gradient (assuming $y_1 = 1$ without loss of generality):

$$
\frac{\partial \mathcal{L}_{\text{ICE}}}{\partial z_i} = \begin{cases} \hat{y}_1 & i = 1 \\ -\hat{y}_i \hat{y}_1 / (1 - \hat{y}_1) & i \neq 1 \end{cases} \tag{12}
$$

As the true-class probability $\hat{y}_1 \to 0$ (i.e., the model becomes wrong on the restricted task), the gradient $\to 0$. Stable.

**KL Divergence from Uniform Distribution (KLU)** — for classification without labels:

$$
\mathcal{L}_{\text{KLU}}(f) = -\frac{1}{|\mathcal{X}|} \sum_i \sum_{j=1}^{C} \frac{1}{C} \log(C \cdot \hat{y}_{ij}) \tag{13}
$$

Its gradient:

$$
\frac{\partial \mathcal{L}_{\text{KLU}}}{\partial z_i} = \hat{y}_i - \frac{1}{C} \tag{14}
$$

The gradient $\to 0$ as $\hat{y}_i \to 1/C$ (uniform distribution — i.e., maximum confusion). Stable. Unlike ICE, KLU suppresses *all* tasks over the restricted-domain data, not just a specific labelled one.

**Denial of Service (DoS)** — for diffusion models:

$$
\mathcal{L}_{\text{DoS}}(f) = \frac{1}{|\mathcal{X}|} \sum_i \|f(x_{t_i}, t_i)\|^2 \tag{16}
$$

Its gradient: $\partial \mathcal{L}_{\text{DoS}} / \partial \hat{\epsilon}_i = 2\hat{\epsilon}_i$. As the model output $\hat{\epsilon}_i \to 0$ (denial of denoising service), gradient $\to 0$. Stable. MSE fails analogously to CE: its gradient $2(\hat{\epsilon}_i - \epsilon_i)$ grows with error.

### 3.6 Algorithm Summary

The full SOPHON loop (Algorithm 1):

```
initialise θ ← θ₀
for t in 1..Iter:
    # FTS phase (ℓ_FTS repetitions)
    for each fine-tuning env (φᵢ, Rᵢ, Tᵢ):
        run K inner-loop steps: f⁰_ϑ = fθ, f^k_ϑ from Eq.(3)
        compute Lᵢ,ₖ = Lα(f^k_ϑ | φᵢ, Tᵢ)
    L_FTS = Σᵢ Σₖ γᵢₖ · Lᵢ,ₖ
    θ ← Adam(θ, ∇θ L_FTS, α)      # Eq.(5)
    
    # NTR phase (ℓ_NTR repetitions)
    L_NTR = Lβ(fθ | O ~ D_S)
    θ ← Adam(θ, -∇θ L_NTR, β)     # Eq.(7)
```

Default hyperparameters: $\alpha = 3 \times 10^{-4}$, $\beta = 5 \times 10^{-4}$, $K = 50$, `Iter = 800`, $\ell_{\text{FTS}} = \ell_{\text{NTR}} = 1$.

---

## 4. Alignment with the Four Immunisation Properties

| Property | How SOPHON addresses it | Assessment |
|---|---|---|
| **Resistance** | Core objective: maximise adversary's loss after fine-tuning. Validated against 3 fine-tuning strategies, 5 optimisers, 5 learning rates, 5 batch sizes. | ✅ Strong, with partial gaps (see below) |
| **Stability** | NTR phase with loss $\mathcal{L}_{\text{NTR}}$ on $\mathcal{D}_S$. Empirically: original-domain accuracy preserved at 96.2% (CAFormer) vs 99.6% unprotected. | ✅ Well-demonstrated |
| **Generalisation** | Multi-strategy simulation ($N$ fine-tuning envs drawn from $\Phi$) and evaluation on multiple restricted domains (5 datasets). Generalisation across unseen optimisers confirmed. | ✅ Substantially demonstrated in-domain; see gap below |
| **Trainability** | Not tested. The paper does not evaluate whether SOPHON-protected models can still be fine-tuned on *benign* tasks. | ❌ Missing piece |

### The missing piece: trainability and cross-domain generalisation

SOPHON's defence is framed as making restricted-domain fine-tuning as hard as training from scratch. But it never asks: *does it also make benign fine-tuning harder?* If the protection degrades the loss landscape globally (not just in the restricted domain), then releasing the SOPHON model into the wild also damages its utility for legitimate fine-tuners. This is a critical open question for any deployment scenario.

Additionally, SOPHON is tested on cross-domain generalisation *within* the restricted-domain paradigm (e.g., model trained on ImageNette suppressed on CIFAR-10, CINIC, STL, MNIST, SVHN), but **cross-harm-domain generalisation** in the LLM sense — e.g., suppressing bioweapons knowledge while simultaneously generalising to CSAM suppression — is not evaluated.

There is also a partial resistance gap. With adaptive learning-rate optimisers (Adagrad, Adadelta, Adam), the protected model eventually begins to recover after 40–50 epochs (vs. 3–4 for training from scratch) — meaning SOPHON offers *weak* resistance rather than strong resistance against these optimisers. The paper's own extended experiments confirm this.

---

## 5. Mechanistic Commonalities with Other Approaches

SOPHON sits at the intersection of several mechanistic families. Below are its key structural parallels in the immunisation landscape.

### Bi-level meta-learning (inner/outer loop) — shared with MLAC/SDM, TAR

All three share the same skeleton:

$$
\min_\theta \; \mathcal{L}_{\text{outer}}\!\left(\text{InnerLoop}(\theta, \mathcal{D}_{\text{harm}}), \mathcal{D}_{\text{harm}}\right) + \lambda \cdot \mathcal{L}_{\text{retain}}(\theta, \mathcal{D}_S)
$$

The key differences are in $\mathcal{L}_{\text{outer}}$:
- **MLAC/SDM**: negated cross-entropy (gradient ascent). Instability problem acknowledged.
- **SOPHON**: ICE or KLU for classification, DoS for generation. Designed to be stable. But still equivalent to gradient ascent — the loss is inverted, not replaced.
- **TAR**: replaces the outer CE loss with an **entropy loss**, targeting flatness of the loss landscape rather than its height. TAR identifies that SDM-style CE ascent only lifts the adversary's loss at the first inner step, and the adversary recovers. SOPHON's ICE/KLU are a partial fix at the loss-function level, but TAR's entropy formulation is the cleaner resolution.

### First-order gradient approximation — shared with MLAC, TAR, Booster

All these methods approximate the second-order terms arising from differentiating through the inner loop. The second-order term $\nabla_\theta \nabla_\vartheta \mathcal{L}$ would require Hessian-vector products; first-order MAML (Reptile-style) replaces these with the gradient of the inner-loop final model treated as fixed. This makes the method tractable but less accurate.

### Multi-strategy simulation — extends MLAC

MLAC samples a single fine-tuning strategy per outer step. SOPHON explicitly samples $N$ strategies $\varphi_i \sim \Phi$ per step, making it closer in spirit to what TAR calls a "train-time adversary set" $\mathcal{A}_{\text{train}}$.

### Local-optimum entrapment — shared with NTL, Condition Number paper

The geometric intuition is identical across NTL, SOPHON, and the Condition Number paper: **push the model into a region of the restricted-domain loss landscape where the Hessian makes escape expensive**. NTL does this purely through representation divergence (no fine-tuning simulation). The Condition Number paper (Wangetal.) explicitly engineers the Hessian condition number $\kappa(H)$ of the restricted-domain loss to be large (ill-conditioned), slowing gradient convergence. SOPHON achieves the same qualitative goal implicitly, via the meta-learning loop, without explicitly targeting the Hessian.

---

## 6. Results and Significance

### Classification (CAFormer on ImageNette → CIFAR-10)

| Method | Orig. domain ACC | Restricted domain ACC after 20 epochs |
|---|---|---|
| Train from scratch | — | 62.7% |
| Fine-tune original model | 99.6% | 84.8% |
| Fine-tune NTL model | 90.2% | 84.4% |
| **Fine-tune SOPHON model** | **96.2%** | **15.2%** |

SOPHON-protected models end up near random-guess accuracy (10% for 10-class CIFAR-10), whereas NTL's protection collapses completely under fine-tuning. The intactness penalty is modest: 99.6% → 96.2%.

This result is replicated across 5 restricted domains, 5 model architectures (CAFormer, ResNet-50/34/18, VGG), 5 optimisers, 5 learning rates, and 5 batch sizes. The weakest result is VGG with Adadelta (56.4% after 40 epochs), compared to ~64% from scratch — still below the fine-tuned original (87.3%).

### Generation (Diffusion model on CIFAR-100 → CelebA face generation)

| Method | Restricted domain MSE after 20 epochs |
|---|---|
| Train from scratch | 0.479 |
| Fine-tune original model | 0.445 |
| **Fine-tune SOPHON model** | **0.705** |

The SOPHON-protected diffusion model cannot denoise faces even after 20 epochs of fine-tuning — well above the scratch baseline. Qualitatively, the generated images remain noise, while the original and scratch baselines converge to recognisable faces.

### Significance relative to the broader field

SOPHON's result represents a **qualitative shift** from NTL: the protection survives fine-tuning, which no prior work had demonstrated. In the context of the immunisation field:

- Compared to **MLAC/SDM** (which predates it and is classification-only on BERT-scale models), SOPHON is broader (classification + generation) and more carefully validated against diverse fine-tuning strategies.
- Compared to **TAR** (which postdates it), SOPHON is weaker against adaptive optimisers (Adam, Adadelta can recover given 40–50 epochs), whereas TAR achieves flat adversary loss even at 1,000 steps. TAR also addresses LLMs directly.
- SOPHON is thus best positioned as the **first complete proof of concept** for non-fine-tunable learning, and the direct precursor that motivated the TAR entropy-loss correction.

---

## 7. Open Challenges and Future Directions

### Calls from the authors (§7)

- **Extension to NLP and LLMs**: The paper acknowledges that classification and diffusion models are only two modalities. Extending SOPHON to autoregressive LLMs (the primary open-weight safety concern) is explicitly listed as future work.
- **More fine-tuning strategies**: LoRA and other PEFT methods are only briefly tested (preliminary experiments confirm SOPHON remains effective, but no rigorous evaluation is presented). With the explosion of LoRA-based harmful fine-tuning, this gap is critical.
- **Computational efficiency**: SOPHON requires three A100 (80GB) GPUs and 800 outer iterations for vision-scale models. Scaling to LLM-scale requires more efficient approximations of the inner loop — possibly through better Hessian approximations or surrogate models.

### From the state of the art

- **The entropy correction**: TAR's key insight — that CE-based outer-loop loss only lifts the adversary's loss early in training, and the model later recovers — is a direct limitation of SOPHON's ICE/KLU approach. The stable gradients of ICE/KLU prevent *divergence*, but they do not guarantee *flatness across all inner-loop steps*. A SOPHON variant using entropy in the outer loop (analogous to TAR) would be a natural and valuable improvement.
- **Trainability validation**: The immunisation definition paper (Rosati et al., EMNLP 2024) identifies trainability as a required condition. SOPHON never tests it. It is entirely possible that the hard-to-escape local optimum induced by SOPHON also penalises *benign* fine-tuning — which would make SOPHON-protected models commercially unreleasable.
- **Mechanistic interpretability of the protection**: What exactly does SOPHON do to the weight geometry? Is the restricted-domain loss landscape actually ill-conditioned (large $\kappa(H)$)? Is the protection localised to specific layers? Answering these questions would connect SOPHON to the Condition Number and LoX lines of work and could suggest more efficient immunisation strategies.
- **Resistance to RL-based attacks**: SOPHON is evaluated exclusively against SFT-based fine-tuning. Harmful RL fine-tuning (RLHF reversal, DPO reversal) has been shown to be more powerful than SFT — bypassing alignment with better Pareto efficiency. Whether SOPHON survives RL-based attacks is unknown.
- **Cross-domain generalisation in the LLM safety sense**: Suppressing a model on facial generation does not imply it will resist fine-tuning toward weaponisation. A theory of what kinds of restricted domains are structurally related — and whether SOPHON protections transfer across them — is absent from the literature.
