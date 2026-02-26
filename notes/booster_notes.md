# BOOSTER — Paper Notes
**Full title:** *BOOSTER: Tackling Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Perturbation*  
**Authors:** Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu — Georgia Institute of Technology  
**Venue:** ICLR 2025

---

## 1. Quick Summary — Mechanistic Point of View

Booster's core mechanistic claim is simple and surgically precise: **the gradient step taken over harmful data is the proximal cause of alignment collapse**, and it can be suppressed before the model is ever released.

The authors name this phenomenon *harmful perturbation* — a single stochastic gradient update in the direction of the harmful loss $h(\mathbf{w})$. They empirically show that:

- Fine-tuning on harmful data reduces $h(\mathbf{w})$ drastically and fast.
- Fine-tuning on benign data only marginally increases $h(\mathbf{w})$.
- Both training-set harmful loss reduction and test-set harmful loss reduction follow the same trend, meaning the effect generalises.

The defence operates entirely in the **alignment stage** (Stage ①), i.e., before the model is released. The insight is that if you can make the loss landscape around the aligned model *locally flat in the harmful direction* — so that a gradient step on harmful data produces very little loss reduction — then the attacker gains very little signal per step. Booster implements this by augmenting the standard alignment objective with a regulariser that **explicitly minimises the rate at which harmful loss decreases after a single simulated harmful gradient step**.

---

## 2. Positioning in the Timeline

### Ancestry and what Booster inherits

| Predecessor | What Booster takes from it |
|---|---|
| **Vaccine** (Huang et al., 2024 / NeurIPS 2024) | Same Georgia Tech group, same fine-tuning-as-a-service threat model, same alignment-stage philosophy; Booster is explicitly framed as a *successor* to Vaccine |
| **RepNoise** (Rosati et al., 2024) | Establishes the use of a *harmful dataset* (harmful prompt–harmful answer pairs) inside the alignment stage; Booster also requires this data and generalises the idea |
| **TAR** (Tamirisa et al., 2024) | Concurrent work using meta-learning to simulate adversarial perturbation; Booster uses a different insight (loss-reduction attenuation rather than entropy-based tamper resistance) |
| **MAML / Meta-learning** (Finn et al., 2017) | Booster explicitly borrows the one-step look-ahead trick: evaluating the objective *after* a gradient step, then backpropagating through it; the Hessian approximation that makes this tractable is also MAML's |

### What makes Booster unique in the field

While Vaccine perturbs **embeddings** to make the alignment distribution robust to downstream drift, and RepNoise projects harmful embeddings into **Gaussian noise**, Booster operates directly on the **weight-space loss landscape**. It is the first method to define a defence target as *attenuating the harmful loss reduction rate* — a second-order-style objective that flattens the harmful loss surface near the aligned weights. This is a mechanistically distinct angle from representation-space methods.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 Setup

Let:
- $\mathbf{w}$ — model weights.
- $f(\mathbf{w})$ — empirical cross-entropy loss over the **alignment dataset** $\mathcal{D}_\text{align}$ (harmful prompt → safe answer pairs).
- $h(\mathbf{w})$ — empirical cross-entropy loss over the **harmful dataset** $\mathcal{D}_\text{harm}$ (harmful prompt → harmful answer pairs).
- $\lambda > 0$ — regulariser intensity.
- $\alpha > 0$ — inner step size (simulated perturbation magnitude).

### 3.2 The Core Objective

$$
\arg\min_{\mathbf{w}} \; f(\mathbf{w}) + \lambda \underbrace{\left( h(\mathbf{w}) - h\!\left(\mathbf{w} - \alpha \frac{\nabla h(\mathbf{w})}{\|\nabla h(\mathbf{w})\|}\right) \right)}_{\text{harmful loss reduction after one normalised harmful step}}
\tag{1}
$$

**Dissecting the regulariser term by term:**

- $h(\mathbf{w})$: the harmful loss at the current weights — a baseline.
- $\mathbf{w} - \alpha \frac{\nabla h(\mathbf{w})}{\|\nabla h(\mathbf{w})\|}$: a *simulated one-step harmful perturbation*. The gradient is **normalised** (unit-norm) before scaling by $\alpha$, so the step size is always exactly $\alpha$ regardless of gradient magnitude. This is a design choice that stabilises training.
- $h\!\left(\mathbf{w} - \alpha \frac{\nabla h(\mathbf{w})}{\|\nabla h(\mathbf{w})\|}\right)$: the harmful loss *after* taking that simulated step.
- The difference $h(\mathbf{w}) - h(\ldots)$ is the **harmful loss reduction** — exactly what the attacker exploits per gradient step.
- Minimising this difference makes the aligned model sit at a point where taking one gradient step on harmful data produces the smallest possible loss reduction, i.e., the **harmful loss landscape is locally flat in the gradient direction**.

> **Geometric intuition:** Standard alignment pushes $\mathbf{w}$ to a low $f(\mathbf{w})$ basin. Booster additionally demands that this basin is a *saddle or plateau* of $h(\mathbf{w})$ in the direction of $\nabla h(\mathbf{w})$ — so harmful gradient descent has almost no traction.

### 3.3 Gradient Update — Full Chain Rule

Differentiating Eq. (1) with respect to $\mathbf{w}$ gives:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \left( \nabla f(\mathbf{w}_t) + \lambda \left( \nabla h(\mathbf{w}_t) - \nabla h\!\left(\mathbf{w}_t - \alpha \frac{\nabla h(\mathbf{w}_t)}{\|\nabla h(\mathbf{w}_t)\|}\right) \cdot \underbrace{\nabla\!\left(\mathbf{w}_t - \alpha \frac{\nabla h(\mathbf{w}_t)}{\|\nabla h(\mathbf{w}_t)\|}\right)}_{\text{contains the Hessian}} \right) \right)
\tag{2}
$$

The underbraced term is $I - \alpha \nabla^2_\mathbf{w} \left(\frac{\nabla h}{\|\nabla h\|}\right)$, which requires second-order information (a Hessian-vector product).

### 3.4 The First-Order Approximation (Practical Algorithm)

Following MAML's standard trick, the second-order term is approximated as the identity (i.e., the gradient of the inner step w.r.t. $\mathbf{w}$ is treated as a constant). This gives the tractable update:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \left( \nabla f(\mathbf{w}_t) + \lambda \left( \nabla h(\mathbf{w}_t) - \nabla h\!\left(\mathbf{w}_t - \alpha \frac{\nabla h(\mathbf{w}_t)}{\|\nabla h(\mathbf{w}_t)\|}\right) \right) \right)
\tag{3}
$$

**Reading Eq. (3) mechanistically:**
- $\nabla f(\mathbf{w}_t)$: gradient descent on alignment quality (standard SFT).
- $\nabla h(\mathbf{w}_t)$: this term **ascends** the harmful loss at the current point.
- $-\nabla h(\mathbf{w}_t - \alpha \hat{g})$: this term **descends** the harmful loss evaluated at the *perturbed weights*. It pulls the perturbed point toward higher harmful loss.
- Combined, $\nabla h(\mathbf{w}_t) - \nabla h(\mathbf{w}_t - \alpha\hat{g})$ is a *finite-difference approximation of the directional derivative of $\nabla h$*, i.e., it approximates the Hessian-vector product $H \cdot \hat{g}$ cheaply. It says: *"flatten the harmful loss curvature in the direction of the harmful gradient."*

### 3.5 Algorithm (Three Passes per Step)

```
Input:  λ (regulariser intensity), α (inner step size), η (learning rate), T (steps)
Output: Aligned model w̃

for t in 1..T:
  (xₜ, yₜ)   ← batch from D_align
  (x'ₜ, y'ₜ) ← batch from D_harm

  Pass 1: ∇f(wₜ)          on (xₜ, yₜ)
  Pass 2: ∇h(wₜ)          on (x'ₜ, y'ₜ)

  w̃ₜ = wₜ - α · ∇h(wₜ) / ‖∇h(wₜ)‖      # simulated perturbation

  Pass 3: ∇h(w̃ₜ)          on (x'ₜ, y'ₜ)  # harmful gradient at perturbed point

  g(wₜ) = ∇f(wₜ) + λ(∇h(wₜ) - ∇h(w̃ₜ))
  wₜ₊₁ = wₜ - η · g(wₜ)
```

Cost: **3× forward/backward passes** relative to plain SFT, and ~1.74× wall-clock time versus SFT, ~0.69× versus RepNoise.

### 3.6 Key Hyperparameters

| Param | Role | Failure mode if wrong |
|---|---|---|
| $\lambda$ | Regulariser weight; scales the defence signal relative to alignment loss | Too small → degrades to SFT; too large → alignment loss cannot compete, model refuses everything or collapses |
| $\alpha$ | Simulated step magnitude | Too small → regulariser vanishes (reduces to SFT); too large → simulated perturbation overshoots, no longer representative of actual attack steps |

---

## 4. Alignment with the Four Immunisation Properties

### ✅ Resistance — **Primary verified property**

Booster is designed and evaluated almost entirely for resistance. The regulariser directly targets the mechanism by which harmful fine-tuning succeeds: it minimises the per-step loss reduction an attacker can achieve. Empirical results show that harmful score barely rises over 2000 fine-tuning steps where SFT-aligned models collapse quickly. Resistance is demonstrated against the standard HFTA with a mixed harmful/benign fine-tuning set at $p = 0.1$ harmful ratio.

### ✅ Stability — **Verified but secondary**

Booster preserves fine-tuning accuracy (FA) on downstream benign tasks (SST2, AGNEWS, GSM8K) at levels comparable to plain SFT and superior to RepNoise. Benchmark accuracy (GSM8K on aligned model) drops only 0.4% versus SFT, compared to 4.3% for Vaccine and 1% for RepNoise.

### ⚠️ Generalisation — **Partially addressed, significant gap**

Within-domain generalisation to *unseen* harmful data within the same distribution is confirmed (harmful testing loss tracks training loss). However, **cross-domain generalisation** — defending against harmful fine-tuning using data from a completely different harm category than what Booster was trained with — is not systematically evaluated. The paper uses BeaverTails data for both alignment/Booster training and attack simulation, which is an in-distribution setting by design.

### ❌ Trainability — **Not directly addressed**

Booster does not evaluate whether the immunised model can be efficiently fine-tuned on a benign task from scratch (with no harmful data) at the same rate as an un-immunised model. The experiment shows FA is maintained *given* that fine-tuning proceeds on a contaminated dataset, but the formal trainability condition (comparable convergence speed on a clean $\mathcal{D}_\text{ok}$) is absent from the evaluation design.

---

## 5. Mechanistic Commonalities with Other Approaches

Booster belongs to a cluster of methods that share the same **first-order simulation** pattern:

| Method | Shared Mechanism | Difference |
|---|---|---|
| **TAR** (Tamirisa et al.) | Also simulates the model *after* a harmful gradient step; uses meta-learning framing | TAR uses **entropy loss** in the outer objective to ensure the model's harmful outputs become incoherent, not just resistant; Booster directly minimises loss reduction |
| **Vaccine** (Huang et al.) | Same lab; perturbs embeddings $\epsilon$ to maximise alignment loss, then minimises that worst-case loss (min-max) | Vaccine works in **representation space** (embeddings); Booster works in **weight space** (loss landscape) |
| **SAM / Sharpness-Aware Minimisation** | Also perturbs weights by a gradient step and evaluates the loss at the perturbed point | SAM minimises the *worst-case alignment loss* (sharpness); Booster minimises the *harmful loss reduction at the perturbed point* — different objective, same computational template |
| **MAML** (Finn et al.) | One-step look-ahead gradient, Hessian dropped for tractability | MAML optimises for fast adaptation to *new tasks*; Booster optimises for *resistance to harmful adaptation* — adversarial inversion of meta-learning |
| **RepNoise** (Rosati et al.) | Also uses harmful dataset inside alignment; introduces MMD regulariser to make harmful embeddings Gaussian | RepNoise acts on the representation geometry; Booster acts on the loss surface geometry |

**The broader pattern across the field:** Many immunisation methods — Vaccine, TAR, Booster, SAM-unlearning — all reduce to some form of:
$$
\min_\mathbf{w} \; \mathcal{L}_\text{safe}(\mathbf{w}) + \lambda \cdot R\!\left(\mathbf{w},\, \mathbf{w} + \delta(\mathbf{w})\right)
$$
where $\delta(\mathbf{w})$ is a simulated adversarial perturbation (either in weight space or embedding space), and $R$ measures some notion of vulnerability at or after that perturbation. The differences are in what $R$ measures and how $\delta$ is computed.

---

## 6. Results and Significance

### Headline numbers (Llama2-7B, default settings, 10% harmful ratio, SST2 fine-tuning)

| Method | Harmful Score (HS) ↓ | Fine-Tune Accuracy (FA) ↑ |
|---|---|---|
| SFT (no defence) | 33.70 | 93.12 |
| RepNoise | 32.10 | 93.00 |
| Vaccine | 28.30 | 93.69 |
| **Booster** | **8.30** | **93.23** |

Booster reduces harmful score by **17.26% vs Vaccine** and **20.08% vs RepNoise** (absolute), while matching fine-tuning accuracy, across four downstream tasks (SST2, AGNEWS, GSM8K, and a fourth). These are substantial improvements.

### Significance in the broader landscape

- Booster is the **best-performing pure alignment-stage defence** at its time of publication against standard HFTA.
- It demonstrates that directly targeting the *mechanism* of alignment collapse (harmful loss reduction rate) outperforms targeting its *symptom* (embedding drift) or its *representation* (MMD regularisation).
- However, subsequent work (notably the authors' own T-Vaccine and, more critically, adversarial evaluations by Qi et al. 2024 and Rosati et al. 2024a) shows that alignment-stage defences including Booster can be defeated by increasing attacker learning rate or dataset size — a corner-case that Booster's evaluation does not stress-test.
- Booster outperforms RepNoise while consuming **14.61 GB less GPU memory** in the alignment stage — a practical advantage for deployment pipelines.

### Limitations flagged in results

- The optimal $\lambda$ and $\alpha$ vary by downstream task, but a single pair must be chosen before knowing what fine-tuning task will be requested. Finding task-agnostic hyperparameters is non-trivial.
- Initial harmful training loss is *lower* for Booster than for SFT at fine-tuning step 0 — a paradoxical phenomenon the authors explain by showing that Booster's regulariser changes the generalisation relationship between alignment loss and harmful loss (the model does not push harmful loss into a high-loss region, it flattens it).

---

## 7. Future Work

### From the authors

- **Federated instruction fine-tuning**: Harmful fine-tuning in federated settings cannot be addressed by existing defences, and Booster's alignment-stage idea might extend there via sparse training or quantisation.
- **LLM agents**: Extending the threat model to LLM-agent pipelines where harmful fine-tuning could produce adversarially controlled agents.
- **Benign-data exploitation**: A speculative direction: optimise the aligned model so that fine-tuning on benign data *increases* the harmful loss, effectively counterbalancing harmful perturbations. This could allow the model to "self-repair" alignment during legitimate use.

### From the state of the art

- **Hyperparameter robustness**: The field needs alignment-stage defences that do not require task-specific tuning of $\lambda$ and $\alpha$. Adaptive schemes (e.g., using validation harmful score as a signal during alignment) are unexplored.
- **Stronger adversaries**: Standard evaluations use fixed learning rates and dataset sizes. Booster (and peers) need stress-testing against optimised attackers who tune their learning rate, batch size, and fine-tuning algorithm (LoRA, full fine-tuning, GRPO). Davies et al. 2025 show fundamental limitations of fine-tuning API defences under this stronger threat model.
- **Cross-domain resistance**: Booster's regulariser is trained on a specific harmful domain (BeaverTails). Cross-domain immunisation — where the defender does not know the harm domain the attacker will use — remains largely unsolved.
- **Composability**: Booster notes it can be combined with Vaccine or RepNoise. Systematic study of defence composition (and whether gains are additive or subadditive) is missing.
- **Theoretical guarantees**: Booster, like its peers, provides only empirical resistance. Bounding the number of harmful steps required to break the defence (in the spirit of Rosati et al.'s immunisation framework) is an open problem.
- **Beyond SFT attacks**: Booster is designed against SFT-based HFTA. RL-based harmful fine-tuning (e.g., GRPO with a harmful reward) may produce different gradient distributions for which Booster's normalised step simulation is not representative.
