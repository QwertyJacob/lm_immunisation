# Part 1.B.2 — Mechanistic Deep Dive: How "Make HFT Difficult" Actually Works

> **Session:** Morning · Block B · ~15 minutes  
> **Role of this segment:** Stop telling the story. Start dissecting the machines. Three mechanistic families, same immune goal — different levers pulled. Equations are explained before they are written, and again after.

---

## The Shared Engineering Tension

Every method in Block 1 has to simultaneously satisfy two competing objectives:

- **Resistance:** the model should be hard to push toward harmful behaviour via fine-tuning.
- **Trainability:** the model should remain easy to fine-tune for legitimate downstream tasks.

These are genuinely in tension. Any mechanism that raises the cost of harmful optimisation *in general* will also raise the cost of benign optimisation. The engineering challenge is to make that cost asymmetric — expensive for the attacker, cheap for the legitimate user.

We'll see three different levers that attempt to create this asymmetry.

---

## Lever 1 — Bi-Level Optimisation: Making the Weights a Bad Starting Point

### The Core Intuition

A fine-tuner is an optimiser. An optimiser is fast when it starts close to a solution. The idea here is to place the model's weights as far from any harmful solution as possible — to make the model a *terrible initialisation* for harmful tasks, while keeping it a good initialisation for everything else.

This is a meta-learning problem: learn *not to learn* harmful tasks.

### MLAC — The Original Template

The mechanism is a **bi-level optimisation**. An inner loop simulates the harmful fine-tuner; an outer loop uses the inner loop's outcome to push weights in the opposite direction.

Formally: let $\theta$ be the model parameters, $\mathcal{D}_h$ a harmful dataset, and $\mathcal{L}_h$ the harmful loss. The inner loop simulates $K$ steps of a harmful adversary:

$$\theta'(\theta) = \text{Adversary}_K(\theta; \mathcal{D}_h)$$

This gives us the hypothetical weights after $K$ steps of harmful fine-tuning. The outer loop then maximises the harmful loss at that point — i.e., makes things bad for the adversary *even after they've had $K$ steps*:

$$\max_{\theta} \; \mathcal{L}_h\big(\theta'(\theta);\, \mathcal{D}_h\big) \quad \text{subject to} \quad \mathcal{L}_d(\theta;\, \mathcal{D}_d) \leq \epsilon$$

where $\mathcal{L}_d$ is a desirable-task loss kept as a constraint so the model doesn't just destroy itself. The gradient of the outer objective with respect to $\theta$ passes *through* the inner loop — this is the second-order gradient computation that makes bi-level optimisation expensive.

> **What this does geometrically:** it pushes $\theta$ toward regions of parameter space where the harmful loss landscape is flat or has no useful gradient signal — the model becomes "stuck" for the adversary.

> **What this does in practice (MLAC, 2022):** tested on BERT-tiny, blocking demographic-information extraction, it works. Adversaries with up to 1000 fine-tuning steps cannot recover useful harmful performance. Inner loops as shallow as $K=16$ steps during training are sufficient.

---

### TAR — The Entropy Upgrade

MLAC's outer loop maximises cross-entropy loss $\mathcal{L}_h$. This sounds right, but it has a pathology: the model can achieve high cross-entropy by producing *confidently wrong* outputs — e.g., always predicting the same token with certainty. The adversary's loss explodes in the first few inner-loop steps, then recovers quickly as the adversary figures out the degenerate pattern. The model wins the battle for step 1 and loses the war for steps 2–64.

**TAR's fix:** replace the cross-entropy outer loss with an **entropy maximisation** objective. A model that produces maximally uncertain outputs ($H = \log|V|$, the entropy of a uniform distribution over vocabulary) is genuinely useless to the adversary at every step of fine-tuning — there is no exploitable gradient signal.

The full TAR objective is:

$$\min_{\theta} \; \lambda_{\mathrm{TR}} \cdot \mathbb{E}_{\text{attack} \sim \mathcal{A}_{\text{train}}} \Big[\mathcal{L}_{\mathrm{TR}}\big(\text{attack}(\theta);\, \mathcal{D}_{\mathrm{TR}}\big)\Big] \;+\; \lambda_{\text{retain}} \cdot \mathcal{L}_{\text{retain}}\big(\theta;\, \mathcal{D}_{\text{retain}}\big)$$

where $\mathcal{L}_{\mathrm{TR}}$ is the **negative entropy** of the model's output after an attack, and $\mathcal{D}_{\mathrm{retain}}$ is a normal capability dataset to prevent the model from collapsing on benign tasks.

The term $\text{attack}(\theta)$ is not differentiable through the fine-tuning process directly — TAR approximates this with first-order MAML-style gradients and accumulates tamper-resistance gradients separately to avoid holding $K$ computation graphs in memory simultaneously (a non-trivial systems engineering problem at LLM scale).

> **Result:** maximising entropy instead of cross-entropy means the adversary's loss stays high throughout their entire trajectory — not just at the first step. TAR trains on Llama-3-8B with multiple simulated adversaries at different learning rates and fine-tuning paradigms, achieving robust hazardous knowledge restriction on WMDP while preserving benign fine-tunability.

---

### Booster — Slowing Convergence Velocity

**Booster** takes a different angle on the same bi-level intuition. Instead of maximising the adversary's final loss, Booster minimises the *rate of reduction* of the harmful loss — i.e., how quickly fine-tuning converges. If the attacker converges slowly, they need more data and compute to break alignment. The objective:

$$\min_{w} \; f(w) \;+\; \lambda \left[ h(w) \;-\; h\!\left(w - \alpha \frac{\nabla h(w)}{\|\nabla h(w)\|}\right) \right]$$

where $f(w)$ is the alignment loss (keeping the model safe and useful), $h(w)$ is the harmful entropy loss, and the bracketed term is the difference in entropy *before and after one normalised harmful gradient step*. Minimising this term means that a single harmful gradient step produces as little entropy reduction as possible — the attacker barely moves.

Note the normalised step $\frac{\nabla h}{\|\nabla h\|}$: this is a unit step in the harmful gradient direction, decoupled from the learning rate. This gives Booster stability that TAR-style training can lack (TAR's cross-entropy variant collapses; Booster sidesteps this by never maximising cross-entropy directly).

---

## Lever 2 — Adversarial Perturbation on Representations: Anchoring the Residual Stream

### The Core Intuition

Harmful fine-tuning works by perturbing the model's weights $W$, which in turn perturbs the hidden embeddings $\mathbf{e}_l$ for each layer $l$. If a fine-tuner adds $\Delta W$ to a projection matrix, then the output at layer $l$ shifts by $\epsilon_{\mathrm{ft}} = \Delta W \cdot \mathbf{e}_{l-1}$ — a **Harmful Embedding Drift (HED)**. The downstream layers then operate on drifted representations, and safety-critical information is lost.

The insight: if we train the model to *maintain its safety behaviour even under embedding perturbations*, then fine-tuning-induced drift becomes less effective. We harden the representations, not just the weights.

### Vaccine — Perturbation-Aware Alignment

**Vaccine** formalises this as a min-max problem during alignment:

$$\min_{w} \; \max_{\|\epsilon\| \leq \rho} \; \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}\!\left(\left(\tilde{f}_{w_L, \epsilon_L} \circ \cdots \circ \tilde{f}_{w_1, \epsilon_1} \circ \mathcal{T}\right)(x_i),\; y_i\right)$$

where each $\tilde{f}_{w_l, \epsilon_l}(\mathbf{e}_{l-1}) = f_{w_l}(\mathbf{e}_{l-1}) + \epsilon_l$ injects a perturbation at layer $l$, and $\mathcal{T}$ is the tokeniser. The perturbation $\epsilon$ is constrained to an $\ell_2$-ball of radius $\rho$.

Solving the inner maximisation exactly is intractable, but a Taylor expansion to first order gives:

$$\mathcal{L}\!\left(\tilde{f}_{w_L} \circ \cdots\right) \approx \mathcal{L}\!\left(f_{w_L} \circ \cdots\right) + \sum_{l=1}^{L} \epsilon_l^\top \nabla_{\mathbf{e}_l} \mathcal{L}_{w}(\mathbf{e}_l)$$

Maximising this linear form over $\|\epsilon\| \leq \rho$ via Hölder's inequality immediately gives the **closed-form optimal perturbation**:

$$\epsilon^*_{l} = \rho \; \frac{\nabla_{\mathbf{e}_l} \mathcal{L}_{w}(\mathbf{e}_l)}{\left\|\nabla \mathcal{L}_{w}(\mathbf{e}_1, \ldots, \mathbf{e}_L)\right\|_2}$$

This is the gradient of the loss at layer $l$, projected onto the $\ell_2$-ball — exactly the FGSM perturbation, but applied to hidden embeddings rather than inputs. The outer minimisation then trains the model to be robust to this worst-case embedding drift.

> **Algorithm:** two forward-backward passes per step. First pass: compute $\nabla_{\mathbf{e}_l}\mathcal{L}$ and compute $\epsilon^*$. Register $\epsilon^*$ as a forward hook. Second pass: compute gradients of the perturbed loss and update $w$.

> **What this achieves:** the model learns to maintain alignment-relevant information even when each layer's output is adversarially perturbed. When a future fine-tuner causes real HED, the model is already hardened.

---

### T-Vaccine — Surgical Targeting

Vaccine applies perturbations uniformly to all $L$ layers. This is wasteful: the last few layers often have small gradient norms on harmful data and contribute little to safety. Perturbing them degrades utility without improving resistance.

**T-Vaccine** uses the harmful gradient norm as a proxy for safety-criticality:

$$s_{l,t} = \left\|\nabla_{\mathbf{e}_{l,t}} \mathcal{L}_{w_t}(\mathbf{e}_{l,t};\; x_h, y_h)\right\|_2$$

$$p_{l,t} = \frac{s_{l,t}}{\sum_{l'=1}^{L} s_{l',t}}$$

These probabilities define a distribution over layers. At each training step, T-Vaccine samples $\gamma$ layers (the "safety-critical" ones) according to $P_t$ and applies Vaccine-style perturbation only to those. All other layers are frozen for that step.

> **Effect:** comparable or better resistance than Vaccine, with dramatically reduced memory overhead — to the point of being trainable on a consumer RTX 4090 for 7B models. The key insight is that the harmful gradient norm *reveals* which layers matter for safety; the model itself tells you where to look.

---

### Circuit Breakers — Rerouting Instead of Resisting

**Circuit Breakers** take a fundamentally different approach to representation hardening. Instead of making representations *robust to drift*, they make harmful representations lead to *incoherence*.

Formally, let $\text{rep}_\text{orig}(x)$ be the internal representation of harmful input $x$ in the original model, and $\text{rep}_{c/b}(x)$ be the same representation after circuit-breaking. The **Representation Rerouting (RR) loss** is:

$$\mathcal{L}_{\mathrm{RR}} = \mathrm{ReLU}\!\left(\frac{\text{rep}_{c/b}(x_s) \cdot \text{rep}_\text{orig}(x_s)}{\|\text{rep}_{c/b}(x_s)\|_2 \;\|\text{rep}_\text{orig}(x_s)\|_2}\right)$$

This is the cosine similarity of the original and circuit-broken representations, clipped at zero: it penalises any positive alignment between what the original model "thinks" on harmful input and what the rerouted model "thinks". The goal is to drive these representations toward orthogonality — the circuit-broken model is *not thinking the same thoughts* during harmful generation, so harmful generation cannot proceed coherently.

The retain loss keeps benign representations unchanged:

$$\mathcal{L}_{\mathrm{retain}} = \|\text{rep}_\text{orig}(x_r) - \text{rep}_{c/b}(x_r)\|_2^2$$

The full loss is a coefficient-scheduled combination of both: early in training $\mathcal{L}_{\mathrm{RR}}$ dominates (plant the circuit breakers); late in training $\mathcal{L}_{\mathrm{retain}}$ dominates (lock in utility). The circuit-breaker model is implemented via LoRA adapters on top of the frozen base model, avoiding catastrophic forgetting.

> **Key geometric observation:** cosine similarity between original and rerouted representations drops dramatically starting around layer 10, *even during the pre-filling phase* — before generation starts. The circuit-breaking happens early in the forward pass, meaning it's robust to adversarial prompts that attempt to bypass refusal at the output level.

---

## Lever 3 — Loss Landscape Geometry: Making the Terrain Itself Hostile

### The Core Intuition

Whether or not the representation is robust, fine-tuning is still a gradient descent problem. Gradient descent converges fast when the loss landscape is **well-conditioned** — when it looks like a round bowl. It converges slow (or not at all, practically speaking) when the landscape is **ill-conditioned** — when it looks like a narrow valley that gradient descent bounces around in endlessly.

The condition number $\kappa = \sigma_{\max} / \sigma_{\min}$ of the Hessian (or its Fisher approximation) exactly characterises this. High $\kappa$ on the harmful task = slow, erratic convergence for the attacker. The immunisation goal becomes: **maximise $\kappa$ on harmful tasks, minimise $\kappa$ on benign tasks.**

### Condition Number Regularisation

The **Condition Number paper** turns this insight into two differentiable regularisers, operating on the singular value matrix $\mathbf{S}$ of a proxy for the Hessian.

To **ill-condition** the harmful landscape (raise $\kappa$, block attacker):

$$\mathcal{R}_{\mathrm{ill}}(\mathbf{S}) = \frac{1}{2p}\|\mathbf{S}\|_F^2 - \frac{1}{2}\|\mathbf{S}\|_2^2$$

This is the average singular value squared *minus* the maximum singular value squared — minimising it pushes the largest singular value as far above the average as possible, creating a highly anisotropic landscape.

To **well-condition** the benign landscape (keep $\kappa \approx 1$, help legitimate users):

$$\mathcal{R}_{\mathrm{well}}(\mathbf{S}) = \frac{1}{2}\|\mathbf{S}\|_2^2 - \frac{1}{2p}\|\mathbf{S}\|_F^2$$

This is exactly the opposite: minimising it drives the largest singular value toward the average — making all curvature directions equal, i.e., a round bowl, i.e., fast convergence.

The full immunisation loss adds these regularisers to a standard alignment loss, with separate hyperparameters $\lambda_{\text{ill}}$ and $\lambda_{\text{well}}$ for the two terms.

> **Caveat:** the theory is proven under linearity assumptions ($\ell_2$ loss, linear classifier). In practice it transfers to nonlinear LLMs, but the connection is empirical rather than guaranteed.

---

### LoX — No Training Needed

**LoX (Low-Rank Extrapolation)** asks: what if we could engineer the loss landscape without any additional training at all?

The observation: the difference between the aligned model $\theta_{\text{align}}$ and the unaligned base $\theta_{\text{base}}$ — the "alignment delta" $\Delta W = \theta_{\text{align}} - \theta_{\text{base}}$ — encodes the safety direction in weight space. If we decompose $\Delta W$ via SVD:

$$\Delta W = U \Sigma V^\top$$

the top-$k$ singular vectors $U_k, V_k$ span the **safety-critical low-rank subspace** — the principal directions of the alignment update. LoX simply **extrapolates further along this subspace** by a factor $\alpha > 1$:

$$\theta_{\mathrm{LoX}} = \theta_{\text{base}} + \alpha \cdot U_k \Sigma_k V_k^\top$$

This pushes the model from the narrow valley of standard alignment into a flatter, more robust region — visualised as moving the model away from the steep walls of the safety landscape toward an open plateau where fine-tuning in any direction has a long way to go before crossing into unsafe territory.

> **Efficiency:** this is a single SVD decomposition + matrix multiplication after alignment. No additional training, no inner-loop adversary, no extra data. Adds almost zero overhead. Scales to any LLM architecture that uses linear weight matrices (i.e., all of them).

> **Limitation:** the extrapolation factor $\alpha$ requires tuning. Too large: model breaks (low-rank components carry enough signal to destabilise generation). Full-rank extrapolation is particularly brittle. Low-rank ($k$ = effective rank of $\Delta W$, typically $k=6$ for 7B models) extrapolation is much more stable.

---

## Overhead and Practical Concerns

| Method | Training overhead | Memory overhead | Post-deployment cost |
|---|---|---|---|
| MLAC / TAR | ~3–5× slower (inner loop simulation) | ~2–3× more GPU RAM | None |
| Booster | ~2× slower (two-pass gradient) | ~1.5× | None |
| Vaccine | ~2× slower (two-pass) | **High** — all-layer activations stored | None |
| T-Vaccine | ~1.5× slower | **Low** — only $\gamma$ layers stored | None |
| Circuit Breakers | ~1.5× slower (LoRA on frozen model) | Moderate (LoRA params) | None |
| Condition Number | ~1.2× slower (regulariser evaluation) | Low | None |
| **LoX** | **Zero** (post-alignment, training-free) | Zero | None |

The key pattern: overhead is paid **once at alignment time**. Once the model is immunised and released as open-weight, every downstream user benefits from the protection without paying any additional cost. This is what makes these methods candidates for immunisation rather than just "defence at inference time."

---

## Limitations and Calls for Future Work

**The distribution shift problem.** Bi-level methods simulate inner-loop adversaries during training, but real attackers may use learning rates, batch sizes, or fine-tuning paradigms outside the training distribution. TAR partially addresses this by training against a diverse set of adversaries $\mathcal{A}_{\text{train}}$, but OOD generalisation remains an open question.

**The compute budget asymmetry.** If the attacker's budget exceeds the simulated $K$ inner-loop steps, bi-level methods provide weaker guarantees. Defence depth is bounded by training compute.

**Representation methods vs. ITI attacks.** Vaccine, T-Vaccine, and RepNoise are designed against fine-tuning attacks. Inference-time interventions (ITI) that steer the residual stream directly at inference can bypass them — this is precisely the gap that motivated Circuit Breakers and E.T. (this afternoon). The representation-space methods remain largely orthogonal in their assumptions.

**Contextual misdirection.** Representation-rerouting methods can be partially fooled by prompts that dilute the harmful signal in the internal state — e.g., embedding the harmful request inside a hypothetical framing. The circuit breaker "fires" based on representations; if the representation is sufficiently ambiguous, the breaker may not activate.

**The trainability-resistance Pareto frontier.** No method in Block 1 has demonstrated that the Pareto frontier between full resistance and full trainability can be pushed simultaneously to its limits. Current methods achieve partial resistance at some trainability cost. The frontier's shape, and whether it is convex, is not yet characterised.

---

*These limitations are the open map. We'll return to them in Part 2.B.*
