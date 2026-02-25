# A Tutorial on Language Model Immunisation

*Jesús F. Cevallos-Moreno, University of Insubria*

### Introduction: Motivation and Problem Statement *(30 minutes)*

- The *Dual-Use Dilemma*, the *Vulnerability Argument*, and the [Vulnerability Universality](https://arxiv.org/abs/2506.03850).
  - The evolution from access restrictions to structural resistance.
  - **The Vulnerability Argument:** If safety guards can be easily removed, the model is fundamentally unsafe.
  - Historical context: From meta-learned task blocking in BERT-style models (MLAC) to open-weight LLMs.

- Mechanistic effects: [*Shallow* Safety Alignment](https://arxiv.org/abs/2406.05946) and the concept of [*Harmful Embedding Drift*](https://arxiv.org/abs/2406.04313) (HED).
  - **Concept:** Alignment typically adapts the generative distribution primarily over the first few tokens (the "safety shortcut").
  - **Interpretability insight:** Per-token KL divergence analysis showing most of the "KL budget" is spent on initial refusal prefixes (e.g., "I cannot fulfill…").
  - Vulnerability to **harmful embedding drift**: fine-tuning on user data causes hidden embeddings to drift from safe states.

- From Harmful Supervised Fine-Tuning (SFT) to Harmful Reinforcement Learning (RL).
  - Bypassing alignment with as few as 10–100 samples.
  - Mechanistic view: fine-tuning perturbs safety-critical low-rank subspaces.
  - **Superiority of RL:** RL surpasses SFT's Pareto frontier, breaking alignment more effectively while preserving reasoning capabilities for complex harmful tasks.
  - **Interpretability insight:** RL improves metrics by reducing response entropy.
  - **Gradient Bound Theorem:**

$$\|\nabla_{\theta}J(\theta)\| \le C\sqrt{\overline{H(\pi_{\theta})}}.$$

- The [four pillars of immunisation](https://arxiv.org/abs/2402.16382): Resistance, Stability, Generalisation, and Trainability.
  - **Resistance:** weak vs. strong resistance based on the attacker's compute budget.
  - **Stability:** retaining general language modeling capability.
  - **Generalization:** defense against out-of-distribution (OOD) attack datasets.
  - **Trainability:** maintaining utility for benign downstream fine-tuning.

---

### Part 1: Weight-Space Resilience and Adversarial Meta-Learning *(30 minutes)*

- **Mechanistic goal:** Producing model weights that inhabit the [*safety basin*](https://arxiv.org/abs/2506.03850) in the loss landscape.

- **Techniques:** from [bi-level optimisation](https://arxiv.org/abs/2211.14946), [Low-Rank Extrapolation](https://arxiv.org/abs/2506.15606), [Hessian Robustness](https://openreview.net/forum?id=uitj69FqD5).
  - **MLAC (Meta-Learned Adversarial Censoring):** the progenitor technique; uses bi-level optimisation where the outer loop maximises the loss of an inner-loop adversary.
  - **TAR (Tampering Attack Resistance):** modernises MLAC for LLMs using **entropy loss** instead of cross-entropy in the outer loop to prevent the adversary from "quickly recovering" at later steps.
  - **Equation:**

$$\min_{\theta}\; \lambda_{\mathrm{TR}}\, \mathbb{E}_{\text{attack}\sim\mathcal{A}_{\text{train}}} \Big[\mathcal{L}_{\mathrm{TR}}(\text{attack}(\theta);\mathcal{D}_{\mathrm{TR}})\Big] + \lambda_{\text{retain}}\, \mathcal{L}_{\text{retain}}(\theta;\mathcal{D}_{\text{retain}}).$$

  - **LoX (Low-Rank Extrapolation):** a training-free evolution that identifies safety-critical low-rank subspaces through SVD and extrapolates them to move parameters into a flatter, less perturbation-sensitive zone.

- **Visualising unlearning resilience and the safety loss landscape.**
  - **Task vector analysis:** visualizing unlearning resilience as preserving the "unlearning direction" (τᵤ) against the "fine-tuning direction" (τ_ft), keeping them near-orthogonal rather than opposite.
  - **Safety landscape visualisation:** showing how immunisation moves the model out of a "narrow valley" of safety into a robust "flat zone".

- **Limitations and sensitivity to out-of-distribution attacks.** Pure weight-space methods can be sensitive to out-of-distribution learning rates; if the attacker's budget exceeds the simulated inner-loop depth (e.g., K=64), the defence may plateau.

---

### Part 2: Representation Engineering and Residual Stream Intervention *(30 minutes)*

- **Mechanistic goal:** Crafting an [invariant residual stream](https://openreview.net/forum?id=sfz57tKe5E). Ensuring the residual stream (hᵢ) remains invariant to adversarial perturbations, neutralising "embedding drift" before it propagates across layers.

- **Techniques:** [gradient attenuation](https://openreview.net/forum?id=tTPHgb0EtV), representation [noising](https://openreview.net/forum?id=eP9auEJqFg&referrer=%5Bthe%20profile%20of%20Domenic%20Rosati%5D(%2Fprofile%3Fid%3D~Domenic_Rosati2)), [rerouting](https://arxiv.org/html/2406.04313v2), and [vaccination](https://arxiv.org/abs/2410.09760).
  - **Vaccine:** introduces perturbation-aware alignment, adding artificial noise to embeddings during alignment to create resistance to HED (harmful embedding drift).
  - **Optimal perturbation equation:**

$$\epsilon^{*}_{\ell,t} = \rho\, \frac{\nabla_{e_{\ell,t}}\mathcal{L}_{w_t}(e_{\ell,t})}{\left\lVert\nabla\mathcal{L}_{w_t}(e_{1,t},\dots,e_{L,t})\right\rVert}.$$

  - **Targeted Vaccine (T-Vaccine):** improves memory efficiency by using **harmful gradient norms** to identify and selectively perturb only "safety-critical" layers.
  - **Circuit breakers:** uses **representation rerouting (RR)** to map internal representations of harmful processes to an orthogonal space, effectively short-circuiting generation.

- **HED Analysis,** [**Negative Log-Likelihood**](https://openreview.net/forum?id=sfz57tKe5E)**, and cosine similarity probes.**
  - **HED analysis:** using t-SNE to visualise how harmful fine-tuning drags clean embeddings into "malicious clusters" and how immunisation keeps them stationary.
  - **Cosine similarity probes:** monitoring cosine similarity between representations with/without circuit breakers to detect where a harmful generation begins to "break" (often starting dramatically around layer 10).

- **Stress-testing through contextual misdirection.** These methods can be fooled by contextual misdirection (e.g., hypothetical framing) that dilutes the harmful signal in the internal state.

---

### Part 3: Deterministic Constraints and Conditional Model Collapse *(30 minutes)*

- **Mechanistic goal:** Binding fundamental utility to the [safety state](https://arxiv.org/pdf/2505.12186), such that removing safety guardrails triggers the self-destruction of general language modelling capabilities.

- **Techniques:** [Self-Degradation Defence](https://arxiv.org/abs/2507.21182), [Collapse Trap](https://arxiv.org/abs/2505.16559), [Entropy Minimisation](https://arxiv.org/abs/2508.20697), [Perplexity Curation](https://arxiv.org/abs/2405.19358).
  - **SDD (Self-Degraded Defense) / CTRAP:** embeds a "collapse trap" that forces functional inertness if pushed in a harmful direction.
  - **Collapse loss:**

$$l_{\text{Collapse}}(\theta;\mathcal{D}) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[-\frac{1}{|y|}\sum_{t}\log p\big(e\,\big|\,x\circ y_{<t};\theta\big)\right],$$

where *e* is a fixed "error" token.

  - **TOKENBUNCHER:** targets **harmful-RL** by minimising response entropy on harmful queries, removing the exploration that RL relies on.

- **Using entropy-as-reward in RL, injecting stochasticity into low-logit tokens.**
  - **Entropy-as-reward:** using RL against RL by treating high entropy as a penalty, forcing safe deterministic trajectories.
  - **Token noiser:** injecting stochastic mass into low-logit tokens; while invisible in normal tasks, harmful-RL amplifies this noise and induces "gibberish" collapse.

- **Avoiding catastrophic forgetting during trap implantation.** Strong fail-safe defences require careful balancing to avoid catastrophic forgetting of benign tasks during trap implantation.

---

### Part 4: Mechanistic Frontiers and the Horizon of Robustness *(30 minutes)*

- **Fundamental dilemma:** The rigidity of strong resistance, normativity and time, and the "empty shell".
  - **The "empty shell" problem:** If the underlying safety alignment is merely an obfuscation of harmful knowledge rather than its erasure, immunisation techniques essentially lock an empty shell. Attacks can bypass these durable guards by exploiting residual general adaptability—the model's inherent ability to repurpose its "intelligence" to learn new harmful patterns even if specific pathways are blocked.
  - **The rigidity vs. trainability dilemma:** "Strong resistance" (permanent immunity to harmful tuning) risks creating a functionally inert model. Defenders face a trade-off: a model so resilient it cannot be misaligned often suffers from mode collapse, losing trainability for legitimate benign tasks.
  - **The normativity constraint:** Immunisation requires a normative definition of harm, which is context-dependent and contentious. Developers face an "empty signifier" problem: what should be unlearned or immunized against may privilege certain societal values over others.

- **Specific solvable challenges:** Locally neutralising the [butterfly-effect](https://aclanthology.org/2024.findings-acl.322/) and the [challenge of evaluation](https://openreview.net/forum?id=fXJCqdUSVG).
  - **Neutralising the "butterfly effect" in residual streams:** Small adversarial interventions at early layers can amplify through subsequent decoder blocks. A realistic challenge is developing layer-wise defensive modules that locally neutralise these interventions before they propagate to the output.
  - **Addressing RL-based subversion:** Most current immunisations target supervised fine-tuning (SFT), but reinforcement learning (RL) poses a greater systemic risk. RL can bypass alignment-stage defences more effectively than SFT while preserving reasoning capabilities for complex, harmful tasks.
  - **Mitigating "gradient starvation":** During alignment, gradients from large data groups dominate small ones, leading to uneven forgetting where vulnerable subsets of safety data are insufficiently learned and easily overridden.

- **Promising research avenues:** [adversarial hypernetworks](https://arxiv.org/abs/2509.08000), [invariant unlearning](https://openreview.net/forum?id=x2lm33kdrZ), and the case of [ArchLock](https://openreview.net/forum?id=e2YOVTenU9).

  The field is shifting from behaviour-centric supervision to representational-level structural resilience.

  - **Invariant LLM Unlearning (ILU):** Inspired by Invariant Risk Minimisation (IRM), ILU ensures the unlearned state remains stationary under fine-tuning perturbations:

$$\min_{\theta} l_{u}(\theta) + \lambda \sum_{i=1}^{N} \left\lVert\nabla_{w\,|\,w=1} l_{i}(w \circ \theta; \mathcal{D}_{i})\right\rVert_{2}^{2}.$$

  - **Adversarial Hypernetworks (AntiDote):** A differentiable neural network H_φ generates adversarial LoRA patches (Uₗ, Vₗ) conditioned on internal activations:

$$(U_{\ell},V_{\ell}) = H_{\phi}\big(X_{\ell}(x;\theta)\big).$$

  - **Self-Destructive Modeling (SEAM):** Couples benign and harmful gradients so they adopt opposing directions:

$$\mathcal{L}_{\mathrm{sd}}(\theta) = \operatorname{sim}\big(g_{a}(\theta), g_{b}(\theta)\big).$$

  - **Entropy-as-reward defense:** Minimizing response uncertainty (entropy) on harmful queries makes the policy more deterministic; as entropy shrinks, the policy gradient is bounded and can vanish in practice:

$$\|\nabla_{\theta}J(\theta)\| \le C\sqrt{\overline{H}(\pi_{\theta})}.$$

  - **Hessian-based immunisation:** Using the condition number of the Hessian to control convergence speed. By maximising the condition number for harmful tasks, malicious tuning can be made exponentially slower while benign tasks remain well-conditioned.