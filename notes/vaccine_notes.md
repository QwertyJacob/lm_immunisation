# Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack

**Huang, Hu & Liu — Georgia Tech — NeurIPS 2024**
[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/873c86d9a979ab80d8e2919510d4446b-Abstract-Conference.html) | [Code](https://github.com/git-disl/Vaccine)

---

## 1. Quick Mechanistic Summary

Vaccine's starting point is an empirical observation: when a user fine-tunes an already-aligned LLM on data that contains even a small fraction of harmful examples (~10%), the **hidden embeddings of the original alignment data shift**. The authors call this the **Harmful Embedding Drift (HED)** phenomenon. They show that this drift (measured as the L2 norm of the difference between pre- and post-fine-tuning embeddings over alignment inputs) correlates tightly with the model's harmful score — the more drift, the more broken the alignment.

The mechanistic explanation is simple: the user fine-tuning perturbs the attention weight matrices $W_l$ by some $\tilde{W}_l$. For an alignment input $x$, the output of that layer changes from $f(x) = W_l x$ to $\tilde{f}(x) = W_l x + \tilde{W}_l x = f(x) + \epsilon_{ft}$, where $\epsilon_{ft} \triangleq \tilde{W}_l x$ is the harmful embedding drift. The alignment knowledge was encoded in a region of embedding space; the drift moves the model out of that region.

Vaccine's defence is an **alignment-stage-only intervention**: rather than protecting the model at fine-tuning time (which requires access to user data), it **trains the model to be insensitive to bounded embedding perturbations** during alignment. If the model learns to maintain a low alignment loss even when each layer's hidden embedding is pushed adversarially within a ball of radius $\rho$, then the actual (unseen, unconstrained) HED introduced later by user fine-tuning will be less likely to break the alignment. The alignment is, in the medical sense of the word, vaccinated.

---

## 2. Timeline Positioning

### Where Vaccine sits in the 20-paper landscape

Vaccine (arXiv February 2024, NeurIPS 2024) is **the founding paper of the perturbation-aware alignment-stage defence family**. It is the first paper to

1. formally diagnose HED as the mechanistic cause of alignment breaking, and
2. propose a purely alignment-stage solution that requires no knowledge of future user data.

### What it inherits

| Ancestor | What Vaccine borrows |
|---|---|
| **FGSM** (Goodfellow et al. 2014) | The idea of a single-step gradient-sign perturbation as a worst-case surrogate |
| **SAM** (Foret et al. 2020) | The FGSM-style dual forward/backward pass to find the sharpness-maximising perturbation and then minimise under it; the ρ-ball constraint |
| **Meta-learning** (MAML, Finn et al. 2017; Ripple, Kurita 2020) | The high-level structure of making alignment robust to subsequent gradient steps; the alignment-only stage philosophy |
| **Continual learning / catastrophic-forgetting literature** (EWC, Kirkpatrick 2017) | The motivation: prevent the aligned model from forgetting alignment knowledge under later fine-tuning |
| **Harmful fine-tuning attack papers** (Qi et al. 2023; Yang et al. 2023; Zhan et al. 2023) | The threat model: SFT on user data containing a small harmful fraction |

### Its unique contribution

- **Proposes HED** as a *measurable, causal* diagnostic, not just an empirical observation.
- **Operates with zero knowledge of the downstream user data** — unlike meta-learning or Ripple, which need simulated attack data in the alignment phase.
- **Perturbation lives in embedding space**, not weight space — this distinguishes it from weight-space methods (TAR, LoX) and from RepNoise (which targets the distribution of harmful embeddings, not their drift).
- Direct descendants: **T-Vaccine** (layer-wise, memory-efficient), **Booster** (extends to harmful data via a harmful-perturbation regulariser), **VAA** (adds data-aware Group DRO on top of the SAM-style inner maximisation).

---

## 3. The Math — Step by Step

### 3.1 Setup

Denote:
- $w = (w_1, \ldots, w_L)$: model weights across $L$ transformer layers.
- $e_l$: hidden embedding at layer $l$, so $e_l = f_{w_l}(e_{l-1})$.
- $\mathcal{T}(x_i) = e_{i,0}$: tokeniser output (initial embedding).
- $\{x_i, y_i\}_{i=1}^N$: alignment dataset (harmful prompt → safe answer pairs).
- $\mathcal{L}$: cross-entropy alignment loss.

### 3.2 The Mini-Max Problem

Vaccine formalises the alignment objective as:

$$\min_{w} \max_{\|\epsilon\| \le \rho} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}\!\left((\tilde{f}_{w_L, \epsilon_L} \circ \cdots \circ \tilde{f}_{w_1, \epsilon_1} \circ \mathcal{T})(x_i),\; y_i\right) \tag{1}$$

where the perturbed forward pass at each layer is defined as:

$$\tilde{f}_{w_l, \epsilon_l}(e_{l-1}) = f_{w_l}(e_{l-1}) + \epsilon_l \quad \forall l \in [L]$$

and $\epsilon = (\epsilon_1, \ldots, \epsilon_L)$ is the joint perturbation across all layers, constrained to the global L2 ball $\|\epsilon\| \le \rho$.

**Inner problem** (maximiser): find the worst-case embedding perturbation that maximises alignment loss — simulating what harmful fine-tuning will do to the embeddings.

**Outer problem** (minimiser): find model weights that keep the alignment loss low *even under* that worst-case perturbation.

### 3.3 Solving the Inner Maximisation: the Optimal Perturbation

The inner problem is intractable exactly because the perturbation propagates non-linearly through all layers. Vaccine approximates the perturbed loss using a **first-order Taylor expansion** applied sequentially at each layer:

$$\mathcal{L}(\tilde{f}_{w_L,\epsilon_L} \circ \cdots \circ \tilde{f}_{w_1,\epsilon_1} \circ \mathcal{T}(x_i), y_i) \approx \mathcal{L}(f_{w_L} \circ \cdots \circ f_{w_1} \circ \mathcal{T}(x_i), y_i) + \sum_{l=1}^{L} \epsilon_l^T \frac{d\mathcal{L}}{de_l} \tag{2}$$

The first term is constant w.r.t. $\epsilon$. The inner maximisation then reduces to:

$$\arg\max_{\|\epsilon\| \le \rho} \sum_{l=1}^{L} \epsilon_l^T \nabla_{e_l} \mathcal{L}_w(e_l) = \arg\max_{\|\epsilon\| \le \rho}\; \epsilon^T \nabla \mathcal{L}_w(e_1, \ldots, e_L)$$

where the concatenated gradient is $\nabla \mathcal{L}_w(e_1, \ldots, e_L) = (\nabla_{e_1}\mathcal{L}_w(e_1), \ldots, \nabla_{e_L}\mathcal{L}_w(e_L))$.

By **Hölder's inequality**, $\epsilon^T g \le \|\epsilon\| \cdot \|g\|$, with equality when $\epsilon$ is aligned with $g$. Under the constraint $\|\epsilon\| \le \rho$, the optimum is achieved by:

$$\boxed{\epsilon_l^* = \rho \;\frac{\nabla_{e_l} \mathcal{L}_w(e_l)}{\|\nabla \mathcal{L}_w(e_1, \ldots, e_L)\|}} \tag{3}$$

**Key insight about the denominator**: the norm in the denominator is the norm of the *global* (all-layer) concatenated gradient, not the per-layer gradient norm. This means each layer's perturbation magnitude is proportional to its local gradient but is normalised globally. Layers with a larger gradient contribution get a larger perturbation, but the total perturbation stays within the ρ-ball. This is the normalised gradient ascent direction across the residual stream as a whole.

### 3.4 Solving the Outer Minimisation: the Two-Pass Algorithm

With $\epsilon^*$ in hand (from the first forward/backward pass), Vaccine injects the perturbation into each layer via a **forward hook**:

$$\tilde{f}_{w_l, \epsilon_{l,t}}(e_{l,t}) = f_{w_l}(e_{l,t}) + \epsilon_{l,t}$$

and performs a **second forward/backward pass** through the fully-perturbed network to obtain the gradient $\tilde{g}_t$ that will actually update the weights:

$$w_{t+1} = \text{Optimizer\_Step}(w_t,\; \tilde{g}_t)$$

**Algorithm 1 (Vaccine)**:

```
Input:  ρ (perturbation intensity), T (training steps), L (layer count)
Output: aligned model w_{T} ready for fine-tuning

for step t in T:
    Sample batch (x_t, y_t)
    
    # Pass 1: compute optimal perturbation
    backward ∇L_{w_t}(e_{1,t}, ..., e_{L,t})
    for each layer l:
        ε_{l,t} = ρ · ∇_{e_{l,t}} L_{w_t}(e_{l,t}) / ‖∇L_{w_t}(e_{1,t},...,e_{L,t})‖
        register forward hook: f̃_{w_l, ε_{l,t}}(e) = f_{w_l}(e) + ε_{l,t}
    
    # Pass 2: perturbed forward/backward, actual weight update
    g̃_t = ∇L( (f̃_{w_L,ε_L} ∘ ··· ∘ f̃_{w_1,ε_1} ∘ T)(x_t, y_t) )
    w_{t+1} = Optimizer_Step(w_t, g̃_t)
```

The overhead relative to standard SFT is exactly **one extra forward/backward pass per step** (2× wall-clock time), and a small amount of extra GPU memory to track the per-layer perturbations (~0.11 GB on top of standard SFT on A100).

### 3.5 LoRA Implementation (Double-LoRA)

At the alignment stage, the pretrained weights are frozen and a LoRA adaptor (rank 8) is trained on alignment data with the Vaccine perturbation procedure. At the user fine-tuning stage, the alignment LoRA is merged into the base model, and a *fresh* LoRA adaptor is trained on user data by standard SFT. The two-adaptor design ("Double-LoRA") keeps the alignment and task-specific components separated, which turns out to reduce harmful scores compared to reusing the same adaptor ("Single-LoRA").

### 3.6 What This Means Mechanistically

The whole construction is an **adversarial robustness argument applied to the residual stream**. The FGSM/SAM connection is literal: Vaccine's inner-loop computation of $\epsilon^*$ is a one-step FGSM on the embedding space of the alignment loss. SAM uses the same trick to find flat loss landscapes for generalisation. Vaccine repurposes it to find embedding directions that are most destabilising for alignment, and then trains the model to resist those directions. The hypothesis is that if the model is robust to the worst-case bounded embedding perturbation, the real-world (unbounded but empirically bounded) perturbation from harmful fine-tuning will also be tolerated.

---

## 4. Alignment Against the Four Immunisation Properties

### ✅ Resistance (primary strength)

Vaccine's core claim is resistance: after alignment with Vaccine, fine-tuning on partially harmful data produces significantly lower harmful scores than standard SFT. The mechanism is designed for resistance — the mini-max objective explicitly trains the model to tolerate alignment-breaking perturbations. Empirically, Vaccine achieves up to 9.8% lower harmful score compared to SFT at 10% harmful ratio, and larger reductions at higher harmful ratios.

### ✅ Stability (well supported)

Fine-tune accuracy (FA) is preserved to within 1.8% of SFT on most tasks. The embedding space is made robust, not destroyed — the model still learns downstream tasks efficiently. The Double-LoRA architecture helps by keeping alignment and downstream adaptors separate.

### ⚠️ Generalisation (partially supported)

Vaccine tests across three models (Opt-2.7B, Llama2-7B, Vicuna-7B) and four downstream tasks (SST2, AGNEWS, GSM8K, AlpacaEval), showing consistent reduction in harmful score. This demonstrates in-distribution generalisation. However:
- The alignment data (BeaverTails) and the harmful fine-tuning data are drawn from the same source. **Cross-domain generalisation** — does immunity trained on toxic text transfer to, say, bioweapon queries? — is not tested.
- The perturbation budget $\rho$ is a hyperparameter that the defender must set. A sufficiently large harmful data ratio (e.g., $p=0.2$) still causes substantial alignment-breaking, indicating that generalisation degrades under strong attack.

### ❌ Trainability (the missing piece)

Vaccine does not explicitly test whether the immunised model can be fine-tuned on a *benign* downstream task with the same efficiency as an unimmunised model. The FA results confirm the model can be fine-tuned and still achieve good task accuracy, but this is not the same as testing whether the Vaccine alignment makes benign fine-tuning harder. Combining Vaccine with EWC at fine-tuning time further reduces harmful scores but at a substantial FA cost (up to 39.2% for SST2), suggesting the robustness introduced by Vaccine is somewhat incompatible with highly aggressive further training. **Trainability in the formal sense** (same convergence rate for benign tasks as unimmunised model) is not demonstrated.

---

## 5. Mechanistic Commonalities with Other Approaches

| Method | Mechanistic family | How the "adversary" is simulated | What is perturbed |
|---|---|---|---|
| **Vaccine** | Embedding-space adversarial robustness | FGSM-style one-step gradient ascent on $\mathcal{L}_{align}$ in embedding space | Hidden embeddings at every layer |
| **SAM** (Foret 2020) | Weight-space sharpness | FGSM on weight space to find the sharpness direction | Weights directly |
| **TAR** (Tamirisa 2024) | Weight-space meta-learning | Full inner SGD loop (multi-step) simulating the attacker | Weights via adversarial fine-tuning steps |
| **Booster** (Huang 2024) | Harmful-perturbation attenuation | Gradient ascent on harmful data loss; regulariser reduces the resulting gradient step | Weights, via a harmful perturbation regulariser |
| **RepNoise** (Rosati 2024) | Representation noising | MMD-based pushing harmful embedding distribution toward Gaussian noise | Harmful data hidden states |
| **VAA** (Chen 2025) | Group DRO + SAM | FGSM-style SAM inner step per data group; EXP3 sampler upweights vulnerable groups | Weights, per-group adaptively |
| **T-Vaccine** (Liu 2024) | Layer-wise Vaccine | Same as Vaccine but selects top-k layers by gradient norm; freezes others | Hidden embeddings at selected layers only |

**The core gradient ascent pattern** — first backward pass to find the worst-case perturbation direction, second backward pass to update weights against it — is shared by Vaccine, SAM, Booster, and VAA. The main axes of variation are: (i) *where* the perturbation lives (embedding space vs. weight space), (ii) *how many steps* the inner adversary takes (one-step FGSM vs. multi-step PGD), (iii) *what data* drives the adversary (alignment data only, as in Vaccine, vs. harmful data as in Booster), and (iv) *how the perturbation budget is allocated* (uniform across layers as in Vaccine vs. layer-selective as in T-Vaccine vs. data-group-selective as in VAA).

---

## 6. Results and Significance

### Headline numbers (Llama2-7B, 10% harmful ratio, SST2 unless noted)

| Metric | Non-Aligned | SFT | EWC | Vaccine |
|---|---|---|---|---|
| Harmful Score ↓ | 82.40 | 52.80 | 50.60 | **42.80** |
| Fine-tune Accuracy ↑ | 94.00 | 94.80 | 87.40 | **93.00** |

- Vaccine reduces HS by **~10 pp** over SFT and by **~8 pp** over EWC, while paying only ~1.8 pp in FA compared to SFT (vs. EWC's 7 pp FA loss).
- Gains are consistent across Opt-2.7B, Llama2-7B, and Vicuna-7B, and across SST2, AGNEWS, GSM8K, and AlpacaEval.
- The benefit is larger at higher harmful ratios: at $p=0.2$, Vaccine's advantage over SFT grows substantially.
- Larger models benefit more: for AGNEWS, Vaccine achieves a 2 pp improvement over SFT for Opt-2.7B but an **11 pp improvement** for Llama2-7B.

### Significance relative to the field

At the time of publication, Vaccine was the **first alignment-stage defence against HFT that required no knowledge of user data**. Prior approaches were either (a) fine-tuning-stage solutions (EWC, Freeze) requiring per-request overhead, (b) meta-learning solutions (MAML, Ripple) requiring access to user data during alignment, or (c) post-hoc fixes. Vaccine's claim — that a one-time alignment-stage modification can provide persistent resistance — is the founding argument of the entire alignment-stage defence programme. Subsequent papers (Booster, T-Vaccine, CTRL, VAA) all accept this framing and improve upon Vaccine's specific mechanism.

The paper's weaker point relative to later work: Booster (published six months after Vaccine's preprint) reduces harmful scores by an additional 17–20 pp by incorporating harmful data and a gradient-attenuation regulariser, showing that Vaccine's purely alignment-data-based approach leaves significant room for improvement. RepNoise targets the hidden state distribution more aggressively. Neither of these was known at Vaccine's writing.

---

## 7. Future Work

### From the authors

- **Extension to RLHF**: the authors acknowledge that Vaccine was evaluated only on SFT-based alignment; extending the perturbation-aware idea to RLHF reward models is explicitly listed as future work.
- **System-level optimisation**: the 2× wall-clock cost comes entirely from the second forward/backward pass. The authors suggest sparsification, quantisation, or factorisation of the gradient/perturbation in one of the two passes.
- **Incorporating harmful data**: the authors note that using only the alignment dataset "may not be sufficient enough to counter the harmful attack" — a gap explicitly filled by Booster.
- **Iterative vs. one-step inner optimisation**: comparing Vaccine's one-step FGSM approximation to multi-step PGD (as in LAT) is acknowledged as a natural follow-up.
- **Safety basin analysis**: they cite Peng et al.'s safety basin concept as a potential geometric lens for visualising why Vaccine works, suggesting a theoretical grounding of HED in terms of loss landscape geometry.

### According to the current state of the art

- **Cross-domain and OOD generalisation** is the most urgent open problem. VAA (2025) partially addresses this by using Group DRO to up-weight the most vulnerable subsets of alignment data, but cross-domain transfer (bioweapons ↔ hate speech) remains understudied.
- **RL-based attacks** break alignment more efficiently than SFT-based attacks (lower entropy gradient, better safety-capability Pareto frontier). Vaccine and its descendants were all designed for and tested against SFT-based HFT. Whether embedding-perturbation robustness transfers to RL-based attacks is unknown.
- **The ρ hyperparameter problem**: Vaccine's defence quality is sensitive to $\rho$. Larger $\rho$ gives lower harmful scores but harms fine-tune accuracy. No principled way to set $\rho$ without oracle knowledge of future attack strength is proposed. This is an instance of the general mismatch between the calibrated defender and the uncalibrated attacker.
- **Trainability formalisation**: none of the Vaccine-family papers formally tests whether immunised models satisfy the trainability condition (same convergence rate for benign fine-tuning). This is a significant gap for the practical deployment narrative.
- **Integration with mechanistic interpretability**: T-Vaccine's layer selection heuristic (by gradient norm) is a step toward identifying *which* layers carry safety-critical information. Probing whether the Vaccine-hardened layers correspond to features identified by interpretability tools (e.g., linear probes for harm-related directions) would connect the immunisation programme to the broader mechanistic interpretability literature.
