# Paper Notes — LAPT
**"Probing the Robustness of Large Language Models Safety to Latent Perturbations"**
Gu et al., arXiv:2506.16078, June 2025

---

## 1. Quick Mechanistic Summary

This paper makes three contributions bundled together, and it is important to keep them separate because they have different natures:

**ASA (Activation Steering Attack)** — the diagnostic tool. A perturbation $\delta$ drawn from $\mathcal{N}(0, I_d)$ is normalised to match the first two moments of a target hidden activation $h_t^{(l^*)}$ (instance normalisation), and injected at a specific transformer layer $l^*$ and generation step $t$. The perturbed activation $h'^{(l^*)} = h^{(l^*)} + \delta'$ replaces the clean one and is propagated forward. The key finding is that this random injection, applied repeatedly at every generation step, causes already-aligned models to output harmful completions with very high probability — without any gradient optimisation, any crafted prompt, or any labelled data.

**ASAgrad** — a gradient-guided variant. Instead of random noise, the perturbation direction is the sign of the gradient of the teacher-forced loss on a harmful target suffix $y^*$, computed at inference time via a single forward-backward pass. This is FGSM applied to intermediate activations rather than input embeddings. It substantially improves attack success rates and provides mechanistic insight: the NLL loss surface is much sharper along this direction than along random directions, revealing that alignment creates a thin-walled basin in the latent space.

**LAPT (Layer-wise Adversarial Patch Training)** — the defence. Having identified which layers are most vulnerable (via the ASABench benchmark), LAPT fine-tunes the model by injecting the same class of random perturbations into those fragile layers and training with standard cross-entropy against the clean safe response. The model learns: *"even if my hidden state at layer $l$ is randomly displaced, I should still produce the safe output."* A model interpolation step blends the hardened model back toward the original to recover any degraded general capability.

The paper's central mechanistic claim: **current alignment shapes the input-output mapping but does not impose local robustness constraints on the latent space.** Safety is enforced at the surface; the residual stream beneath remains structurally vulnerable.

---

## 2. Timeline Positioning

### What LAPT Inherits

**From Vaccine / T-Vaccine (Huang et al., 2024; Liu et al., 2025):** The core idea of injecting perturbations into hidden states *during training* and asking the model to resist their effect is shared. Vaccine introduced uniform perturbation of embedding layers; T-Vaccine refined this by targeting only safety-critical layers (identified by gradient norm). LAPT follows the same "targeted perturbation" logic but focuses on *inference-time latent attacks* rather than fine-tuning attacks, and its "fragile layer" identification is empirical (via ASABench) rather than gradient-norm-based.

**From Latent Adversarial Training — LAT (Sheshadri et al., 2024; Casper et al., 2024):** LAT is the most direct predecessor and is cited in the related work. LAT also injects adversarial perturbations into intermediate activations during training. LAPT's main formal difference is the **normalisation scheme** (instance normalisation to match the activation statistics), the **layer selection via ASABench** rather than uniform treatment, and the **model interpolation** post-processing step.

**From FGSM (Goodfellow et al., 2014):** The gradient-based variant ASAgrad is explicitly cast as an FGSM adaptation to LLM activations, acknowledging the non-differentiability of tokenisation as the reason to operate on activations rather than input tokens.

**From Circuit Breakers (Zou et al., 2024):** The framing of alignment as a representational problem — not just an input-output problem — is shared. Circuit Breakers reroute harmful representations to incoherence; LAPT instead tries to make representations locally robust to perturbation without rerouting them.

### What Is Unique

The paper's genuine contribution is **the attack-benchmark-defence pipeline framed around inference-time activation injection as the threat model**, rather than fine-tuning. Prior work (Vaccine, LAT, Circuit Breakers) primarily defends against *weight modification attacks* (i.e., the attacker fine-tunes the model). LAPT addresses a different adversary: one who modifies the **residual stream at inference time**, which requires no access to gradients through the full model and no training data. This is a weaker threat model in terms of attacker compute, but a more realistic one for white-box API-accessible systems.

The **ASABench** benchmark — 4,862 validated attack instances with precise layer-wise attribution — is a concrete infrastructure contribution not replicated elsewhere in the literature.

The **NLL probe as a diagnostic signal** (repurposing the model's own loss on its safe response as a measure of latent fragility) is an elegant and novel evaluation framing, though the idea of tracking NLL under perturbation is implicitly present in several earlier representation-engineering papers.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 Notation

- Model parameterised by $\theta$; conditional distribution $\pi_\theta(y \mid x)$
- Input sequence $x = (x_1, \ldots, x_{|x|})$, output sequence $y = (y_1, \ldots, y_{|y|})$
- $h_t^{(l)} \in \mathbb{R}^d$ — hidden activation at generation step $t$, layer $l$
- $y_{<t} = (y_1, \ldots, y_{t-1})$ — output prefix before step $t$

### 3.2 Perturbation Normalisation (Eq. 1)

Given raw perturbation $\delta \in \mathbb{R}^d$ (e.g., $\delta \sim \mathcal{N}(0, I_d)$), the normalised perturbation is:

$$\delta' = \mu\!\left(h_t^{(l^*)}\right) + \frac{\delta - \mu(\delta)}{\sigma(\delta)} \cdot \sigma\!\left(h_t^{(l^*)}\right)$$

where $\mu(\cdot)$ and $\sigma(\cdot)$ are the **element-wise mean and standard deviation** across the $d$ dimensions of a single vector. This is instance normalisation (Huang & Belongie, 2017): it standardises $\delta$ to zero mean and unit variance, then rescales it to share the mean and standard deviation of the target activation. The purpose is to keep the perturbed hidden state statistically in-distribution, preventing degenerate generation (the model produces token soup without normalisation, particularly in Llama architectures where perplexity under unnormalised perturbation exceeds $10^5$).

The injection is additive:

$$h_t^{\prime\,(l^*)} \leftarrow h_t^{(l^*)} + \delta'$$

### 3.3 NLL Probe (Eq. 2)

$$\mathcal{L}(x, y) = -\sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$

This is the negative log-likelihood of the model's *original safe response* $y$ given the original prompt $x$, evaluated **under the perturbed model** (i.e., with $\delta'$ injected at each step). A higher NLL after perturbation means the model finds its own safe output less likely — the perturbation has pushed the model toward the boundary of the safety basin.

This is the analogue of the cross-entropy loss on the correct class in image classification adversarial attacks: increasing it corresponds to moving away from the safe output distribution.

### 3.4 Gradient-Based Attack — ASAgrad (Eq. 7)

Given harmful prompt $x$ and target harmful suffix $y^*$, form the pseudo-input $x + y^*$ and compute the teacher-forced loss $\mathcal{L}(x + y^*)$ over the tokens of $y^*$. Backpropagate to obtain $\nabla_{h^{(l)}} \mathcal{L}$. The perturbation is:

$$\delta' = \alpha \cdot \text{sign}\!\left(\nabla_{h^{(l)}} \mathcal{L}(x + y^*)\right)$$

with $\alpha = 1$ by default (after normalisation via Eq. 1). This is a **single-step, inference-time, parameter-free** operation. No weight updates, no iterative search, no auxiliary model. The gradient is computed once and discarded.

The reason harmful suffixes are more effective than refusal suffixes as targets: safety-aligned models have been explicitly trained to suppress $p(y^* \mid x)$ for harmful $y^*$, making the gradient landscape steep in that direction. The gradient sign therefore points precisely away from the alignment constraint.

### 3.5 Multi-Token Perturbation Framework (App. G, Eq. 9-10)

For a set of target generation steps $\mathcal{T} = \{t_1, t_2, \ldots, t_m\}$, inject at each:

$$h_{t_k}^{(l^*_{t_k})} \leftarrow h_{t_k}^{(l^*_{t_k})} + \delta_{t_k}, \quad \forall\, t_k \in \mathcal{T}$$

Due to autoregressive generation, the perturbation at step $t_k$ influences not only $y_{t_k+1}$ but all subsequent tokens through the KV cache. The deviations compound:

$$\Delta z_{t_k} = \hat{z}_{t_k} - z_{t_k}$$

and the cumulative KL divergence:

$$\text{KL}(z_t \| \hat{z}_t) = \sum_i z_t^{(i)} \log \frac{z_t^{(i)}}{\hat{z}_t^{(i)}}$$

increases monotonically with token position $t$ across all tested layers and models, confirming that perturbation effects accumulate throughout generation.

### 3.6 LAPT Training Objective (Section 3.2)

For each input $x$ with corresponding safe response $y$ and fragile layer $l$, inject a normalised random perturbation $\tilde{\delta}$ (via Eq. 1):

$$\tilde{h}^{(l)} \leftarrow h^{(l)} + \tilde{\delta}$$

propagate forward to obtain perturbed logits $\tilde{z}$, and minimise:

$$\mathcal{L}_{\text{LAPT}} = \text{CE}(\tilde{z},\, y)$$

where $y$ is the **original clean safe response** (not a perturbed or adversarial target). The model is asked: *given that your activations were randomly displaced at layer $l$, still produce the safe output.* This is standard supervised fine-tuning on the safety examples, but with the activations perturbed during the forward pass.

### 3.7 Model Interpolation (App. K, Eq. 13)

After LAPT, the hardened model $\theta_a$ is blended with the original model $\theta_b$:

$$\theta_{\text{final}} = \lambda \theta_a + (1 - \lambda)\theta_b, \quad \lambda \in [0, 0.5]$$

$\lambda$ is tuned as the largest value such that CommonsenseQA accuracy of $\theta_{\text{final}}$ remains within 0.05 of the baseline $\theta_b$. Empirically, $\lambda$ ranges from 0.1 (Llama-3.1-8B-Instruct, already close to satisfactory) to 0.5 (Llama-3.2-3B-Instruct and Qwen-2.5-7B-Instruct).

---

## 4. Immunisation Properties — Assessment

Using the four-property framework of Rosati et al. (2024):

### Stability ✅ (Partially Verified)

This is the property LAPT most cleanly validates. General task performance (GSM8K, CommonsenseQA) is preserved within 0.05 of baseline across all models, and the interpolation step is explicitly designed to enforce this. The result is credible.

**Caveat:** two benchmarks is a thin evaluation surface. No evaluation of conversational quality, instruction following on diverse tasks, or calibration is performed.

### Resistance ⚠️ (Partially Verified, Narrow Threat Model)

LAPT reduces PASR substantially (e.g., Qwen-2.5-7B-Instruct: PASR drops from 0.36 to 0.13; Llama-3.2-3B-Instruct: from 0.60 to 0.28). However:

- The adversary is **random Gaussian noise**, which is the weakest possible attack. Against $\text{ASA}_\text{grad}$, the paper shows no defence results — the reader cannot know how LAPT performs under the more effective gradient-guided attack.
- The paper does not test against fine-tuning attacks (the primary threat model of most other immunisation papers). LAPT hardens against inference-time activation injection; it says nothing about resistance to weight modification.
- No evaluation against adaptive attacks — an adversary who knows LAPT is deployed and adjusts their injection strategy accordingly.

### Generalisation ⚠️ (Weakly Verified)

LAPT-trained models are evaluated on AdvBench (held-out 420 samples) and HEx-PHI (Table 7), showing improved safety scores. However:

- All attack surfaces are activation injection of the same statistical type used during LAPT training. Cross-domain generalisation (e.g., does LAPT resistance to latent perturbations also improve resistance to prompt-based attacks or fine-tuning attacks?) is not tested.
- LAPT is trained on ASABench which covers 8 specific models and a specific seed dataset (AdvBench). Out-of-distribution harm categories are not explored.

### Trainability ❌ (Not Evaluated)

This property is entirely absent from the paper. Whether LAPT-hardened models can be fine-tuned on benign tasks with similar efficiency to the original model is not tested. Given that LAPT modifies the model's internal response to perturbations at specific layers, and that model interpolation is used to recover general capability, there is a reasonable concern that trainability may be impaired — but no evidence either way is provided.

### Missing Piece

The paper establishes a proof of concept that **representational hardening against latent injection is possible without catastrophic forgetting of general capabilities.** What it does not establish is whether this hardening translates to resistance against the attacks the broader immunisation literature considers primary (fine-tuning, low-rank adaptation, preference inversion). It sits in a gap between the alignment-robustness literature and the immunisation literature without fully connecting to either.

---

## 5. Mechanistic Commonalities with Other Approaches

### The Shared Skeleton: Perturbation + Train-to-Resist

The deepest commonality across the representation-space immunisation family is a single structural pattern:

> *Simulate an attack on the hidden state during training; train the model to produce safe/useful outputs despite the attack.*

This appears, with variations, in Vaccine, T-Vaccine, RepNoise, LAT, and LAPT. The differences are in:

| Aspect | Vaccine | T-Vaccine | RepNoise | LAT | LAPT |
|---|---|---|---|---|---|
| Where injected | Embedding layer | Selected layers | MLP layer | Intermediate activations | Fragile layers (from ASABench) |
| Layer selection | All | Gradient norm | MLP only | Multiple | Empirical (peak LASR) |
| Perturbation type | Uniform random | Uniform random | Max-loss + Gaussian | Gradient-based adversarial | Normalised Gaussian |
| Training target | Safe output | Safe output | Harmful loss max + Gaussian noise | Safe output | Safe output |
| Primary threat | Fine-tuning | Fine-tuning | Fine-tuning | Latent attacks | Latent injection |

LAPT is closest to LAT but adds the instance-normalisation constraint and the empirical layer selection.

### Gradient Ascent / Adversarial Direction as Common Theme

The broader pattern across the entire field: methods either (a) inject random perturbations and train to be robust (Vaccine, LAPT), or (b) use gradient-based adversarial perturbations for stronger immunity (LAT-adversarial, TAR's inner loop, ASAgrad). The gradient-based variant is consistently more effective but more expensive. LAPT's random-perturbation training is closer to (a), which means it is training against the weakest adversary in its own threat model.

### Model Interpolation as a Recurring Patch

The model interpolation trick (linear interpolation between hardened and original parameters) appears in LAPT, LAT (Sheshadri et al., 2024), and is related to the weight averaging used in TAR. It is essentially a post-hoc stability correction: accept that adversarial training degrades general capabilities, and merge back. This is an engineering workaround, not a principled solution to the stability-resistance trade-off.

---

## 6. Results Summary and Significance

### Key Numbers

| Model | PASR Before LAPT | PASR After LAPT | Avg. PASR Reduction | GSM8K Change |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 0.60 | 0.28 | −0.35 | −0.01 |
| Qwen-2.5-7B-Instruct | 0.36 | 0.13 | −0.20 | −0.04 |
| Llama-3.1-8B-Base | 0.40 | 0.20 | −0.21 | +0.09 |
| Llama-3.1-8B-Instruct | 0.35 | 0.30 | −0.07 | −0.03 |

MASR (the more pessimistic metric) is not reported post-LAPT in Table 3, which is a notable omission — it would show what fraction of prompts remain dangerous on at least one layer after defence.

### Significance vs. the Broader Literature

**Against the fine-tuning-attack literature (Vaccine, TAR, etc.):** Direct comparison is not possible because the threat models differ. LAPT is not claimed to resist fine-tuning attacks, and no experiment tests this. The results are significant *within the inference-time latent injection threat model* but cannot be extrapolated outside it.

**Against prior latent adversarial training (LAT):** LAPT likely improves over LAT on the specific attack (ASArandom) but no head-to-head comparison is provided. The paper cites LAT but does not evaluate it. This is a significant omission given that LAT is the closest methodological predecessor.

**Against circuit breakers:** Circuit breakers achieve near-zero harmful output rates under a wide range of unseen attacks on Llama-3-8B (average ASR 3.8% vs. 76.7% for refusal-trained baseline). LAPT achieves PASR reductions of 20–35 percentage points but residual attack success remains in the 13–30% range. On this metric, circuit breakers are substantially stronger — but they also operate under a broader threat model that includes but extends beyond activation injection.

**Bottom line:** LAPT provides a targeted, lightweight, compute-efficient improvement for a specific and newly-characterised attack surface. Its numbers are meaningful but not state-of-the-art relative to the strongest defences, and it is not evaluated against the attacks those defences were designed for.

---

## 7. Future Work

### From the Authors

The authors' own calls, implicit and explicit:

- Extending LAPT to defend against the gradient-based variant ($\text{ASA}_\text{grad}$), which is more effective but was not tested post-LAPT.
- Integrating LAPT with prompt-based defences (the paper notes ASA's composability with GCG shows layered attacks are more powerful; layered defences should be explored).
- Theoretical analysis of the multi-token perturbation framework (App. G is brief; no convergence or error bounds are given).
- Scaling the evaluation to larger models and broader harm categories.

### From the State of the Art

- **Against fine-tuning attacks:** Can inference-time adversarial training (LAPT-style) also improve resistance to weight-modification attacks? The representations hardened by LAPT may or may not survive fine-tuning. This is the most important open question for LAPT's relevance to the immunisation literature.
- **Adaptive adversaries:** All evaluations use fixed-strategy attackers. An adversary who observes that LAPT is deployed and adjusts the injection to exploit LAPT's blind spots (e.g., non-targeted layers, gradient-based directions not covered by random training) is not evaluated. The "durability" critique (Qi et al., 2025, *On Evaluating the Durability of Safeguards for Open-Weight LLMs*) applies directly: defences evaluated only against their own training distribution cannot be considered durable.
- **Theoretical gap:** LAPT has no formal robustness guarantees. It trains on a finite sample of perturbation directions (random Gaussians) and expects generalisation. No PAC-learning-style bound, no characterisation of the geometry of the hardened latent space. The Hessian-based approaches (Condition Number paper) and the safety basin literature provide more principled frameworks that LAPT does not connect to.
- **Trainability:** Completely unaddressed. The immunisation framework (Rosati et al., 2024) explicitly requires this condition for practical deployment. LAPT should be evaluated under Condition 4 before being considered a complete immunisation method.
- **Cross-threat generalisation:** The "Your Task May Vary" problem — does hardening against Gaussian activation injection generalise to semantic activation steering (concept vectors, contrastive activation addition)? No evidence is provided. Given that ASAgrad already shows gradient-guided directions break LAPT-hardened models more easily (implicitly: LAPT was not trained against gradient directions), the generalisation gap is real and likely large.

### From Criticism Papers

**Qi et al. (2025), "On Evaluating the Durability of Safeguards for Open-Weight LLMs" (ICLR 2025):** This paper establishes a benchmark for evaluating safeguard durability against diverse tampering attacks. Its core critique of the field applies to LAPT: defences evaluated against a single attack class (here: random Gaussian injection) and a small held-out benchmark (ASABench test split, 40% of 4,862 samples) cannot claim durable robustness. Durable evaluation requires testing against adaptive, diverse, and compute-intensive adversaries with white-box model access.

**Wei et al. / "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications":** Safety-critical parameters are remarkably sparse (~3% of weights) and separable from utility-critical parameters. If LAPT's resistance mechanism is localised to a small parameter subspace, a targeted low-rank attack can likely bypass it. The paper does not examine whether LAPT changes the sparsity or separability of safety-critical subspaces.

**Qi et al. (2024), "Safety Alignment Should Be Made More Than Just a Few Tokens Deep":** This paper shows that alignment is primarily enforced over the first few tokens of a refusal response. LAPT, which trains on entire response sequences, may or may not address this shallowness — the paper does not analyse token-position-wise resistance of the LAPT-trained models.

---

*Primary source: Gu et al. (2025), arXiv:2506.16078.*
*Cross-references: Rosati et al. (2024) [immunisation framework]; Sheshadri et al. (2024) [LAT]; Huang et al. (2024) [Vaccine]; Liu et al. (2025) [T-Vaccine]; Zou et al. (2024) [Circuit Breakers]; Qi et al. (2025) [durability]; Tamirisa et al. (2025) [TAR].*
