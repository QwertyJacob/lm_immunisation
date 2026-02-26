# TAR — Tamper-Resistant Safeguards for Open-Weight LLMs
**Tamirisa, Bharathi, Phan et al. — ICLR 2025**
*Lapis Labs · UIUC · UCSD · UC Berkeley · CMU · Harvard · U Chicago · Gray Swan AI · Center for AI Safety*

---

## 1. Quick Mechanistic Summary

TAR's core mechanistic bet is simple and bold: **if you train a model to be a bad starting point for gradient-based fine-tuning on harmful content, you make tampering hard by design rather than by obscurity.**

The method operates in two sequential phases.

**Phase 1 — Install an Initial Safeguard.** Before any tamper-resistant training occurs, a conventional safeguard is baked in. For weaponization knowledge restriction, this is a *Random Mapping* safeguard derived from representation engineering: the residual stream activations for hazardous-domain inputs are steered toward random, semantically uninformative vectors using a cosine-distance loss. This collapses hazardous knowledge representation in activation space without corrupting retain-domain representations. For harmful request refusal, this phase is RLHF/DPO as usual. The key point is that **the Phase 1 safeguard does not need to be tamper-resistant itself** — it just needs to establish a low pre-attack safety\_metric. Phase 2 will do the hard work.

**Phase 2 — Adversarial Meta-Learning Outer Loop.** Starting from the Phase 1 checkpoint, TAR runs an outer optimisation loop that simulates many SFT attackers in the inner loop and then updates the *outer model* to be a worse starting point for those attackers. The crucial mechanistic move is that the outer loss is **negative entropy over the attacked model's output distribution**, not negative cross-entropy. This forces the attacked model toward maximum-entropy (uniform) output distributions on hazardous queries, rather than simply driving up its loss on specific token labels. The retain-set loss additionally anchors residual stream hidden states to their Phase 1 positions, preventing capability collapse.

The net result is a model whose weight-space geometry places it at a local minimum of the tamper-resistance loss that is simultaneously a local *maximum* of the adversary's expected progress — a **saddle point engineered to frustrate fine-tuning**.

---

## 2. Timeline Positioning

TAR (ICLR 2025, ArXiv late 2024) arrives at a specific moment in the immunisation literature:

**What it inherits:**

- **MLAC (Henderson et al., 2023)** is the direct conceptual ancestor. MLAC introduced bi-level meta-learning to prevent BERT-style classifiers from learning harmful tasks, using a negated adversary loss in the outer loop. TAR is explicitly the MLAC idea scaled and adapted for autoregressive LLMs — the authors even implement an `MLAC-AR` baseline to measure their own progress beyond MLAC.
- **Representation Engineering / Circuit Breaking (Li et al., Zou et al., 2024)** provides the Phase 1 safeguard for knowledge restriction. The retain loss in Equation 2 is also RepE-inspired: it penalises residual stream drift away from the Phase 1 checkpoint.
- **First-order MAML (Finn et al., 2017; Nichol et al., 2018)** provides the gradient approximation machinery. TAR uses a straight-through estimator to backpropagate through the inner-loop attack, which in the SFT setting reduces to a first-order MAML approximation.
- **SOPHON (Deng et al., 2024)**, a concurrent method for image classifiers and diffusion models, pursues a nearly identical meta-learning idea and is compared directly.

**What makes TAR unique:**

The critical departure from all predecessors is the **entropy loss as the tamper-resistance objective**. Negating cross-entropy (as MLAC does) only forces high loss on the specific labels present in the forget set — the adversary can recover at later training steps once those labels become predictable again. Maximising entropy over the entire output vocabulary forces the adversary's model toward *uniform random output*, which is a much more structurally resistant target: it cannot be undone by simply re-learning a concentrated distribution over correct tokens. This is the key insight that makes TAR meaningfully better than MLAC-AR.

TAR also appears slightly before the *Immunisation Definition Paper* (Rosati et al., EMNLP 2024) formally codified the four immunisation properties, so it is implicitly validating those properties empirically without having used them as a design framework.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 Threat Model and Notation

Let $\theta$ denote pre-trained LLM parameters. After applying an initial safeguard $G$, the defender releases $\theta_G$. A **tampering adversary** has unrestricted access to $\theta_G$ and applies a compute-bounded attack $\text{attack}: \theta_G \mapsto \theta_G'$ (concretely, 500 steps of SFT) to minimise a safety\_metric. The defender wants $\text{safety\_metric}(\theta_G')$ to remain high across a broad set of test-time adversaries $\mathcal{A}_\text{test}$.

Two evaluation domains are considered:

- **Weaponization knowledge restriction**: safety\_metric = error rate on WMDP forget-set MCQs; capabilities\_metric = accuracy on MMLU retain-set.
- **Harmful request refusal**: safety\_metric = complement of HarmBench ASR; capabilities\_metric = MT-Bench score.

### 3.2 Phase 1 — Initial Safeguard (Knowledge Restriction)

Let $h_\theta(\mathcal{D})$ denote the distribution of post-decoder-layer residual stream activations for inputs from distribution $\mathcal{D}$. The Random Mapping safeguard minimises a loss that pushes activations for hazardous inputs $\mathcal{D}_\text{forget}$ toward a randomly sampled target vector $r \sim \mathcal{N}(0, I)$, while keeping retain-set representations anchored. Concretely:

$$\mathcal{L}_\text{RM} = \mathbb{E}_{x \sim \mathcal{D}_\text{forget}} \left[ 1 - \cos\!\left(h_\theta(x),\, r\right) \right] + \mathbb{E}_{x \sim \mathcal{D}_\text{retain}} \left[ \mathcal{L}_\text{LM}(\theta, x) \right]$$

This creates a model $\theta_{G_0}$ that has near-random-chance performance on hazardous WMDP questions (≈25%) but preserved MMLU performance. However, fine-tuning trivially recovers this because the random-mapped representations are just a linear subspace rotation — not a structural barrier.

### 3.3 Phase 2 — Tamper-Resistance Training

**The Outer Objective.** Define:
- $\mathcal{D}_\text{TR}$: a proxy safety dataset (hazardous forget set for knowledge restriction, refusal pairs for refusal).
- $\mathcal{D}_\text{retain}$: a capabilities proxy dataset (instruction-tuning data).
- $\mathcal{A}_\text{train}$: a distribution of SFT-based tampering adversaries.
- $\mathcal{L}_\text{TR}$: the tamper-resistance loss (details below).

The defender solves:

$$\min_{\theta} \;\; \lambda_\text{TR} \cdot \mathbb{E}_{\text{attack} \sim \mathcal{A}_\text{train}} \Big[ \mathcal{L}_\text{TR}\!\left(\text{attack}(\theta);\; \mathcal{D}_\text{TR}\right) \Big] + \lambda_\text{retain} \cdot \mathcal{L}_\text{retain}(\theta;\; \mathcal{D}_\text{retain}) \tag{1}$$

The defender's outer loss is evaluated **on the post-attack model**, not on $\theta$ itself. This is the bi-level structure: the defender optimises over what happens *after* the adversary has moved.

**The Retain Loss.** A representation-engineering-anchored loss preserves capabilities:

$$\mathcal{L}_\text{retain}(\theta;\; \mathcal{D}_\text{retain}) = \mathbb{E}_{x \sim \mathcal{D}_\text{retain}} \left[ \mathcal{L}_\text{LM}(\theta, x) + \left\| h_\theta(x) - h_{\theta_{G_0}}(x) \right\|_2^2 \right] \tag{2}$$

The first term is standard language modelling cross-entropy on the retain set. The second term is an $L_2$ penalty on residual stream drift from the Phase 1 checkpoint $\theta_{G_0}$. This is a form of **activation-space regularisation** — it forces the outer loop to preserve the geometric structure of retain-set representations in hidden state space, not just in output space.

**The Tamper-Resistance Loss — The Critical Choice.**

For **weaponization knowledge restriction**, prior work (MLAC) used:

$$\mathcal{L}_\text{TR}^{\text{MLAC}} = -\mathcal{L}_\text{CE}(\text{attack}(\theta);\; \mathcal{D}_\text{TR})$$

i.e., just negating cross-entropy. The problem is that this only creates high loss at *specific label positions* in the forget set. During adversary training, once the adversary has moved a few steps, gradient signal at those positions saturates, the loss is "won" early, and the adversary is free to recover hazardous knowledge at later steps.

TAR's fix: use **negative entropy** of the output distribution:

$$\mathcal{L}_\text{TR}^{\text{TAR}} = -H\!\left(p_{\text{attack}(\theta)}(\cdot \mid x)\right) = \sum_{v \in \mathcal{V}} p_v \log p_v \tag{3}$$

Minimising this outer loss means the outer loop pushes the attacked model's output distribution on hazardous inputs toward **maximum entropy** — a uniform distribution over the entire vocabulary. This is structurally harder to escape than high cross-entropy on specific labels: to recover hazardous knowledge, the adversary must concentrate probability mass on the correct tokens out of $|\mathcal{V}| \approx 128,\!000$, fighting the outer loop's push toward maximum diffusion. Figure 3 in the paper shows this dramatically: with cross-entropy loss, adversary loss in the inner loop falls after ∼100 steps; with entropy loss, it remains flat at maximum entropy for all 1000 test-time steps.

For **harmful request refusal**, $\mathcal{L}_\text{TR}$ is chosen as the DPO loss on refusal preference pairs, which more naturally captures the preference-based nature of that task.

**The Inner Loop — Approximating Gradient Through the Attack.**

The attack $\text{attack}(\theta)$ applies $K$ steps of SFT. Because it is not feasible to differentiate through this computation explicitly at LLM scale, TAR uses a **straight-through estimator**:

$$\text{attack}(\theta) = \theta + \text{attack}'(\theta), \quad \frac{\partial \mathcal{L}_\text{TR}(\text{attack}(\theta))}{\partial \theta} \approx \frac{\partial \mathcal{L}_\text{TR}(\theta')}{\partial \theta'}$$

where the gradient of $\mathcal{L}_\text{TR}$ evaluated at the post-attack weights $\theta'$ is passed straight through to $\theta$ without differentiating through the inner optimisation steps. In the SFT setting, this approximation is equivalent to **first-order MAML** (FOMAML): it ignores second-order terms involving the Hessian of the inner-loop loss. The resulting gradient estimate is biased but computationally tractable at LLM scale.

**The Full Algorithm (Algorithm 1).**

```
θ₀ ← Apply Initial Safeguard to θ

for i = 1 to N:                                  # outer loop steps
    g_TR ← 0
    Sample x_TR ~ D_TR
    
    for k = 1 to K:                              # simulate K attackers
        Sample attack ~ A_train
        g_TR ← g_TR + (1/K) ∇_{θ_{i-1}} L_TR(attack(θ_{i-1}), x_TR)
    end for
    
    Sample x_r ~ D_retain
    g_retain ← ∇_{θ_{i-1}} [L_LM(θ_{i-1}, x_r) + ‖h_{θ_{i-1}}(x_r) − h_θ(x_r)‖²]
    
    θᵢ ← θ_{i-1} − η (λ_TR · g_TR + λ_retain · g_retain)

return θ_N
```

Key hyperparameters in practice: $N = 100$ outer steps, $K = 64$ inner adversaries, $\lambda_\text{TR} = 4.0$, $\lambda_\text{retain} = 1.0$, outer LR $\eta = 6 \times 10^{-5}$ with ScheduleFree AdamW.

**Why this is inverted MAML.** Standard MAML seeks $\theta$ that is *close* to optimality (low loss) after a few gradient steps on many tasks. TAR seeks $\theta$ that is *far from optimality* (high entropy loss) on adversaries' tasks even after many gradient steps. The inner loop length $K$ acts in the opposite direction from standard MAML: larger $K$ gives the adversary more room to try to recover, and TAR must resist all of it. Accordingly, increasing $K$ increases robustness at the cost of compute (see Figure 7 in the paper), which is the opposite trade-off from standard meta-learning.

---

## 4. Immunisation Properties — What TAR Covers and What It Misses

### Resistance ✅ (Primary Claim)

TAR's central empirical result is **weak resistance**: it dramatically increases the number of fine-tuning steps, attack hyperparameter combinations, and attack algorithms that an adversary must exhaust before hazardous accuracy begins to recover. Across 26 adversaries using 500 SFT steps, TAR keeps Biosecurity WMDP forget accuracy near random chance (35.2% average post-attack vs. ~62% for baselines). The loss plateau phenomenon (Figure 6) — adversary cross-entropy staying near max entropy for up to 1000 steps — is the mechanistic signature of this resistance. **Strong resistance** (mathematically proven impossibility of recovery) is not claimed and is likely impossible to prove in this framework.

### Stability ✅ (Addressed, with a Caveat)

The $L_2$ residual stream anchoring in $\mathcal{L}_\text{retain}$ and the LM cross-entropy on retain data are specifically designed for stability. Retain MMLU drops by ~10.6% on average relative to the no-defense baseline, which is a non-trivial degradation, but TAR is the first tamper-resistant method that manages to preserve capabilities at all while achieving meaningful resistance. For harmful request refusal, MT-Bench drops by 1.7 points. Stability is demonstrated, not just claimed.

### Generalisation ✅ (Partially Demonstrated)

TAR generalises to adversaries unseen at training time — specifically, adversaries using more inner-loop steps than the K=64 used during training, adversaries with different learning rates, different datasets, and different optimisers. The key result (Figure 6) shows that training with K=64-step inner adversaries produces a model that resists 1000-step test-time adversaries. Cross-domain generalisation (training on one hazardous domain, resisting attacks on another) is not systematically evaluated. The adversary distribution $\mathcal{A}_\text{train}$ must be representative enough of test-time adversaries — TAR is explicitly sensitive to this (PEFT attacks and out-of-distribution LR attacks expose gaps).

### Trainability ✅ (Demonstrated in Appendix)

Table 3 shows that TAR-safeguarded models fine-tuned on benign economics data (unrelated to the safeguard) improve on economics MMLU by ~5 points without recovering hazardous knowledge. This is a crucial finding: **structural tamper-resistance is compatible with benign downstream fine-tuning**, at least in the limited setting tested. This was not obvious — it was plausible that the outer loop would also damage the gradient landscape for benign tasks.

### Missing Piece

The single largest gap is **input-space robustness**. TAR explicitly ignores input-space jailbreaking adversaries and does not optimise against them. Pre-attack performance against prompt-based jailbreaks is comparable to baselines but not better. A complete immunisation solution would need to combine TAR's weight-space resistance with input-space robustness. Additionally, the 10.6% retain accuracy drop is a real cost, and the sensitivity to the specific adversary distribution during training (e.g., PEFT attacks break through) is a known limitation acknowledged by the authors.

---

## 5. Mechanistic Commonalities with Other Approaches

**Bi-level optimisation as the shared skeleton.** TAR, MLAC, SOPHON-AR, and (in a different form) Antidote all use a bi-level optimisation structure: an inner loop simulates the adversary, and the outer loop updates the defended model. The differences are almost entirely in what objective the outer loop minimises and how the gradient is approximated. TAR's entropy loss, MLAC's negated cross-entropy, SOPHON's inverse cross-entropy loss, and Antidote's DPO-based outer loss are all variations on this theme.

**Straight-through / first-order gradient approximation.** Because differentiating through a multi-step SFT inner loop is computationally prohibitive at LLM scale, all meta-learning approaches use some form of gradient approximation. TAR uses a straight-through estimator (= FOMAML in the SFT case). Antidote uses a similar first-order approximation with LoRA adversaries. This is the universal practical compromise — nobody is computing second-order terms (true MAML Hessians) at LLM scale.

**Residual stream anchoring as the retain mechanism.** TAR's $L_2$ penalty on residual stream activations (Equation 2) is mechanistically similar to Vaccine's perturbation-invariant embedding alignment and RepNoise's Gaussian anchoring of harmful representations. The shared intuition is that anchoring hidden states rather than (only) output distributions is a more structural way to prevent capability drift. The difference is that TAR anchors *retain-set* representations while RepNoise anchors *harmful-input* representations — they are anchoring opposite sides of the safety/capability boundary.

**Entropy maximisation as the defence objective.** The specific choice of negative entropy as $\mathcal{L}_\text{TR}$ is TAR's unique contribution within the meta-learning family. Circuit-breaking (Zou et al., 2024) also uses a cosine-distance loss that implicitly pushes hazardous representations toward random vectors (Phase 1 of TAR), but does not incorporate any adversarial outer loop. The entropy objective is TAR's answer to the question: "what does it mean for a model to be *maximally resistant* to re-learning, independent of the specific labels being re-learned?"

---

## 6. Results Summary and Significance

### Weaponization Knowledge Restriction

On Llama-3-8B-Instruct, across 26 adversaries and three hazardous domains (Biosecurity, Chemical Security, Cybersecurity):

- **TAR average post-attack forget accuracy**: ~35.2% (Bio), ~27.1% (Chem), ~28.6% (Cyber) — all near random chance (25%).
- **Best prior baseline** (RMU or LLMU): post-attack forget accuracy climbs to 55–70% for most adversaries, meaning hazardous knowledge is substantially recovered.
- **Retain MMLU**: TAR drops ~10.6% vs. the no-defense baseline. This is a real cost, but prior methods that achieved this level of unlearning typically showed even larger capability degradation.
- **Benign fine-tuning compatibility** (Table 3): +5pp MMLU improvement on economics domain without hazardous knowledge recovery.

### Harmful Request Refusal

On Llama-3-8B-Instruct across 5 tampering attacks:

- **TAR post-attack HarmBench ASR**: 63.9% (lower is better).
- **Refusal Trained baseline**: 72.5%. **RR**: 84.8%. **R2D2**: 78.3%.
- **MT-Bench**: TAR 6.3 vs. baseline 8.1 — a meaningful drop, acknowledged.

### Significance

TAR is the **first published method** to demonstrate meaningful tamper-resistance against a diverse red-team of 26+ adversaries for autoregressive LLMs. Prior work on tamper-resistance existed only for image classifiers and BERT-style models. The paper's main claim — that "progress on tamper-resistance is possible" — was genuinely uncertain before this work. The Qi et al. (2024) evaluation confirms that TAR's robustness is real (after correcting for a data contamination issue in the initial release), though with nuances around dataloader seed sensitivity at 1000+ attack steps.

The comparison against MLAC-AR and SOPHON-AR (Table 12) is particularly important: those methods, which share the meta-learning skeleton, fail to maintain both capability and tamper-resistance simultaneously. TAR's entropy loss + Phase 1 safeguard combination is what makes the difference.

---

## 7. Future Work

### From the Authors

- **PEFT robustness**: TAR has "varying robustness" to LoRA/QLoRA-style PEFT attacks. The authors believe targeted patching of these vulnerabilities is possible (Appendix D.2 shows this is tractable).
- **Input-space + weight-space joint defence**: TAR does not explicitly optimise for input-space jailbreaking robustness; the authors flag this as a natural extension.
- **Scalability to larger models**: All experiments use Llama-3-8B. Whether the Phase 2 outer loop scales gracefully to 70B+ parameter models is left open.
- **Expanding the adversary distribution**: The method's robustness depends on how representative $\mathcal{A}_\text{train}$ is. More systematic adversary curation — including RL-based attacks, not just SFT — is an obvious direction.
- **Capability preservation**: The 10.6% retain accuracy cost is acknowledged; future work should reduce this gap.

### From the State of the Art

- **RL-based tampering attacks**: The *Evaluating Durability of Safeguards* framework (Qi et al., 2024) and the *ET paper* explicitly point out that RL-based attacks (which surpass SFT's Pareto frontier and operate with lower response entropy) may circumvent defences trained only against SFT attackers. TAR's inner adversary set is restricted to SFT, leaving RL-based tampering as an open vector.
- **Adversary distribution coverage**: The immunisation definition paper (Rosati et al., EMNLP 2024) and the Brittleness paper both stress that evaluation across a **wide range** of attack algorithms, datasets, and hyperparameters is necessary for confidence. The Qi et al. (2024) durability paper found dataloader seed sensitivity in the initial TAR release — a reminder that what looks like robustness can be a training artefact. After the fix, the robustness claim is reinstated for ≤500 steps but remains variable beyond that.
- **Cross-domain generalisation**: None of the current evaluations test whether a TAR model safeguarded against Biosecurity knowledge resists attacks that use, say, cybercrime data — a gap directly called out in the immunisation definition paper's generalisation condition.
- **Theoretical resistance bounds**: The immunisation definition paper explicitly calls for theoretical bounds on the number of training steps required to break a defence. TAR provides no such bound; the loss plateau is empirical and eventually does fall. Understanding *when* and *why* it falls — and whether the collapse is thermodynamically inevitable — is a deep open question.
- **The empty-shell problem**: Aggressive outer-loop training risks creating a model that satisfies tamper-resistance metrics by having its general capabilities hollowed out rather than by genuinely internalising safety. The 10.6% retain accuracy cost is a warning sign. Future work must demonstrate that the retained capabilities are real and diverse, not just performance on MMLU as a proxy.
