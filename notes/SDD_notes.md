# Paper Notes: SDD — Self-Degraded Defense against Malicious Fine-tuning

**Authors:** Zixuan Chen, Weikai Lu, Xin Lin, Ziqian Zeng  
**Affiliation:** South China University of Technology  
**Venue:** arXiv 2025  
**Link:** https://arxiv.org/abs/2507.21182

---

## 1. Quick Summary — Mechanistic Point of View

SDD is a **dataset-level immunisation method**: before releasing an open-weight model, the developer applies a single supervised fine-tuning pass on a carefully constructed dataset where every harmful instruction is paired with a **high-quality but completely irrelevant benign response**.

The mechanistic bet is this: when a malicious actor later fine-tunes the model on harmful `(instruction, harmful_response)` pairs, the optimizer is forced — by the very mathematics of preference optimisation — to *decrease* the model's probability of producing the high-quality irrelevant content it was trained to output. Because that content is high-quality, coherent language, decreasing the model's ability to produce it is equivalent to degrading its general language-generation capability. The model becomes incoherent before it becomes harmful. The self-destruction is not explicitly programmed into the parameters; it is an **emergent consequence of the attacker's own optimisation dynamics**.

No adversarial gradients, no Hessian tricks, no bi-level look-ahead. SDD is the most mechanistically minimal of all immunisation methods: one dataset, one standard SFT pass, one emergent trap.

---

## 2. Timeline Positioning — Inheritance and Uniqueness

### The Landscape at the Time

By the time SDD appears, the immunisation literature has already established several families of techniques:

| Cluster | Representative papers | Core mechanism |
|---|---|---|
| Adversarial meta-learning | MLAC, TAR, NTL, SOPHON | Bi-level optimisation; push params into safety basin |
| Representation engineering | Vaccine, T-Vaccine, RepNoise, Booster, Circuit Breakers | Gradient attenuation, activation noising, residual-stream rerouting |
| Loss-landscape geometry | Condition Number, LoX | Hessian structure; hard-to-learn harmful subspace |
| **Conditional collapse** | **SDD**, CTRAP, SEAM, TokenBuncher | Make MFT cause self-destruction |

SDD is the **founding paper of the conditional collapse cluster** — the conceptual seed of the entire "make it self-destruct" approach.

### What SDD Inherits

- **From Henderson et al. (MLAC):** the broad intuition that a defense can work by making the attacker's optimisation fail to converge, rather than by blocking access or hardening representations.
- **From Qi et al. and Yang et al. (attack papers):** the empirical observation that as few as 100 harmful samples are sufficient to break alignment. SDD accepts this as a hard constraint — the defense must survive even a tiny poisoning budget — and uses it to motivate the need for a structural trap rather than a preference-shift correction.
- **From RepNoise (Rosati et al.):** the idea that safety information embedded in harmful representations can be deliberately made hard to recover. SDD applies this philosophy at the *output distribution* level rather than the hidden-state level.

### What Is Unique to SDD

1. **Response transplant as a trap.** Prior methods tried to make harmful directions unreachable in weight space. SDD instead makes them *self-defeating* at the output level by redefining what the model's "original response" to a harmful query is. That response — benign, high-quality, irrelevant — becomes the anchor that MFT must destroy.

2. **Formal theoretical grounding for the collapse cluster.** SDD provides Theorem 1 (alignment degradation) and Theorem 2 (general capability collapse) — the only formal proofs in the collapse cluster establishing *sufficient conditions* under which MFT provably causes capability loss.

3. **Zero gradient engineering.** Every other method in the collapse cluster (CTRAP, SEAM, TokenBuncher) requires explicit gradient manipulations during immunisation. SDD requires none. The trap is purely data-driven.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 Model Abstraction

SDD borrows the feature-space framework of Lin et al. (2023). An LLM $f$ is modelled as a pair $(\Phi, \mathbf{w})$ where:
- $\Phi \in \{0,1\}^{d_t}$ is a **feature selector** — a binary mask over a $d_t$-dimensional feature space.
- $\mathbf{w} \in \mathbb{R}^{d \times K}$ is a **linear classifier head** mapping to $K$ output classes.
- The final output is $\mathbf{w}^\top (\mathbf{x} \Phi)$.

Features are categorised as either:
- **Invariant features** $\mathcal{V} = \{x_{v,i}\}_{i=1}^{d_v}$ — stable, predictive across distributions, generalise OOD.
- **Spurious features** $\mathcal{S} = \{x_{s,j}\}_{j=1}^{d_s}$ — unstable correlations, do not generalise.

Harmful data is distributionally spurious by nature: it represents a narrow, idiosyncratic behaviour that a general-purpose model has no reason to encode as an invariant. This empirical claim is the load-bearing assumption of the entire theory.

### 3.2 The Fine-Tuning Ensemble Assumption

Let $\bar{f} = (\bar{\Phi}, \bar{\mathbf{w}})$ be the SDD-immunised model and $f^* = (\Phi^*, \mathbf{w}^*)$ the MFT-optimal model. The fine-tuned model $\tilde{f}$ is modelled as a **weight-space ensemble (WSE)**:

$$\tilde{f} = \lambda \bar{f} + (1-\lambda) f^*, \quad \lambda \in [0,1]$$

More precisely, **Assumption 1** states that there exists a near-optimal $f^*$ with:
$$\Phi^* = \frac{\tilde{\Phi} - \lambda \bar{\Phi}}{1-\lambda}, \quad \mathbf{w}^* = \frac{\tilde{\mathbf{w}} - \lambda \bar{\mathbf{w}}}{1-\lambda}$$

such that $\|\xi_t(f_\text{opt}) - \xi_t(f^*)\| \leq \varepsilon$ for $\varepsilon \to 0$. This is a standard interpolation-in-weight-space assumption, consistent with the empirical literature on model merging and linear mode connectivity.

Feature counts are tracked with the following notation (the asterisk replaces the tilde or bar):

| Symbol | Meaning |
|---|---|
| $\bar{n}_v, \bar{n}_s$ | Invariant/spurious features in $\bar{f}$ |
| $\tilde{n}_v, \tilde{n}_s$ | Same for $\tilde{f}$ |
| $n_v^*, n_s^*$ | Same for $f^*$ |
| $n_{vo}^*, n_{so}^*$ | Features shared between $\bar{f}$ and $f^*$ (overlap) |

The composite input to the ensemble model is:

$$\tilde{\mathbf{x}} = \lambda \sum_{\bar{i}=1}^{\bar{n}_v - n_{vo}^*} x_{v,\bar{i}} + \lambda \sum_{\bar{j}=1}^{\bar{n}_s - n_{so}^*} x_{s,\bar{j}} + (1-\lambda)\sum_{i^*=1}^{n_v^* - n_{vo}^*} x_{v,i^*} + (1-\lambda)\sum_{j^*=1}^{n_s^* - n_{so}^*} x_{s,j^*} + \sum_{i=1}^{n_{vo}^*} x_{v,i} + \sum_{j=1}^{n_{so}^*} x_{s,j}$$

And the fine-tuned classifier:
$$\tilde{\mathbf{w}} = \lambda \bar{\mathbf{w}} + (1-\lambda) \mathbf{w}^*$$

### 3.3 Theorem 1 — MFT Breaks Alignment

**Lemma 1** (OOD accuracy upper bound): Under the Small Noise and Orthogonal Features assumptions inherited from Lin et al. (2023), the OOD accuracy of $\tilde{f}$ on task $t$ is bounded by:

$$\xi_t(\tilde{f}) \leq F_p\!\left( \frac{\sqrt{(1-p)(\bar{n}_s + n_s^* + 2n_{so}^*) + \bar{n}_v + n_v^* + 2n_{vo}^*}}{\sqrt{\bar{n}_s + n_s^* + 14n_{so}^*}} \right)$$

where $F_p$ is a monotone function and $p$ is the small-noise parameter.

**Theorem 1 (Alignment degradation):** The drop in alignment accuracy $\xi_A(\tilde{f}) - \xi_A(\bar{f})$ is bounded above by a quantity that is an *increasing function of $n_s^*$*. Since harmful data is spurious by construction, $n_s^*$ is large, making this bound provably negative. MFT **mathematically degrades alignment** — this is not just an empirical observation.

### 3.4 Theorem 2 — General Capability Collapse

**Theorem 2 (General capability collapse):** If the immunised model satisfies:
$$\bar{n}_v > n_v^* \quad \text{and} \quad \bar{n}_s < n_s^*$$

then after MFT:
$$\xi_G(\tilde{f}) < \xi_G(\bar{f})$$

The fine-tuned model is *strictly worse* at general tasks than the immunised model was. This is the formal guarantee of self-degradation.

The two conditions have a clean semantic interpretation:
- $\bar{n}_v > n_v^*$: SDD training on high-quality, semantically rich benign content loads the model with more invariant features than the MFT-optimal model (which is trained on narrow, spurious harmful data) will retain.
- $\bar{n}_s < n_s^*$: The irrelevant but high-quality responses do not reinforce harmful-domain-specific correlations, leaving $\bar{f}$ with fewer spurious features than the MFT target $f^*$ will introduce.

**SDD's entire dataset design is an attempt to engineer these two inequalities into the model without gradient manipulation.**

### 3.5 The Preference-Optimisation Argument

SDD gives a second, complementary argument grounded in preference learning, which makes the self-destruct mechanics immediately transparent.

Frame MFT as a **Bradley-Terry preference optimisation** problem. The attacker wants the model to prefer the harmful response $y_c$ over the model's current output $y_o$:

$$\max_\theta \; p(y_c \succ y_o \mid x) = \max_\theta \frac{\pi_*(y_c|x)/\pi_\theta(y_c|x)}{\pi_*(y_c|x)/\pi_\theta(y_c|x) + \pi_*(y_o|x)/\pi_\theta(y_o|x)}$$

To increase this ratio, the optimizer must simultaneously:
1. Increase $\pi_*(y_c|x)/\pi_\theta(y_c|x)$ → shift model toward harmful output.
2. **Decrease** $\pi_*(y_o|x)/\pi_\theta(y_o|x)$ → reduce model's probability of generating $y_o$.

Step 2 is the trap. Under standard alignment, $y_o$ is a brief refusal (*"I cannot help with that."*). Reducing the probability of producing a single short refusal is a narrow operation — it leaves general language capability intact.

Under SDD, $y_o$ is **high-quality, multi-sentence benign content** — a cooking tutorial, a biographical sketch, a technical explanation. Decreasing the model's probability of producing *that* is equivalent to decreasing its probability of producing well-structured, coherent responses across the board. **The attacker is forced to degrade general language quality as an inescapable side-effect of their attack.**

### 3.6 Dataset Construction

The SDD training dataset $\mathcal{D}_\text{SDD}$ consists of pairs $(x, y_o)$ where:
- $x$ is a harmful instruction drawn from **BeaverTails** (14 harm categories, ~8K entries).
- $y_o$ is a high-quality benign response sampled from **LIMA** or **Alpaca** (response side only, randomly re-matched without corresponding instructions).
- A **SentenceBERT semantic similarity filter** removes any pair where $\cos(\text{emb}(x), \text{emb}(y_o)) > \tau$, preventing accidental topical overlap between the harmful instruction and the benign response.

Training is standard **cross-entropy SFT**:
$$\mathcal{L}_\text{SDD}(\theta) = -\mathbb{E}_{(x,y_o)\sim\mathcal{D}_\text{SDD}} \left[ \frac{1}{|y_o|} \sum_{t=1}^{|y_o|} \log \pi_\theta(y_{o,t} \mid x, y_{o,<t}) \right]$$

No gradient penalty, no second-order terms, no dual-loop structure. The entire immunisation is a single forward-backward pass over this dataset, applicable at any stage of the training pipeline (pre-training, SFT, RLHF stage).

### 3.7 The SDD_reject Variant

To align with AI safety community norms around *explicit* refusal (rather than just generating off-topic content), a simple variant prepends a fixed prefix to every $y_o$:

> *"I refuse to answer your question for responsible and ethical reasons. I provided an irrational answer to your question."*

SDD_reject achieves comparable MFT defense effectiveness while substantially increasing the explicit rejection rate (as measured by GPT-4 evaluation).

---

## 4. Immunisation Properties

### Resistance ✅ (Strong but Conditional)

SDD achieves strong empirical resistance. Even when attackers use **20× the training data** (scaling from 100 to 2,000 harmful samples), SDD-protected models exhibit markedly lower harmfulness scores than unprotected baselines. Theorem 2 provides formal sufficient conditions under which resistance is guaranteed.

The critical caveat: the resistance guarantee is **conditional** on $\bar{n}_v > n_v^*$ and $\bar{n}_s < n_s^*$. SDD has no explicit mechanism that *enforces* these inequalities; it relies on the intuition that training on high-quality irrelevant text tends to produce this feature distribution. The theory says "if these conditions hold, collapse follows" but does not prove they always hold after SDD training. This is the principal theoretical gap.

### Stability ✅ (Verified)

SDD maintains general capabilities under benign fine-tuning (BFT). Because BFT does not target harmful instruction–response pairs, it does not force $\pi_\theta(y_o|x)$ downward, and the self-destruct mechanism is never triggered. Empirically confirmed on MT-Bench and MMLU-class benchmarks.

### Generalisation ⚠️ (Partial)

SDD is trained and evaluated on BeaverTails (14 categories). In-domain generalisation (held-out attack samples from the same 14 categories) is demonstrated. **Cross-domain generalisation** — attacks from harm categories not seen during immunisation — is not systematically evaluated. This is a meaningful gap, since the feature-space theoretical conditions may not hold for OOD harmful data distributions.

### Trainability ✅ (Explicitly Verified)

SDD-immunised models can be benign-fine-tuned with no degradation relative to unimmunised baselines. This is a genuine strength: unlike methods that destroy capability as the price of resistance (RepNoise in strong settings), SDD's self-destruct is *conditional* on attacker intent. A benign user never triggers it.

### The Missing Piece

The single most important gap is **resistance to non-SFT attacks**. The preference-optimisation argument assumes the attacker uses a standard SFT or DPO-style objective where decreasing $\pi_\theta(y_o|x)$ is mechanically necessary. RL-based fine-tuning (demonstrated by TokenBuncher, 2025) can update the policy gradient without forcing this decrease, potentially bypassing the self-destruct entirely.

---

## 5. Mechanistic Commonalities with Other Approaches

### With CTRAP

Both SDD and CTRAP exploit the fact that MFT must decrease $\pi_\theta(y_o|x)$ to succeed. In SDD this is **implicit** (emergent from dataset construction); in CTRAP it is **explicit** (a collapse loss $\ell_\text{Collapse}$ forces the model toward a degenerate constant output $e$, planted in the specific direction of harmful gradients via a bi-level look-ahead). CTRAP is mechanistically SDD with a deliberate, directional, gradient-informed trap instead of an emergent one.

### With SEAM

SEAM operates in **gradient space**: it pushes adversarial gradients and benign gradients to be geometrically opposed (negative cosine similarity), so any harmful fine-tuning step is geometrically destructive to benign performance. SDD operates in **output-distribution space**: it makes $y_o$ so high-quality that any decrease in $\pi_\theta(y_o|x)$ carries downstream damage. Both achieve interference between the harmful and benign objectives, from entirely different spaces.

### With TAR and MLAC

TAR uses a meta-learning bi-level loop where the outer objective maximises *entropy* loss on harmful tasks *after* simulated fine-tuning steps — a gradient-based simulation of "where would one MFT step land me, and how do I make that place bad for the attacker?" SDD achieves the same philosophical goal (make the attacker's gradient work against them) without any gradient computation. The trap is planted through the *data distribution* rather than the *loss landscape*.

### With RepNoise

Both RepNoise and SDD aim to destroy harmful information structure. RepNoise does this by **adding noise to harmful representations** in activation space, making them hard to recover. SDD does this by **replacing the harmful output with irrelevant content** in distribution space, making the replacement expensive to override. RepNoise is a representation-level intervention; SDD is a data-level intervention. The effect — MFT finds no coherent structure to exploit — is analogous.

### The General Pattern

SDD belongs to a broad pattern visible across many immunisation methods: **the defender shapes the model's behaviour in the vicinity of harmful queries, then relies on the attacker's optimizer to backpropagate damage to the attacker's own objective.** The variation across methods is in *where* (weight space, gradient space, activation space, output distribution) and *how explicitly* (deliberate trap vs. emergent consequence) this coupling is installed.

---

## 6. Experimental Results

### Setup

- **Base model:** LLaMA-2-7B-Chat (main); Phi-2 (2.7B) and GLM-3 (6B) as additional backbones.
- **Attack dataset:** BeaverTails-Evaluation (100 samples standard; up to 2,000 samples in stress test), AdvBench.
- **Evaluation metrics:** BeaverTails harmlessness score (GPT-4 judge); MT-Bench; MMLU.
- **Baselines:** Vanilla (aligned, unprotected), Vaccine, T-Vaccine, RepNoise, CTRL.

### Main Results

SDD achieves substantially lower harmful-score post-MFT compared to all baselines. The gap widens as the attacker's data budget increases — the larger the attack, the more the self-destruct fires.

**Defense efficiency:** With only 500 SDD training samples (from AdvBench), SDD defends against attacks using up to 10,000 harmful samples — a **20× asymmetry in data budget**, favouring the defender.

**Capability preservation:** On MT-Bench and MMLU, SDD-immunised models score comparably to the vanilla baseline before any attack. Stability is satisfied.

**Multiple backbones:** Results hold on Phi-2 and GLM-3, suggesting the mechanism is not architecture-specific — it is a property of the SFT optimisation dynamics rather than of transformer architecture details.

**SDD_reject:** Comparable harmlessness scores with substantially higher explicit rejection rate (GPT-4 measured).

### Significance Relative to Other Papers

- SDD is **simpler than TAR/MLAC** (no bi-level loop) and achieves comparable resistance at lower computational cost.
- SDD **outperforms Vaccine and RepNoise** on the harmlessness-after-MFT metric.
- SDD **preserves capability better than RepNoise** in its strongest form, where aggressive noise injection can degrade general performance.
- SDD has the **strongest formal theoretical grounding** among collapse-cluster methods (Theorem 1 + Theorem 2), compared to CTRAP (empirical justification), SEAM (Hessian-free approximation theorem), and TokenBuncher (entropy bound theorem).
- SDD **does not address RL-based attacks** — a gap that TokenBuncher (2025) later demonstrates is critical, and that establishes the frontier that SDD's successor methods must cross.

---

## 7. Calls for Future Work

### From the Authors

1. **Responsible output variant.** Generating irrelevant responses rather than explicitly refusing is misaligned with AI safety community norms. SDD_reject is a patch; the authors call for more principled ways to combine resistance with clear refusal behaviour without sacrificing defense effectiveness.

2. **Cross-domain generalisation.** The defense is trained on BeaverTails (14 categories). Systematic evaluation against OOD attack categories is explicitly flagged as missing.

3. **Adaptive attackers.** An attacker who knows the SDD construction could modify the attack objective to avoid targeting $y_o$. No countermeasure is proposed; this is left as an open problem.

### From the State of the Art

The immunisation literature subsequent to SDD identifies additional gaps:

1. **RL-based attacks.** TokenBuncher (2025) demonstrates that RL fine-tuning bypasses SFT-based self-destruct mechanisms entirely. A future SDD-type method needs an RL-aware formulation — possibly coupling the data-level trap with an entropy-based objective.

2. **Controlled enforcement of Theorem 2's conditions.** The collapse guarantee requires $\bar{n}_v > n_v^*$ and $\bar{n}_s < n_s^*$, but SDD has no explicit training objective that enforces these inequalities. Future work could add an auxiliary regularisation term that explicitly maximises $\bar{n}_v$ and minimises $\bar{n}_s$ on harmful-prompt distributions, turning the sufficient condition into a controlled one.

3. **Gradient-space collapse directionality.** CTRAP improves on SDD by planting the collapse in the *specific gradient direction* a harmful fine-tuner follows. SDD's emergent trap is isotropic — it works regardless of the MFT direction, but provides no guarantee of *how quickly* collapse is triggered. Combining SDD's data simplicity with CTRAP's directional precision is an open engineering challenge.

4. **Pre-alignment base model access.** All collapse-cluster methods — SDD included — are powerless against an attacker who has access to the *pre-immunisation* base model and re-trains from scratch. This is a fundamental architectural limitation of any post-training immunisation strategy. No current method has a solution.

5. **Evaluation completeness.** The immunisation definition framework (Rosati et al., 2024) requires all four conditions to be demonstrated simultaneously: resistance, stability, generalisation, and trainability. SDD's evaluation satisfies stability and trainability, partially addresses resistance, and leaves cross-domain generalisation unexamined. Future immunisation papers should report all four.
