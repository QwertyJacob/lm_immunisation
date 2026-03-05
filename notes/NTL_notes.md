# Non-Transferable Learning (NTL)
**Wang et al., ICLR 2022**
*"Non-Transferable Learning: A New Approach for Model Ownership Verification and Applicability Authorization"*

---

## 1. Quick Mechanistic Summary

NTL is a training-time intervention that deliberately **degrades a model's ability to generalise beyond a designated source domain**, without impairing performance within it. The mechanism has two parallel components:

**Component 1 — Representation Divergence.** The feature extractor Φ is trained so that latent representations of source-domain and auxiliary-domain inputs are maximally distant in RKHS space (measured via MMD with a Gaussian kernel). This is the geometric anchor: if two distributions of representations are far apart, a classifier head adapted to one will transfer poorly to the other.

**Component 2 — Selective Task-Incompetence.** The classifier Ω is trained to increase its KL divergence loss on auxiliary-domain inputs while maintaining low loss on source-domain inputs. From an information-theoretic lens, this drives $I(z; y \mid n=\text{aux}) \to 0$ while preserving $I(z; y \mid n=\text{src}) = I(x; y \mid n=\text{src})$.

Together, both components force the model to encode **domain identity as an irreducible feature** of every representation, making the entire network's utility domain-exclusive. The two regimes are:

- **Target-Specified NTL**: the forbidden domain (auxiliary) is known and directly used.
- **Source-Only NTL**: the auxiliary domain is *generated* via a conditional GAN that synthesises neighbourhood distributions at multiple distances and directions from the source manifold.

---

## 2. Timeline Positioning

### When and where NTL sits

NTL was published at **ICLR 2022**, making it one of the earliest papers in the corpus to formalise domain-exclusivity as a deliberate training objective. It predates the LLM-oriented immunisation literature (e.g., SDM 2023, TAR 2025, Vaccine 2024, RepNoise, Booster, AntiDote) by one to three years. Its primary framing is **IP protection for vision classifiers**, not LLM safety; this is its defining contextual limitation.

### What NTL inherits

| Ancestor Concept | Origin in NTL |
|---|---|
| Information Bottleneck (Tishby et al., 2000) | Inverted IB: NTL *maximises* $I(z;n)$ instead of minimising it |
| Domain Adaptation & Generalization literature | NTL explicitly inverts the DA objective: maximise rather than minimise domain discrepancy |
| MMD as a distributional distance (Gretton et al.) | Used as the tractable proxy to drive representation separation |
| RKHS / kernel methods (Sriperumbudur et al., 2009) | Theorem 2 proof relies on the SVM-margin bound in RKHS |
| Conditional GAN training (CGAN + InfoGAN) | Used for source-only auxiliary domain generation |
| IP watermarking literature | Target-Specified NTL subsumes backdoor-based watermarking as a special case |

### NTL's unique contribution

NTL is the **first work to frame immunity as a geometric property of representations** rather than as an output-layer behaviour. Every previous watermark or ownership verification method operated at the level of predictions or parameters; NTL instead sculpts the **shape of the latent manifold** to be domain-exclusive. This is the conceptual bridge from watermarking to what will later become representation-noising-based immunisation (RepNoise, Vaccine). SOPHON (IEEE S&P 2024) is NTL's direct successor: it augments the geometry argument with explicit non-fine-tunability via MAML-style inner loops.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 Setup

A neural network is split into a **feature extractor** $\Phi$ and a **classifier** $\Omega$. Given a source domain $\mathcal{S} = \{(x,y) \mid x \sim P^S_X, y \sim P^S_Y\}$ and an auxiliary domain $\mathcal{A} = \{(x,y) \mid x \sim P^A_X, y \sim P^A_Y\}$, we write $z = \Phi(x)$ for the latent representation.

---

### 3.2 Information-Theoretic Grounding

**Proposition 1** (Data Processing Inequality applied to IB).

The Markov chain $(y, n) \to x \to z$ (where $n$ is the domain index, a *nuisance*) gives, via DPI and the chain rule:

$$I(z; x) - I(z; y \mid n) \geq I(z; n)$$

**Reading this**: to maximise how much $z$ encodes about domain membership $n$, the right-hand side is pushed up by either (a) keeping $I(z;x)$ large (don't compress the input) and/or (b) making $I(z; y \mid n)$ small (destroy task-useful information in the auxiliary domain). NTL does (b).

---

### 3.3 Loss Design — Component 1: Selective KL Divergence

Define per-domain KL divergence losses for the full model $\Omega \circ \Phi$:

$$\mathcal{L}_S = \mathbb{E}_{x \sim P^S_X}\!\left[D_{\mathrm{KL}}\!\left(P(\Omega(\Phi(x))) \,\|\, P(y)\right)\right]$$

$$\mathcal{L}_A = \mathbb{E}_{x \sim P^A_X}\!\left[D_{\mathrm{KL}}\!\left(P(\Omega(\Phi(x))) \,\|\, P(y)\right)\right]$$

**Theorem 1** (in the paper) formally establishes that increasing $D_{\mathrm{KL}}(P(\hat{y}) \| P(y))$ decreases $I(z; y)$, via the DPI applied to the Markov chain $z \to \hat{y} \to y$ and the monotonicity of $I(\hat{y}; y)$ with respect to $P(\hat{y}, y)$.

The first-stage loss is:

$$\mathcal{L}^*_{\mathrm{ntl}} = \mathcal{L}_S - \min(\beta,\, \alpha \cdot \mathcal{L}_A)$$

- **$\mathcal{L}_S$ is minimised**: source performance is preserved (standard CE/KL on source samples).  
- **$-\min(\beta, \alpha \cdot \mathcal{L}_A)$ is maximised**: the model is pushed to *increase* its auxiliary-domain loss, i.e., to predict incorrectly on auxiliary inputs.  
- **$\beta$** (upper bound) prevents $\mathcal{L}_A$ from dominating and destabilising training.  
- **$\alpha = 0.1$** scales the auxiliary contribution; the min-clipping ensures a ceiling of $\beta = 1.0$.

At this point, only $\Omega$ (the classifier head) has been made domain-sensitive. Φ's representations may still be geometrically close across domains, which means a fine-tuner could recover performance on the auxiliary domain by simply replacing or retraining $\Omega$ with a few labelled samples.

---

### 3.4 Loss Design — Component 2: MMD-Based Representation Separation

**Theorem 2** (in the paper) connects geometric separation of distributions to mutual information.

The Gaussian-kernel Maximum Mean Discrepancy between the source and auxiliary representation distributions is:

$$\mathrm{MMD}(P_{Z|0}, P_{Z|1};\, \exp) = \mathbb{E}_{z,z' \sim P_{Z|0}}\!\left[e^{-\|z-z'\|^2}\right] - 2\,\mathbb{E}_{z \sim P_{Z|0},\, z' \sim P_{Z|1}}\!\left[e^{-\|z-z'\|^2}\right] + \mathbb{E}_{z,z' \sim P_{Z|1}}\!\left[e^{-\|z-z'\|^2}\right]$$

The proof shows that as MMD increases toward saturation, the gap between the means $\mu_0, \mu_1$ of the two distributions widens and their variances $\sigma_0, \sigma_1$ contract (via the RKHS-SVM margin bound of Sriperumbudur et al. 2009). Both effects monotonically increase $I(z; n)$ — formally tracked through the Jensen–Shannon-like integrals of Eq. (13) in the paper.

This justifies maximising MMD as a tractable surrogate for maximising $I(z; n)$.

The **full NTL loss** combines both components:

$$\mathcal{L}_{\mathrm{ntl}} = \mathcal{L}_S - \min\!\left(\beta,\; \alpha \cdot \mathcal{L}_A \cdot \mathcal{L}_{\mathrm{dis}}\right)$$

where

$$\mathcal{L}_{\mathrm{dis}} = \min\!\left(\beta',\; \alpha' \cdot \mathrm{MMD}\!\left(P_{x \sim P^S_X}(\Phi(x)),\; P_{x \sim P^A_X}(\Phi(x));\; \exp\right)\right)$$

The scaling factors $\alpha = \alpha' = 0.1$ and ceilings $\beta = \beta' = 1.0$ are fixed empirically. The product $\mathcal{L}_A \cdot \mathcal{L}_{\mathrm{dis}}$ ensures that both the task incompetence *and* the geometric separation of representations jointly amplify each other in the auxiliary domain.

---

### 3.5 Source-Only Case: GAN-Based Auxiliary Domain Synthesis

When the target domain is unknown, NTL must *manufacture* an auxiliary domain. A conditional GAN (combining CGAN and InfoGAN principles) is trained on source data. Crucially, $G$ is not just trained to match the source; it is **progressively perturbed** to generate distributions at multiple MMD distances ($\mathrm{dis} \in \{0.1, 0.2, \ldots, 0.5\}$) and multiple directions from the source manifold (achieved by sequentially freezing subsets of $G$'s layers during optimisation).

The augmentation loss is:

$$\mathcal{L}_{\mathrm{aug}} = -\min\!\left\{\mathrm{dis},\; \mathrm{MMD}\!\left(P_{x \sim P^S_X}(D_z(x)),\; P_{y \sim P^S_Y}(D_z(G(\mathrm{noise}, y)));\; \exp\right)\right\} + \mathbb{E}_{y \sim P^S_Y}\left[D_{\mathrm{CE}}\!\left(D_m(G(\mathrm{noise}, y)),\, y\right)\right]$$

The first term pushes $G$'s output away from the source (up to distance $\mathrm{dis}$). The second term (cross-entropy on labels from $D$'s multi-class head) prevents semantic collapse — the generated images must still look like plausible samples of the correct label. The diversity of distances and directions is what ensures the resulting NTL model degrades performance across the full neighbourhood of the source manifold, not just one direction.

---

### 3.6 Ownership Verification and Applicability Authorization as Instances of NTL

- **Ownership verification**: a pixel-level mask patch is applied to source images to define the auxiliary domain. The model learns to misclassify patched inputs while performing well on clean inputs. Since this misclassification behaviour is encoded in the *geometry of Φ* (not just in $\Omega$), it is resistant to head-retraining or fine-tuning attacks (FTAL, RTAL, EWC, AU, overwriting, pruning all fail).

- **Applicability authorization**: the source domain itself carries an authorized patch; the union of un-patched source images and GAN-generated neighbourhood data forms the auxiliary domain. The model learns to perform *only* on patched-source data, degrading on everything else.

---

## 4. Alignment with the Four Immunisation Properties

| Property | NTL's coverage | Assessment |
|---|---|---|
| **Resistance** | ✅ Strong. Source-Only NTL degrades non-source performance to ~10–20% accuracy. Target-Specified NTL is resistant to 6 state-of-the-art watermark removal attacks including full fine-tuning (FTAL) and re-initialised classifier retraining (RTAL). | Well-verified empirically. Resistance is encoded in Φ, not Ω, so classifier replacement alone does not break it. |
| **Stability** | ✅ Good. Source performance drops by <2.5% across all experiments. | Verified. The $\mathcal{L}_S$ term explicitly protects source performance. |
| **Generalisation** | ⚠️ Partial. Target-Specified NTL generalises to the chosen target domain. Source-Only NTL generalises across all tested non-source domains (digits, CIFAR10/STL10, VisDA). However, the evaluation is on computer-vision tasks only, and generalisation to semantically diverse, out-of-distribution harm domains (as required in LLM immunisation) is not addressed. | The paper shows breadth across multiple non-source domains but does not frame generalisation as a formal condition, nor test cross-domain generalisation in the immunisation sense. |
| **Trainability** | ❌ Not addressed. The paper does not evaluate whether NTL-processed models remain efficiently fine-tunable on *benign, in-distribution* downstream tasks. This is the critical missing piece. | From the architecture, Φ has been deliberately shaped to encode domain identity, which will likely impair efficient adaptation even to harmless new tasks — a cost the authors do not measure. |

**The missing piece** is trainability: NTL does not ask whether a model immunised with NTL can still be efficiently and safely fine-tuned for legitimate downstream tasks. In the immunisation framework (Rosati et al., 2024), this is an explicit condition that downstream applicants will demand. The geometric compression of Φ around source-domain features is at odds with remaining generally plastic.

---

## 5. Mechanistic Commonalities with Other Approaches

### 5.1 The adversarial two-domain loss structure
Like **MLAC** (Henderson et al., 2023), **TAR** (Tamirisa et al., 2025), and **SOPHON** (Deng et al., 2024), NTL operates on two simultaneously tracked objectives: minimise loss on a "protected" distribution, maximise (or at least not minimise) loss on a "forbidden" distribution. In NTL this is the $\mathcal{L}_S - \min(\beta, \alpha \mathcal{L}_A)$ structure; in TAR it becomes a retain/forget split with entropy loss; in MLAC it is a bi-level meta-learning inner/outer loop. The **shared skeleton** is: *protect performance on D_src, destroy it on D_aux.*

### 5.2 MMD as a distributional regulariser
The use of MMD to measure and drive apart representation distributions appears later in **RepNoise** (Rosati et al., 2024), which pushes harmful representations toward Gaussian noise. The conceptual move is the same — enforce a large distance between the representation distribution of "safe" inputs and "harmful" inputs in RKHS — but the directionality differs: NTL maximises the gap between two data domains, RepNoise pushes one domain toward a fixed reference (white noise).

### 5.3 GAN-based auxiliary generation
The use of a conditional GAN to synthesise the "harmful" or "auxiliary" distribution when it is unknown is a direct antecedent of later work on synthetic poison/harmful data generation used in adversarial training. The key mechanism — generate OOD data close to source but semantically shifted, then use it to train resistance — is a structural ancestor of augmentation strategies in TAR and Booster.

### 5.4 The Hessian/second-order gradient absence
NTL does **not** use Hessian approximations or second-order gradient ascent in its inner loop. This distinguishes it from the MAML-flavoured approaches (MLAC, SDM, SOPHON, TAR). NTL achieves its effect through a **single-level** optimisation that jointly updates Φ and Ω on both domains in one pass. This makes it computationally simpler but also less resistant to high-budget attacks — an attacker with many gradient steps can eventually fine-tune Ω on the target domain, recovering some performance (though Φ's geometry makes this harder than in standard models).

---

## 6. Results Summary and Significance

### Target-Specified NTL
On 5 digit datasets (MNIST, USPS, SVHN, MNIST-M, SYN-D), CIFAR10/STL10, and VisDA:
- **Average relative target-domain performance drop: ~80%** (e.g., USPS→MT: 86.4% → 14.5%)
- **Average source performance drop: <2.5%**
- Resistant to all 6 tested watermark removal attacks (FTAL, RTAL, EWC, AU, overwriting, pruning), with patched-image accuracy remaining at ~10% after all removal attempts

### Source-Only NTL
- **Average non-source performance drop: ~72–83%** across all tested dataset pairs
- Source performance preserved within ~2.5%
- Applicability authorization: model achieves 88–98.5% on authorized patch-equipped source data, <43% on all other inputs

### Significance relative to the corpus
NTL's results are **strong for their time and setting** (vision classification, 2022), but they cannot be directly compared with LLM immunisation benchmarks (harmfulness scores, MMLU, perplexity). The paper's key empirical contribution is showing that resistance is durably encoded in the extractor's geometry, surviving even aggressive fine-tuning attacks — a property that later approaches (TAR, SOPHON, Booster) must work hard to replicate in the much larger parameter spaces of LLMs. NTL also provides the clearest theoretical grounding (IB → Proposition 1 → Theorem 1 → Theorem 2) of any paper in this corpus at the time of its publication.

---

## 7. Calls for Future Work

### From the authors
The authors explicitly identify:
1. **Extension to NLP/LLM tasks**: the paper ends on the prediction that NTL could restrict language model generalization to certain tasks — a direct forecast of the work SOPHON (2024) and later papers would pursue.
2. **Semantic segmentation and object detection**: modalities where generating auxiliary data requires different strategies.
3. **Multi-Task Learning**: restricting what *tasks* a model can learn (not just what data domains), connecting to task-blocking in MLAC.
4. **Combining with cryptography**: NTL + secret-key protection as layered IP security.

### From the state-of-the-art (as of 2025–2026)
1. **LLM adaptation**: The core NTL objective has already migrated to SOPHON (2024), which adds a MAML-based non-fine-tunability term. But NTL's representation-distance argument has not yet been rigorously adapted for autoregressive models operating on token sequences, where the notion of "domain" is far more entangled with semantic content than in image classification.
2. **Trainability**: No work has formally studied whether NTL-processed models satisfy the trainability condition (Rosati et al., 2024). This is a gap: a model that cannot be fine-tuned for legitimate purposes has limited commercial viability.
3. **Cross-domain generalisation in harm space**: NTL's auxiliary domain is either a known target or a GAN-generated neighbour. For open-weight LLM immunisation, the attack domain is semantically adversarial and unknown. The multi-direction/multi-distance augmentation strategy is a promising building block, but bridging from pixel-space distances to semantic harm space requires new formalisms.
4. **Resistance to RL-based attacks**: NTL was evaluated only against SFT-style attacks. As shown in the ET.pdf paper in this corpus, RL-based attacks (which reduce entropy and break alignment more efficiently) represent a fundamentally different threat model that NTL has not been tested against.
5. **Evaluation under the formal immunisation conditions**: NTL should be re-evaluated against the four-condition framework of Rosati et al. (2024), with explicit measurement of weak/strong resistance, stability on general benchmarks, cross-domain generalisation, and trainability. This remains entirely open.
