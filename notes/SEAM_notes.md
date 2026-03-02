# SEAM — Self-Destructive Language Model
**Wang, Zhu, Wang · Stony Brook University · 2025 (preprint)**

---

## 1. Quick Mechanistic Summary

SEAM's central idea is to make a model *complicit in its own destruction*. Rather than resisting harmful fine-tuning by raising its cost (the Block 1 strategy), SEAM engineers the parameter space so that **gradient descent on harmful data is geometrically equivalent to gradient ascent on benign data**. The model does not fight the attacker — it passively lets the attacker ruin it.

The mechanism works in gradient space. SEAM forces the adversarial gradient $g_a$ (computed on harmful data) and the benign gradient $g_b$ (computed on innocuous data) to point in opposite directions at the model's current weights. The consequence is immediate and elegant: any step the attacker takes along $-g_a$ is a step along $+g_b$, degrading benign performance. The harder the attacker pushes, the more the model's general language modelling ability collapses. The attacker wins the battle and inherits a broken shell.

Three loss terms collaborate to create and sustain this trap:

- **Self-destructive loss** $\mathcal{L}_\text{sd}$: aligns $g_a$ and $g_b$ into opposing directions via cosine similarity maximisation (i.e. cosine minimisation = push toward $-1$).
- **Unlearning loss** $\mathcal{L}_\text{ul}$: gradient ascent on harmful data, raising the starting harmful loss so the attacker must take many more steps — long enough for the self-destructive gradient coupling to inflict real damage.
- **Utility preservation loss** $\mathcal{L}_\text{up}$: SFT on harmful prompts paired with GPT-4o-generated refusal responses, keeping the model coherent and safe before any attack.

The implementation challenge is that optimising $\mathcal{L}_\text{sd}$ involves the Hessian, which is intractable at LLM scale. SEAM sidesteps this with a **Hessian-free finite-difference estimate**: perturb the parameters by a small $\epsilon$ in the direction perpendicular to each gradient, recompute gradients, and use the resulting gradient difference as a proxy for the Hessian-vector product.

---

## 2. Timeline Positioning

| Block | Role | Key antecedents |
|---|---|---|
| Block 1 — Weight-Space Resistance | Passive resistance; raise cost of harmful fine-tuning | MLAC, TAR, Vaccine, Booster, RepNoise, RMU |
| **Block 2 — Conditional Collapse** | **Active trap; harmful fine-tuning destroys general utility** | MLAC (conceptual ancestor), CTRAP (first LLM-scale collapse paper) |

SEAM is **paper 15 of 20** in the tutorial taxonomy, placing it in the first wave of Block 2 (Conditional Collapse) papers alongside CTRAP (paper 14) and SDD (paper 18).

### What SEAM inherits

- From **MLAC (Henderson et al., 2022)**: the name "self-destructing models" and the goal of making harmful fine-tuning structurally difficult — but SEAM completely changes the mechanism (MLAC raises cost; SEAM triggers collapse).
- From **Vaccine / T-Vaccine**: the use of an adversarial dataset $\mathcal{D}_\text{adv}$ simulating the attacker, computed at alignment time.
- From **Booster**: the idea of using a second-order or look-ahead gradient evaluation — Booster approximates a Hessian step with three gradient passes; SEAM uses four gradient passes in its finite-difference scheme.
- From **RepNoise**: the layer-wise gradient ascent strategy (SEAM applies layer-wise gradient ascent in $\mathcal{L}_\text{ul}$ to avoid catastrophic forgetting).
- From **TAR**: the meta-learning intuition that the defender should simulate what the attacker will do and prepare the parameter space accordingly.

### SEAM's unique contribution

SEAM is the first paper to make the **gradient directions** (not the loss values, not the parameter distances, not the representations) the primary object of immunisation. No prior method in the literature explicitly minimises the cosine similarity between harmful and benign gradients. This is not just a twist — it changes the failure mode: while all Block 1 methods become ineffective if the attacker is patient enough, SEAM's trap actually *worsens* with attack intensity. The gradient coupling makes high-intensity attacks more self-destructive than low-intensity ones.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 Setup

Let $f_\theta$ be the model, $\mathcal{D}_\text{adv}$ a harmful dataset, $\mathcal{D}_\text{bgn}$ a benign dataset (Alpaca), and $\mathcal{D}_\text{aln}$ an alignment dataset (harmful prompts paired with GPT-4o refusal responses). Define:

$$g_a(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_\text{adv}} \nabla_\theta \ell(f_\theta(x), y), \quad g_b(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_\text{bgn}} \nabla_\theta \ell(f_\theta(x), y)$$

### 3.2 The Three Losses

**Self-destructive loss** (the trap itself):
$$\mathcal{L}_\text{sd}(\theta) = \cos(g_a(\theta),\, g_b(\theta)) = \frac{\langle g_a, g_b \rangle}{\|g_a\|\|g_b\|}$$

Minimising this pushes cosine similarity toward $-1$. When $\cos(g_a, g_b) = -1$, gradient descent along $g_a$ is gradient ascent along $g_b$ — benign performance must degrade whenever the attacker fine-tunes.

**Unlearning loss** (amplifier — extends the number of steps before the attacker gains harmful ability):
$$\mathcal{L}_\text{ul}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}_\text{adv}} \ell(f_\theta(x), y)$$

Applied with layer-wise gradient ascent and a logarithmic transformation to prevent catastrophic forgetting.

**Utility preservation loss** (keeps the model coherent and safe pre-attack):
$$\mathcal{L}_\text{up}(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_\text{aln}} \ell(f_\theta(x), y)$$

Note the design choice: $\mathcal{D}_\text{aln}$ uses **harmful prompts** paired with refusal responses — not a generic benign corpus. This anchors the model's internal representation of harmful contexts to refusal behaviour and prevents $\mathcal{L}_\text{ul}$ from destroying the model's ability to recognise danger.

**Combined objective:**
$$\mathcal{L}(\theta) = \mathcal{L}_\text{ul}(\theta) + \alpha\,\mathcal{L}_\text{up}(\theta) + \beta\,\mathcal{L}_\text{sd}(\theta)$$

with defaults $\alpha=1$, $\beta=0.01$.

### 3.3 Why the Hessian Appears

To minimise $\mathcal{L}_\text{sd}$ by gradient descent, one needs $\nabla_\theta \mathcal{L}_\text{sd}$. Expanding the cosine similarity:

$$\nabla_\theta \mathcal{L}_\text{sd} = \frac{H_a\,\bar{g}_b - c\,H_a\,\bar{g}_a}{\|g_a\|} + \frac{H_b\,\bar{g}_a - c\,H_b\,\bar{g}_b}{\|g_b\|}$$

where $\bar{g}_a = g_a/\|g_a\|$, $\bar{g}_b = g_b/\|g_b\|$, $c = \bar{g}_a^\top \bar{g}_b$, and $H_a$, $H_b$ are the Hessians of $\mathcal{L}_a$ and $\mathcal{L}_b$ w.r.t. $\theta$. This simplifies to:

$$\nabla_\theta \mathcal{L}_\text{sd} = \frac{H_a\,\delta_a}{\|g_a\|} + \frac{H_b\,\delta_b}{\|g_b\|}$$

where $\delta_a = \bar{g}_b - c\bar{g}_a$ (component of $\bar{g}_b$ orthogonal to $\bar{g}_a$) and $\delta_b = \bar{g}_a - c\bar{g}_b$. For a 7B-parameter model, $H_a$ is a $7\text{B} \times 7\text{B}$ matrix — completely intractable.

### 3.4 The Hessian-Free Estimate (Eq. 6)

From the first-order Taylor expansion:

$$\nabla_\theta \mathcal{L}_a(\theta + \epsilon\,\delta_a) = \nabla_\theta \mathcal{L}_a(\theta) + \epsilon\,H_a\,\delta_a + \mathcal{O}(\|\epsilon\,\delta_a\|^2)$$

Rearranging: $H_a\,\delta_a \approx \frac{1}{\epsilon}\left(\nabla_\theta\mathcal{L}_a(\theta+\epsilon\,\delta_a) - \nabla_\theta\mathcal{L}_a(\theta)\right)$

Substituting both Hessian-vector products into the gradient formula gives the estimate:

$$\widehat{\nabla_\theta \mathcal{L}_\text{sd}(\theta)} = \frac{1}{\epsilon}\left(\frac{g_b(\theta + \epsilon(\bar{g}_a - c\bar{g}_b)) - g_b(\theta)}{\|g_b(\theta)\|} + \frac{g_a(\theta + \epsilon(\bar{g}_b - c\bar{g}_a)) - g_a(\theta)}{\|g_a(\theta)\|}\right)$$

The perturbation directions $(\bar{g}_a - c\bar{g}_b)$ and $(\bar{g}_b - c\bar{g}_a)$ are exactly the components of each normalised gradient **perpendicular to the other** — the directions in which cosine similarity would increase fastest. In practice this costs **four backward passes per step**: two gradient evaluations at $\theta$ (for $g_a$ and $g_b$) and two at perturbed checkpoints.

### 3.5 Error Bound (Theorem 1)

Expanding to second order and invoking local Hessian Lipschitz continuity with constants $L_a^H$, $L_b^H$:

$$\left\|\widehat{\nabla_\theta\mathcal{L}_\text{sd}} - \nabla_\theta\mathcal{L}_\text{sd}\right\| \leq \frac{\epsilon}{2}\left(\frac{L_a^H}{\|g_a(\theta)\|} + \frac{L_b^H}{\|g_b(\theta)\|}\right) + \mathcal{O}(\epsilon^2)$$

The error shrinks as $\epsilon \to 0$ (approximation improves) but numerical instability grows. The empirical sweet spot is $\epsilon = 10^{-3}$. Larger gradient norms $\|g_a\|$, $\|g_b\|$ also tighten the bound — an informative observation: the estimate is most reliable exactly when the model is far from collapse (gradients are large and well-defined).

---

## 4. Immunisation Properties — Assessment

| Property | Verified? | Evidence | Notes |
|---|---|---|---|
| **Resistance** | ✅ Strong | HS ≈ 0–5% across all 11 attack configurations (Table in §5.3) | No evaluated attack succeeds at producing informative harmful output |
| **Stability** | ✅ | Pre-attack ZS: 51.6 (base) → 50.8 (SEAM); FS essentially unchanged (Table 1) | Negligible pre-attack utility cost |
| **Generalisation** | ✅ (partial) | Trained on 7/14 BeaverTails categories, tested on the other 7 — SEAM remains effective (Table 3) | Cross-domain attack diversity limited; no RL-based attacks tested |
| **Trainability** | ✅ | Fine-tuning scores on SST2/AGNEWS/GSM8K/AlpacaEval match or exceed base model (Table 1) | Benign fine-tuning is *unaffected* — the trap does not fire on benign gradients |

### The Missing Piece: Resistance to RL-Based Attacks

All 11 attack configurations evaluated use SFT or LoRA fine-tuning with standard gradient descent (AdamW or SGD). SEAM's trap is geometrically defined on SFT gradients. Harmful Reinforcement Learning (RL) — which uses policy gradient updates, not supervised cross-entropy — follows a fundamentally different trajectory through parameter space. The gradient coupling between $g_a$ and $g_b$ as defined may not generalise to RL's exploratory update rule. TokenBuncher (paper 20), published shortly after, confirms this gap explicitly and is designed to complement SEAM specifically for RL attackers.

---

## 5. Mechanistic Commonalities with Other Approaches

### Finite-difference / Hessian-approximation gradient steps

This family of tricks appears across four papers in the tutorial, all trying to compute or approximate second-order information without forming the full Hessian:

| Paper | Trick | How second-order info is approximated |
|---|---|---|
| **Booster** | Look-ahead harmful perturbation | Finite difference in *parameter space* along the harmful gradient direction: evaluate harmful loss at $\theta - \alpha\,\nabla h/\|\nabla h\|$, subtract from current gradient |
| **SEAM** | Hessian-free gradient estimate | Finite difference in *gradient space* along the orthogonal components of $g_a$, $g_b$: evaluate gradient at $\theta + \epsilon\,\delta_a$, subtract |
| **CTRAP** | Collapse loss at perturbed weights | Evaluate collapse loss at $\theta - \alpha\nabla_\theta\ell_\text{harmful}$ — a single harmful step look-ahead — similar in spirit to Booster |
| **TAR** | Meta-learning inner loop | Unrolls $K$ gradient descent steps with SGD; the outer loop sees the effect of those steps without computing the full Hessian |

The common underlying intuition: you cannot afford the true Hessian, but you can afford to take a small step in a direction of interest and observe how the gradient changes. This is a directional finite difference of the gradient — a Hessian-vector product without the Hessian.

### Gradient ascent on harmful data

SEAM's $\mathcal{L}_\text{ul}$ (Eq. 3) is negative log-likelihood on harmful data — identical in form to the unlearning gradient ascent used in RMU, RepNoise (layer-wise), and LLM-Unlearning. The difference is purpose: in those papers, gradient ascent on harmful data *is* the defence. In SEAM, it is an *amplifier* — it extends the number of steps the attacker needs before harmful behaviour emerges, giving the gradient coupling more time to do damage.

### Adversarial dataset at alignment time

Vaccine, T-Vaccine, Booster, TAR, RepNoise, and SEAM all assume defender access to a representative $\mathcal{D}_\text{adv}$. This is a structural assumption of nearly every alignment-stage method. The distinguishing factor is *what you do with* $\mathcal{D}_\text{adv}$: Vaccine uses it to simulate embedding drift; Booster uses it to simulate a perturbation; SEAM uses it to compute $g_a$ and then orient it against $g_b$.

---

## 6. Results Summary and Significance

### Core result (Llama2-7B, Table 2 / Figure 3)

| Attack intensity ($\eta$) | Base model HS | SEAM HS | SEAM ZS | Base ZS |
|---|---|---|---|---|
| Pre-attack | 5.0% | 5.0% | 50.8% | 51.6% |
| 2e-5 (weak) | 47.3% | **2.6%** | 47.9% | 51.7% |
| 5e-5 | 77.5% | **3.1%** | 42.7% | 50.6% |
| 8e-5 | 80.4% | **5.5%** | 39.3% | 49.7% |
| 1e-4 | 78.8% | **0.2%** | 25.8% | 49.8% |
| 2e-4 (savage) | 79.5% | **0.0%** | 26.6% | 50.2% |

The key comparison: at $\eta = 2\text{e-4}$, the base model is harmfully capable (HS=79.5%) but utility-intact (ZS=50.2%). SEAM under the same attack has HS=0.0% *and* ZS=26.6% — the model collapses into near-random output. This asymmetry is SEAM's unique signature: the attacker's success at removing alignment simultaneously destroys the model they wanted to weaponise.

The GPT-4o-graded metric (HS-G) confirms the HS classifier is not a surface artefact: across all attack intensities, SEAM HS-G stays at 0–2% while the base model reaches 44–77% HS-G.

### Significance relative to other papers

- SEAM is the **first paper to achieve near-zero harmfulness under high-intensity attacks** while also demonstrating that the degradation is catastrophic (ZS < 30%). All Block 1 baselines (Vaccine, T-Vaccine, Booster, TAR, RepNoise, RMU) show HS rising sharply above 50% at $\eta \geq 8\text{e-5}$.
- Unlike Block 1 methods, SEAM's resistance *strengthens* with attack intensity rather than degrading.
- SEAM matches or slightly exceeds base model fine-tuning accuracy across SST2, AGNEWS, GSM8K, AlpacaEval — a result no prior paper achieves simultaneously with strong resistance under high-intensity attacks.
- Variance analysis over 20 random seeds confirms statistical stability (Table 4 in §C.1).
- Results replicate across Qwen2.5-3b/7b, Llama3-3b/8b — demonstrating model-agnostic applicability.

---

## 7. Calls for Future Work

### From the authors

1. **Optimal benign dataset selection.** SEAM uses Alpaca, but the benign dataset $\mathcal{D}_\text{bgn}$ is a free parameter. Future work should characterise which benign corpora maximise the self-destructive effect — a dataset that spans more of the benign gradient space would tighten the coupling.
2. **Adaptive attacks.** The evaluated attacker is non-adaptive — they use standard SFT without knowledge of SEAM's construction. A white-box adaptive attacker who knows SEAM could, in principle, design an attack objective that reduces $g_a$ norm while staying near $g_b$ direction, attempting to fine-tune "around" the gradient coupling. This is an open and important challenge.
3. **Scale validation.** Experiments cap at 8B parameters (Llama3-8B). Whether the gradient coupling remains effective in 70B+ models — where the parameter space is far higher-dimensional and gradient norms may behave differently — is unknown.

### From the state-of-the-art (beyond the paper)

4. **RL-based attack resistance.** The most pressing gap. TokenBuncher (paper 20) is SEAM's intended complement for harmful RL attacks. A unified defence handling both SFT and RL attackers does not yet exist.
5. **Formal resistance guarantees.** No Block 2 paper (including SEAM) offers a formal lower bound on the number of fine-tuning steps an attacker must take before either causing harm or triggering collapse. SDD makes the strongest theoretical contribution in this space; SEAM's theory is limited to the approximation error bound.
6. **Pre-alignment base model threat.** All Block 2 defences are applied *after* pre-training. An attacker with access to the raw base checkpoint — before alignment — can circumvent the entire SEAM construction by training directly from the unmodified weights.
7. **Interpretability of collapse.** The PCA gradient visualisation in §5.5 is compelling but informal. A mechanistic circuit-level analysis of *why* gradient opposition causes coherence collapse (rather than simply making the model more aligned) would deepen theoretical understanding and potentially suggest tighter trap designs.
8. **Combination with Block 1 methods.** SEAM resists high-intensity attacks but low-intensity attacks produce a slight HS uptick before self-destruction kicks in. Combining SEAM's collapse trap with a Booster-style perturbation attenuation that handles low-intensity attacks cleanly could close this window.
