# Circuit Breakers — Paper Notes
**Improving Alignment and Robustness with Circuit Breakers**  
Zou et al. (2024) — Gray Swan AI / CMU / Center for AI Safety

---

## 1. Mechanistic Summary

The central premise is simple and sharp: rather than training the model to *refuse* harmful requests (which creates a refusal region in representation space that can be bypassed), *reroute* the internal representations that give rise to harmful outputs so they can never complete a harmful generation in the first place.

![](figs/circuit_brakers/_page_1_Figure_0.jpeg)

The mechanism is called **Representation Rerouting (RR)** and is implemented via **LoRRA** — Low-Rank Representation Adaptation. The key insight, borrowed from representation engineering (RepE), is that the sequence of internal activations a model traverses while generating "Here is how to synthesise meth: step 1…" is structurally distinct from the sequence it traverses when generating a refusal. Harmful generation is a *circuit* through activation space. The goal is to short-circuit it: when the model starts heading down that representational path, its own LoRA-modified layers reroute the activations to an orthogonal, incoherent region, causing generation to degenerate (EOS tokens, incoherent output).

This is done **once**, at alignment time, before model release. The LoRA adapters are absorbed or remain attached and the resulting model is the one deployed. No guard model, no inference filter, no additional latency.

---

## 2. Timeline Positioning

Circuit Breakers sits at a critical **pivot point** in the immunisation literature.

**What it inherits:**
- **Representation Engineering / RepE (Zou et al., 2023)** — the direct intellectual parent. RepE showed that internal representations encode high-level properties (honesty, safety, emotion) along linear directions that can be read and written. Circuit Breakers is RepE applied to immunisation.
- **RMU (random model unlearning)** — an immediate predecessor within the same lineage. RMU reroutes harmful representations to a fixed random vector with large norm. Circuit Breakers generalises and critiques this: the random-vector approach is fragile and requires heavy hyperparameter tuning.
- **Vaccine / T-Vaccine (Huang et al., 2024)** — the concurrent alignment-stage adversarial perturbation approach. Both work in representation space; the difference is that Vaccine adds perturbation *during alignment training* to simulate harmful fine-tuning, while Circuit Breakers *modifies the representation map itself*, independent of fine-tuning dynamics.
- **RepNoise (Rosati et al., 2024)** — also adds noise to harmful representations but does so as a noising regulariser, not as a learnable rerouting. Circuit Breakers is the learnable evolution of this idea.

**What makes it unique:**
1. **Attack-agnosticism by design.** Because it targets the *output representation process* rather than any specific input attack, it generalises to attacks never seen during training (GCG, AutoDAN, prefilling, input embedding, RepE attacks — all unseen — are substantially blocked).
2. **Multi-modal and agent generality.** The first paper in this family to demonstrate immunisation-equivalent results in vision-language models and function-calling agents.
3. **The cosine orthogonality loss.** Prior approaches (RMU) used L2 distance to a random target. Circuit Breakers argues that the target does not need to be a random fixed vector — it just needs to be *orthogonal* to the original harmful representation. This is more principled and more stable.

**Timeline anchor:** 2024, concurrent with Booster, Antidote, VAA. Predates the self-destructive SEAM family. It is a Block 1 method (cost-raising), but distinctively a *representation-space* member rather than a weight-space meta-learning one.

---

## 3. The Math

### 3.1 Setup

Let $\mathcal{M}$ be the frozen original (refusal-trained) model. Let $\mathcal{M}_{cb}$ be the same model augmented with trainable LoRA adapters — these will become the circuit breakers. Let $\text{rep}_{\mathcal{M}}(x)$ denote the residual stream activations (at selected layers) of model $\mathcal{M}$ on input $x$.

Two datasets are required:
- **Circuit Breaker Set** $\mathcal{D}_s$: inputs that elicit harmful completions. Crucially, for models with pre-existing refusal mechanisms, these are constructed by removing the harmful user request and keeping only the harmful assistant response — so the model actually generates harmful activations rather than refusals.
- **Retain Set** $\mathcal{D}_r$: benign instructional conversations (UltraChat, XSTest). Ensures circuit breakers do not fire on legitimate requests.

### 3.2 The Two Losses

**Representation Rerouting (RR) Loss — the circuit breaker signal:**

$$\mathcal{L}_s = \text{ReLU}\!\left(\text{cosine\_sim}\!\left(\text{rep}_{\mathcal{M}}(x_s),\; \text{rep}_{\mathcal{M}_{cb}}(x_s)\right)\right)$$

This minimises the cosine similarity between the original harmful representation and the circuit-broken representation. The ReLU ensures we only penalise *positive* cosine similarity — once the representations are orthogonal or anti-correlated, the gradient vanishes and training does not push further. This is the key mechanistic novelty: we are not mapping representations to some arbitrary target; we are simply requiring that the circuit-broken representation carries **no information in the direction of the original harmful representation**.

Compared to alternatives:
- **RMU loss**: $\|\text{rep}_{cb} - \alpha \cdot \text{rep}_{\text{rand}}\|_2$ — requires tuning large constant $\alpha$, and a random target may coincidentally correlate with benign representations.
- **Normalised random vector**: $\|\text{rep}_{cb}/\|\text{rep}_{cb}\| - \text{rep}_{\text{rand}}/\|\text{rep}_{\text{rand}}\|\|_2$ — avoids the $\alpha$ issue but is still arbitrary.
- **Cosine (RR)**: geometry-aware, scale-invariant, gradient-vanishing at orthogonality. Ablations confirm this is the most stable and effective variant.

**Retain Loss — stability signal:**

$$\mathcal{L}_r = \left\|\text{rep}_{\mathcal{M}}(x_r) - \text{rep}_{\mathcal{M}_{cb}}(x_r)\right\|_2$$

This is a straightforward L2 constraint: on benign inputs, $\mathcal{M}_{cb}$ must produce the same representations as $\mathcal{M}$. This preserves pre-existing capabilities and prevents over-refusal.

### 3.3 Training Objective and Schedule

$$\mathcal{L} = c_s \cdot \mathcal{L}_s + c_r \cdot \mathcal{L}_r$$

with a **coefficient schedule** that shifts emphasis over the $T$ training steps:

$$c_s = \alpha\!\left(1 - \frac{t}{2T}\right), \qquad c_r = \alpha\,\frac{t}{2T}$$

Early in training ($t \approx 0$): $c_s \approx \alpha$, $c_r \approx 0$ — the optimiser focuses on breaking the harmful circuit. Late in training ($t \approx T$): $c_s \approx \alpha/2$, $c_r \approx \alpha/2$ — both terms contribute equally, consolidating stability. This prevents the well-known failure mode of aggressive early retention killing the circuit-breaking signal.

The LoRA adapters in $\mathcal{M}_{cb}$ are the only parameters updated. The frozen model $\mathcal{M}$ serves as a stable reference for both loss terms.

### 3.4 Full Algorithm (LoRRA + RR)

```
Require: frozen M, LoRA-augmented M_cb, rep(), D_s, D_r, steps T, hyperparameter α

for t = 1, ..., T:
    x_s ~ D_s,  x_r ~ D_r
    c_s = α(1 − t/2T),  c_r = α(t/2T)
    
    L_s = ReLU(cosine_sim(rep_M(x_s), rep_{M_cb}(x_s)))   ← RR loss
    L_r = ‖rep_M(x_r) − rep_{M_cb}(x_r)‖₂               ← Retain loss
    
    L = c_s · L_s + c_r · L_r
    update LoRA params of M_cb via gradient descent on L
```

### 3.5 What Happens at Inference

Once trained, the LoRA adapters are frozen in $\mathcal{M}_{cb}$. At inference, when a harmful sequence begins to unfold in the residual stream, the modified transformer layers produce activations that are orthogonal to the harmful direction. The model cannot continue the harmful sequence because the representational substrate has been redirected. Empirically, this manifests as degenerate output (EOS tokens, repetition) or incoherence — the circuit breaker has fired.

![](figs/circuit_brakers/_page_4_Figure_0.jpeg)

The cosine analysis in Figure 6 shows this happening *before generation completes* — the circuit breaks as early as the prefilling stage at layer 10, before a single output token is generated. This is mechanistically important: the defence is not reactive, it is structural.

---

## 4. Immunisation Properties Assessment

| Property | Alignment | Assessment |
|---|---|---|
| **Resistance** | ✅ Strong | Demonstrated against 11 diverse unseen attacks. Cosine orthogonality loss is geometry-aware and attack-agnostic by design. |
| **Generalisation** | ✅ Strong | Category ablations (Figure 5) show solid cross-category transfer. Broad harm categories (Harmful, Illegal) generalise well; narrow ones (Chem/Bio) less so. |
| **Stability** | ⚠️ Partial | MT-Bench and Open LLM preserved. However, over-refusal (WildChat: ~6% false positive rate) is a known cost. The retain loss handles it, but imperfectly. |
| **Trainability** | ❌ Missing | This is the critical gap. Circuit Breakers does not address *fine-tuning attacks*. The paper explicitly scopes to adversarial inference-time attacks. If an attacker fine-tunes $\mathcal{M}_{cb}$ on harmful data, the LoRA adapters implementing the circuit breakers are themselves trainable and could be overwritten or dominated by new LoRA adapters. The paper does not test or claim fine-tuning resistance. |

**The honest bottom line on trainability**: Circuit Breakers is not trying to be immunisation against fine-tuning in the strict sense used by Vaccine, RepNoise, TAR, etc. It is immunisation against a different attack surface — adversarial prompting. This is an orthogonal and genuinely important problem, but it means Circuit Breakers, alone, does not satisfy the full immunisation definition: it is breakable by a determined fine-tuner.

---

## 5. Mechanistic Commonalities with Other Approaches

The field has converged on a shared structural pattern — a **dual-objective loss** with a harmful-disruption term and a benign-retention term. Circuit Breakers fits this template with its own twist on each:

**The harmful-disruption term across methods:**
- *Vaccine*: adds adversarial perturbation $\delta$ in embedding space to simulate harmful drift; the loss minimises this drift. First-order gradient attack in the inner loop.
- *RepNoise*: adds noise to harmful representations, pushes them toward a random Gaussian vector (L2 loss to a fixed random target per batch).
- *Booster*: simulates harmful fine-tuning perturbation in weight space, penalises the loss reduction that the perturbation would achieve.
- *RMU*: L2 distance to a large-norm random constant vector in activation space.
- **Circuit Breakers (RR)**: ReLU cosine similarity to the *original* representation, making the target implicit (orthogonal complement) rather than explicit (random vector). This is more principled: it requires only *decoupling* from the harmful direction, not convergence to any particular target.

**The benign-retention term across methods:**
Almost universally L2 distance in representation space between the modified model and the frozen original. This is a near-universal convention.

**Gradient / Hessian approximations:** Unlike TAR, SDM, SOPHON, or Booster, Circuit Breakers makes **no second-order approximation**. The LoRRA algorithm is purely first-order gradient descent. This makes it computationally cheaper but also less theoretically grounded in terms of the curvature of the loss landscape — it does not explicitly reason about what happens when an attacker fine-tunes the model.

**Connection to weight-space methods**: Methods like TAR and SDM operate in parameter space (bi-level meta-learning, Hessian approximations). Circuit Breakers operates purely in activation/representation space. The two can in principle be composed.

---

## 6. Results Summary and Significance

**LLM results (Table 1):**

| Method | Avg ASR ↓ | MT-Bench ↑ | Open LLM ↑ |
|---|---|---|---|
| Mistral — Refusal Trained | 76.7% | 7.60 | 65.4 |
| Mistral — Adv. Trained (R2D2) | 31.7% | 6.00 | 61.2 |
| Mistral — **+ RR (Ours)** | **9.8%** | **7.53** | **65.4** |
| Llama-3 — Refusal Trained | 38.1% | 8.05 | 68.8 |
| Llama-3 — **+ RR (Ours)** | **3.8%** | **8.00** | **68.3** |
| **Cygnet (+ RR + control)** | **0.8%** | **8.21** | **71.9** |

The result that stands out in the broader immunisation literature: **Adversarial training (R2D2) drops MT-Bench by 1.6 points and Open LLM by 4.2 points to achieve 31.7% ASR.** Circuit Breakers achieves 9.8% ASR with *no meaningful capability loss*. This is the clearest empirical refutation of the "robustness-capability tradeoff is unavoidable" assumption that had been dominant since Tsipras et al. (2019) [64].

Cygnet — the flagship Llama-3 finetune integrating RR plus additional control methods — achieves near-zero ASR on all attacks *including hard white-box attacks*, while *exceeding* the original model's MT-Bench score. This is, to the authors' knowledge, the first time a defence has moved a model along both axes simultaneously in a convincing way.

**Multimodal results (Figure 3):** Under PGD attack, compliance drops by 84% compared to original and 85% compared to a safety-prompted baseline, with MMMU/LLaVA-Wild preserved within 0.5%. This is notable because standalone image classifiers have never achieved PGD resistance without steep accuracy penalties.

**Agent results (Figure 4):** 84% and 83% reductions in harmful function-call compliance under Direct Request and Forced Function Call attacks respectively, with Berkeley Function Calling Leaderboard scores preserved.

**Significance relative to peers:** Among representation-space methods, Circuit Breakers establishes the state of the art on inference-time attack robustness. Vaccine and RepNoise are stronger on fine-tuning attack resistance (the attack they were designed for), but they were not tested on the full suite of adversarial prompt attacks that Circuit Breakers handles. The two families are targeting orthogonal attack surfaces.

---

## 7. Future Work and Open Criticisms

### From the Authors

- **Semantically meaningful rerouting targets.** The paper acknowledges routing to the EOS token direction or an explicit refusal direction as a natural extension. Current orthogonality is geometrically sound but semantically neutral — the model just breaks; it does not refuse gracefully.
- **Combining HP + RR.** The harmfulness probe experiments show that representation *reading* (monitoring, not rewriting) is a complementary tool. The authors explicitly leave this combination for future work.
- **Coverage gaps.** The Chem/Bio category shows weaker cross-category generalisation (Figure 5 diagonal). Specialised circuit breaker datasets per harm domain could improve this.
- **Single-turn focus.** The paper is explicit that circuit breaking is evaluated in single-turn conversations. Multi-turn jailbreaks — where the harmful state builds across conversational context — are not addressed.

### From the State of the Art and Criticism Papers

**Critical: fine-tuning breaks it.** The "On Evaluating the Durability of Safeguards for Open-Weight LLMs" (Qi et al., 2025) systematically shows that even strong inference-time defences are undermined when the adversary can fine-tune the model. Circuit Breakers' LoRA adapters implementing the rerouting are themselves modifiable parameters. A sufficiently motivated attacker with GPU access can simply fine-tune them away. The paper does not claim fine-tuning resistance, but the immunisation framing (one-shot pre-release defence against *any* intervention) demands it.

**Structural fragility.** "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications" (Wei et al., 2024) shows that safety-critical regions in aligned models are sparse (~3% of parameters, ~2.5% of ranks). Circuit Breakers' safety contribution is itself concentrated in a LoRA adapter — by definition low-rank. An attacker knowing this can target those specific ranks for modification. Freezing safety-critical parameters does not prevent circumvention; the attack creates new pathways around the frozen region.

**The attack-agnosticism claim needs qualification.** Attack-agnosticism is true *within the inference-time adversarial prompt threat model*. It is not true across threat models. The "Other — Your Task May Vary" paper and broader literature show that combining techniques is necessary because no single method covers the full attack surface.

**Trainability is the missing dimension.** From the taxonomy: Circuit Breakers strongly satisfies **resistance** and **generalisation** over inference-time attacks, partially satisfies **stability**, and does not address **trainability**. For the full immunisation promise — an open-weight model robust to *unconstrained* intervention including fine-tuning — Circuit Breakers must be composed with a weight-space method (e.g., TAR, Vaccine, RepNoise). The circuit breaker protects the inference-time surface; another method must protect the parameter surface.

**Open question:** Can the cosine orthogonality loss be reframed as a weight-space constraint, or extended to the fine-tuning threat model without second-order approximations? This would unify the representation-space and parameter-space approaches and would be the natural next step from Circuit Breakers' framework.
