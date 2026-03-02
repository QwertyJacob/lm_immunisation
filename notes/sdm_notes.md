# Paper Notes: Self-Destructing Models (SDM / MLAC)

**Full title:** *Self-Destructing Models: Increasing the Costs of Harmful Dual Uses in Foundation Models*  
**Authors:** Eric Mitchell, Peter Henderson, Christopher D. Manning, Dan Jurafsky, Chelsea Finn (Stanford, 2022)  
**Venue:** First Workshop on Pre-training: Perspectives, Pitfalls, and Paths Forward @ ICML 2022

---

## 1. Mechanistic Summary

SDM introduces the **task blocking** paradigm: a training-time procedure that bakes resistance to harmful fine-tuning *into* the model's weights before release. The model is not aligned in the traditional sense — it produces correct outputs on desired tasks. It is, however, engineered to be a *terrible initialisation* for any harmful task the designer wishes to block.

The core mechanism is **Meta-Learned Adversarial Censoring (MLAC)**, a bi-level optimisation where:

- An **inner loop** simulates an adversary fine-tuning the model on a harmful task for $K$ steps.
- An **outer loop** maximises the adversary's loss at the end of those $K$ steps, while simultaneously minimising the loss on a desired (benign) task.

The model therefore learns to have weights from which harmful learning is maximally costly — a bad initialisation for the attacker — while still being a good initialisation for legitimate adaptation. The feature extractor $\pi_{\tilde{\theta}}$ is the object being manipulated; it is shared between a fixed desired-task head $w_d$ and an adversarially learned harmful-task head $w_h$.

Crucially, the adversarial head $w_h$ and its learning rate $\alpha_h$ are themselves learned parameters ($\phi$), updated alongside $\tilde{\theta}$ in the outer loop. This ensures the inner-loop adversary is always playing close-to-optimal, preventing the outer loop from finding degenerate solutions that only fool a weak simulated attacker.

---

## 2. Timeline Positioning

```
2022 ── SDM/MLAC ──────────────────────────── 2024 ──── 2025
          │                                     │
          │  [coins vocabulary, first proof]    │
          │                                     ├── TAR (entropy outer loop, LLM scale)
          │                                     ├── Vaccine / T-Vaccine / RepNoise / Booster
          │                                     ├── Condition Number paper
          │                                     ├── Immunisation Definition paper (Rosati et al.)
          │                                     └── LoX (training-free, SVD extrapolation)
          │
          └── NTL / SOPHON (domain anti-adaptation, MAML)
```

### What SDM inherits

- **MAML (Finn et al., 2017):** SDM's bi-level loop structure is a direct descendant of Model-Agnostic Meta-Learning. Where MAML encodes "fast learning" into an initialisation, MLAC encodes "slow harmful learning" into one.
- **Adversarial de-biasing (Edwards & Storkey, 2015):** The use of an adversarial head to suppress information from representations is structurally related. However, SDM's key departure is that it optimises for poor performance *after fine-tuning*, not poor performance *in the original model* — a conceptually important leap.
- **Domain invariance methods (Ganin & Lempitsky, 2015):** gradient reversal for removing information from representations. SDM generalises this to the post-adaptation setting.

### What SDM uniquely contributes

1. **The task blocking framing itself**: The idea that a model can be made to actively resist harmful fine-tuning as a design property, distinct from access control, RLHF, or model editing.
2. **The vocabulary**: "self-destructing models", "task blocking", "bi-level immunisation" — all coined here. Every subsequent paper in this space either cites MLAC or independently rediscovers its logic.
3. **The adversarially learned inner-loop:** $w_h$ and $\alpha_h$ are not fixed; they are adapted jointly with $\tilde{\theta}$, making the simulated adversary close-to-optimal throughout training. This is more principled than using a frozen head with a hand-tuned learning rate.
4. **The cost metric framing**: Defining success as increasing the adversary's data cost (sample efficiency regret), rather than a binary pass/fail on a benchmark.

---

## 3. The Math — MLAC in Full Mechanistic Detail

### Notation

| Symbol | Meaning |
|---|---|
| $m = w_d \circ \pi_\theta$ | Full model: feature extractor $\pi_\theta$ composed with desired-task head $w_d$ |
| $\tilde{\theta}$ | "Blocked" copy of feature extractor parameters being trained |
| $w_h, \alpha_h$ | Adversarial harmful-task head and its learning rate; collected as $\phi = \{w_h, \alpha_h\}$ |
| $\mathcal{D}_d, \mathcal{D}_h$ | Desired-task dataset, harmful-task dataset |
| $\mathcal{L}_d, \mathcal{L}_h$ | NLL losses on desired and harmful tasks |
| $\tilde{f}_k \sim \tilde{\mathcal{F}}$ | Sampled fine-tuning procedure (e.g., Adam for $k$ steps at learning rate $\alpha_h$) |
| $K$ | Number of inner-loop adversarial steps |
| $\eta, \eta_h, \eta_d$ | Outer loop learning rates for $\tilde{\theta}$, $\phi$, $w_d$ respectively |

### The Inner Loop: Simulating the Adversary

For each meta-training step, a fine-tuning procedure $\tilde{f}_k$ is drawn from the proxy adversary class $\tilde{\mathcal{F}}$ and applied to the *harmful* task. Starting from the current blocked model $\tilde{\theta}$, we simulate $K$ gradient steps:

$$\{\theta^k\}_{k=0}^{K},\; \{w_h^k\}_{k=0}^{K} \;\leftarrow\; \tilde{f}_K\!\left(w_h \circ \pi_{\tilde{\theta}};\; \{b_h^k\};\; \alpha_h\right)$$

where each $b_h^k \sim \mathcal{D}_h$ is a fresh harmful data batch. At each step $k$, the loss on a held-out harmful batch $b_h$ is recorded:

$$\ell_h^k = \mathcal{L}_h\!\left(w_h^k \circ \pi_{\theta^k};\; b_h\right), \quad \forall k = 1, \ldots, K$$

This gives us a trajectory of $K$ harmful-loss values.

### Adversary Calibration

A pathological inner-loop solution is for the model to flip all class labels under the harmful task (e.g., in a binary classifier, predicting "male" for "female" and vice versa), yielding high NLL while still leaking information. To prevent this, MLAC adds an **optimal calibration step** at each inner-loop stage.

A linear projection $w_c^k$ is found by solving the following constrained MLE problem via differentiable convex optimisation:

$$w_c^k = \arg\max_W \sum_{i=1}^{|b_h|} \left[\log\text{softmax}\!\left(W \circ m^k\right)^\top y_i \right] \quad \text{s.t.} \quad -1 \le W \le 1$$

where $m^k = w_h^k \circ \pi_{\theta^k}$ is the partially-fine-tuned blocked model after $k$ inner-loop steps. The calibrated loss used in the outer loop then becomes:

$$\ell_h^k = \mathcal{L}_h\!\left(w_c^k \circ w_h^k \circ \pi_{\theta^k};\; b_h\right)$$

This ensures the outer loop is measuring *true inability to learn*, not a degenerate label-flipping outcome.

### The Outer Loop: Three Simultaneous Updates

Using the $K$ inner-loop harmful losses and a fresh desired-task batch $b_d \sim \mathcal{D}_d$:

**1. Update the blocked feature extractor $\tilde{\theta}$** — simultaneously maximise harmful performance (make it harder to learn) and minimise desired-task loss (preserve utility):

$$\tilde{\theta} \;\leftarrow\; \tilde{\theta} - \eta \, \nabla_{\tilde{\theta}} \left[\mathcal{L}_d\!\left(w_d \circ \pi_{\tilde{\theta}};\; b_d\right) - \frac{1}{K}\sum_{k=1}^{K} \ell_h^k \right]$$

The gradient $\nabla_{\tilde{\theta}} \ell_h^k$ passes **through** the inner loop, requiring second-order gradients (implemented via the `higher` library). This is what makes the outer loop genuinely aware of how the inner loop gradient landscape behaves, rather than just reacting to the final outcome.

**2. Update the adversarial head and learning rate $\phi = \{w_h, \alpha_h\}$** — improve the inner-loop adversary's quality:

$$\phi \;\leftarrow\; \phi - \eta_h \, \frac{1}{K} \sum_{k=1}^{K} \nabla_\phi \, \ell_h^k$$

This makes $\phi$ a moving target that the blocked model cannot "outmanoeuvre" by exploiting a fixed, suboptimal adversary.

**3. Update the desired-task head $w_d$** — standard supervised learning:

$$w_d \;\leftarrow\; w_d - \eta_d \, \nabla_{w_d} \mathcal{L}_d\!\left(w_d \circ \pi_{\tilde{\theta}};\; b_d\right)$$

### The Cost Metric

The paper defines the model's benefit to an adversary via **sample efficiency regret**. For a model $M$ with parameters $\theta$, and a re-initialised random baseline $\theta^r$, both adapted using procedure $f$ on a dataset $D^n$ of size $n$:

$$E^n_{\text{data}} = \mathbb{E}\left[\mathcal{L}_h\!\left(f(\theta^r; D^n)\right) - \mathcal{L}_h\!\left(f(\theta; D^n)\right)\right]$$

An ideal self-destructing model has $E^n_{\text{data}} \le 0$: the FM offers no sample-efficiency advantage over random initialisation for the harmful task. The adversary gains nothing by using the model.

---

## 4. Alignment with the Four Immunisation Properties

The four properties are defined formally in Rosati et al. (2024):

| Property | What SDM does | Assessment |
|---|---|---|
| **Resistance** (blocking harmful fine-tuning for a bounded budget) | Demonstrated empirically: adversaries with up to 1000 fine-tuning steps cannot recover harmful performance above the random baseline. | ✅ **Primary strength** — the paper is designed around this. However, only *weak* resistance is shown empirically; *strong* resistance (theoretical guarantees on attacker budget) is not established. |
| **Stability** (preserving benign performance at $t=0$, before any attack) | The desired-task NLL $\mathcal{L}_d$ is explicitly included in the outer-loop objective as a counterbalancing term. Zero-shot and few-shot profession classification performance is retained. | ✅ **Demonstrated** — though only on a narrow benchmark (profession classification vs. gender identification in "Bias in Bios"). |
| **Generalisation** (resistance generalises to new harmful data, not just training-time samples) | Not systematically tested. Only one harmful task, one dataset. No cross-domain evaluation. | ❌ **Missing piece** — the paper acknowledges this directly. The blocked model was never shown to resist harmful tasks from domains different from those used during MLAC training. |
| **Trainability** (benign fine-tuning efficiency is preserved) | Not tested at all. The desired-task head is updated during MLAC, but the paper does not test whether a downstream user can efficiently *fine-tune* the released model on new harmless data. | ❌ **Missing piece** — the immunisation definition paper later identifies this as a critical gap for SDM. |

**The core gap:** SDM tests a model before and after *harmful* fine-tuning. It does not ask whether the blocked model remains a good initialisation for *harmless* downstream tasks beyond the one used in MLAC training. The resistance mechanism could, in principle, harm the loss landscape for benign adaptation too — this is never measured.

---

## 5. Mechanistic Commonalities with Other Approaches

The bi-level optimisation structure in MLAC is the founding template that most subsequent weight-space methods inherit or react to.

**Shared gradient-level structure:**

| Method | Outer objective | Inner loop |
|---|---|---|
| **MLAC (SDM, 2022)** | $\max_\theta \mathcal{L}_h(\theta'(\theta)) - \mathcal{L}_d(\theta)$ | $K$ steps of gradient descent on $\mathcal{L}_h$, with learned head + LR |
| **TAR (2024)** | $\min_\theta \mathbb{E}_{\text{attack}}[\mathcal{L}_\text{TR}(\text{attack}(\theta))] + \lambda \mathcal{L}_\text{retain}$ where $\mathcal{L}_\text{TR}$ = **negative entropy** | Same structure; first-order MAML approximation for LLM scale |
| **SOPHON (2024)** | Explicit non-fine-tunability objective via MAML | MAML meta-gradient on harmful domain |
| **Booster (2024)** | Minimise harmful loss *reduction* after perturbation (gradient attenuation) | Simulated harmful perturbation as a single gradient step |
| **Vaccine (2024)** | Preserve alignment under simulated embedding drift | Uniform perturbation on all embedding layers as inner loop proxy |

**Key architectural differences from SDM:**

- **TAR** identifies that MLAC's use of cross-entropy in the outer loop creates a degenerate solution: the model can produce *confidently wrong* outputs, which gives high cross-entropy to the adversary in step 1 but is easily exploited in later steps. TAR fixes this by replacing cross-entropy with **entropy maximisation** — a model producing truly random outputs is resistant at every step of the attacker's trajectory.

- **Vaccine / T-Vaccine / Booster** replace the full inner-loop simulation with a lighter-weight **perturbation proxy** (adding noise to embeddings or computing a single gradient step) — much cheaper than $K$-step second-order differentiation, enabling scaling to LLM-sized models.

- **Condition Number paper** re-reads the bi-level structure geometrically: "making the loss landscape flat" is equivalent to making the Hessian's **condition number** large (i.e., an ill-conditioned landscape with no clear gradient direction). SDM achieves this implicitly; the Condition Number paper makes it an explicit training objective.

- **LoX** abandons training entirely: it identifies the safety-critical low-rank subspace post-alignment via SVD of $\Delta W = \theta_{\text{align}} - \theta_{\text{base}}$ and extrapolates further along it, moving the model into a flatter, more robust region without any inner loop.

---

## 6. Results and Significance

**Experimental setup:** BERT-tiny (small classifier) on the "Bias in Bios" dataset. Desired task: profession classification. Blocked task: gender identification. Harmful dataset sizes: 20, 50, 100, 200 samples. Adversary: up to 1000 fine-tuning steps.

**Headline results:**

- **Without MLAC (vanilla BERT-tiny):** fine-tuning on 20 samples reaches ~0.75 gender identification accuracy. Performance scales with data size.
- **With MLAC:** post-fine-tuning gender accuracy remains at ~0.55–0.58 regardless of dataset size — indistinguishable from a randomly initialised model.
- **Adversarial Censoring (AC, Edwards & Storkey baseline):** does *not* prevent effective fine-tuning; the adversary recovers harmful performance with moderate data.
- **Inner loop depth:** as few as $K=16$ inner-loop steps during MLAC training are sufficient to block an adversary using up to $K=1000$ steps. The ratio is striking and practically significant.
- **Desired task:** profession classification performance is retained at near-BERT-level zero-shot accuracy.

**Significance relative to the broader literature:**

SDM is a proof of concept, not a deployment-ready solution. Its significance is categorical rather than metric: it *demonstrates that the paradigm is viable* on a classifier at small scale. All quantitative comparisons in the literature that matter (TAR vs. Vaccine vs. RepNoise on Llama-3, etc.) come later. SDM's contribution is to establish:

1. That the inner-loop simulation depth ($K$) during training does not need to match the adversary's actual budget — a 16-step inner loop blocks a 1000-step adversary. This is the key non-obvious empirical finding.
2. That adversarial censoring on the original model (the natural baseline) is insufficient — fine-tuning recovers harmful capability even when the original model hides it.

---

## 7. Calls for Future Work

### From the authors (MLAC paper itself)

1. **Scale to larger FMs**: MLAC was only tested on BERT-tiny. The second-order gradient computation required for the inner-loop pass is computationally expensive; whether it can be approximated cheaply enough for billion-parameter models was left open.
2. **Generalisation testing**: Study whether blocking learned on one harmful dataset transfers to related but unseen harmful datasets (in-domain) and to entirely different harm domains (cross-domain).
3. **Stronger adversary classes**: The paper used only Adam fine-tuning in $\tilde{\mathcal{F}}$. Real adversaries might use prefix tuning, adapter layers, LoRA, or other PEFT methods with different gradient dynamics.
4. **Trainability testing**: The paper tests desired-task *zero-shot performance*, not desired-task *fine-tunability*. Future work must verify that the blocked model remains a good initialisation for harmless downstream fine-tuning.
5. **Concealed architectural triggers**: Hidden self-destruct mechanisms not visible to the adversary as a structural modification — a speculative direction toward cryptographic-style guarantees.

### From the state of the art (2024–2025 perspective)

6. **The inner-loop generalisation gap (TAR, 2024):** MLAC's outer loop uses cross-entropy, which can be gamed. TAR shows that entropy maximisation is substantially more robust. Future work on MLAC variants should adopt this correction.

7. **The durability problem (Qi et al., 2024 — "On Evaluating the Durability of Safeguards for Open-Weight LLMs"):** This paper shows that many safeguards, including those using MLAC-style logic, fail when the attacker has access to a *larger training budget than simulated* during the inner loop, or uses *out-of-distribution hyperparameters* (e.g., very low or very high learning rates, or RL-based fine-tuning). The fix is not obvious: the defender cannot simulate all possible adversaries. SDM's proxy class $\tilde{\mathcal{F}}$ is necessarily a strict subset of the true adversary class $\mathcal{F}$.

8. **Brittleness of safety-critical regions (Wei et al., 2024):** Safety mechanisms in aligned models occupy a remarkably sparse subspace (~3% of parameters at neuron level, ~2.5% at rank level). MLAC works by restructuring the loss landscape globally, but if harmful capability is encoded in similarly sparse, disentangled regions, a sufficiently fine-grained attacker can bypass the task-blocking mechanism by targeting only those regions.

9. **Trainability in practice (Immunisation Definition, Rosati et al., 2024):** Directly identifies SDM's adversarial loss method as failing to preserve trainability in their demonstration. This is a concrete, formal critique: an immunised model that cannot be fine-tuned on harmless datasets is commercially unusable and will face social pressure to have its defences undone.

10. **RL-based attackers (structural gap):** SDM pre-dates the discovery that RL-based harmful fine-tuning surpasses SFT's Pareto frontier (breaking alignment more effectively while preserving reasoning quality). MLAC was designed against gradient-descent SFT adversaries; its resistance against RLHF-reversal attackers is untested and theoretically unclear.
