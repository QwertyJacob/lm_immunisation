## Part 1.4 — Defining Language Model Immunisation

> *Parts 1.1–1.3 built the empirical case and the motivating intuition. Part 1.4 provides the mathematical vocabulary. We introduce the formal definition of immunisation — primarily following Rosati et al. (2024) — and then enrich it with a complementary geometric perspective from the condition number literature. We close with a set of careful preliminary considerations, drawn from two critical papers, that every reader should carry into the deeper technical sections of this tutorial.*

---

### 1.4.1 Why Formalism Is Necessary Here

Before the Rosati et al. (2024) paper, work on defending aligned models against harmful fine-tuning proceeded without shared definitions. Each paper chose its own evaluation protocol, its own attack benchmark, and its own notion of "success." The result was a literature where results were effectively incomparable: a method that looked impressive on one paper's evaluation might have looked mediocre on another's, not because of a real performance difference but because the questions being asked were different.

Rosati et al. make a pointed observation: without formal conditions, future papers could present defences that completely ruin model capability while appearing safe, or defences that generalise to the specific attack used in evaluation but collapse against any other. The formalisation does not solve the problem — the field is still working toward truly satisfactory immunisation — but it provides the shared language without which progress cannot be tracked.

The formalism centres on a single key insight: **the relevant quantity is not "does the defence work?" but "does the defence work against an attacker with a given compute budget?"** This framing is what elevates immunisation above generic notions of robustness. A defence that holds for 50 fine-tuning steps but fails at 500 is different from a defence that holds for 50,000 steps. The right level of protection depends on the realistic capabilities of the adversary, not on a single evaluation snapshot.

---

### 1.4.2 The Threat Model: Harmful Fine-Tuning Attacks (HFTAs)

The canonical attacker in the immunisation literature is not a nation-state with unlimited compute. They are an actor with a **limited training budget**: more than zero (they can fine-tune), but substantially less than the cost of pre-training an equivalent LLM from scratch. This assumption is both realistic and important. If we assumed unlimited attacker compute, the problem would be intractable — any model can be destroyed by a sufficiently expensive attacker. The assumption of a bounded budget is what makes the problem solvable in principle.

Formally, the attacker holds a harmful dataset:

$$\mathcal{D}_{\text{harmful}} = \{(X_i, Y_i)\}_{i=1}^{N},$$

where $X_i$ are harmful prompts and $Y_i$ are targeted harmful responses. The attacker minimises the cross-entropy loss over this dataset by gradient descent, taking training steps $t \in \{1, \ldots, T\}$ up to their compute budget $T$:

$$\theta[t^*] = \operatorname{argmin}_{\theta[t]} \;\mathbb{E}_{(X,Y) \sim \mathcal{D}_{\text{harmful}}} \;\mathcal{L}\!\left(M_{\theta[t]}(X), Y\right). \tag{1}$$

The attacker operates in the **white-box** setting: they have full access to the model weights, and can modify architecture, loss function, learning rate, and inference process without restriction. This corresponds to the realistic open-weight deployment scenario — once the weights are public, the attacker can do anything with them.

The defender cannot monitor or control the fine-tuning process. Their only opportunity to act is **before release**: during or after the alignment phase, prior to the model being made available. This is the defining constraint of immunisation, and it is what distinguishes it from fine-tuning-time defences (which modify the training pipeline) and post-hoc defences (which operate on already-degraded models).

A model $M^*$ is called **immunised** against the HFTA defined by Eq. (1) with respect to $\mathcal{D}_{\text{harmful}}$ if it satisfies the four conditions defined below.

---

### 1.4.3 The Four Immunisation Conditions

#### Condition 1 — Resistance

Let $f(\cdot)$ be a harmfulness evaluation function (a proxy classifier, human evaluation, or LLM judge) and $\phi$ be an acceptable harmfulness threshold. Resistance is the core condition: the immunised model must not converge to harmful behaviour under the attacker's training procedure.

**Strong resistance** requires that the model *never* exceeds the harmfulness threshold, regardless of how many training steps the attacker takes:

$$\forall\, t \in T, \quad f\!\left(M^*_{\theta[t]},\, \mathcal{D}_{\text{harmful}}\right) \leq \phi. \tag{2}$$

This is an absolute safety guarantee — the reversed gear that can never be corrected regardless of attacker skill or budget. It is the ideal, but also the most technically demanding condition.

**Weak resistance** relaxes the guarantee: the model may eventually exceed the harmfulness threshold, but only after so many training steps that the attack is economically infeasible. The defender maximises the cost of the attack:

$$\max_t\, f\!\left(M^*_{\theta[t]},\, \mathcal{D}_{\text{harmful}}\right) \leq \phi, \tag{3}$$

where the $\max$ is taken up to some specified budget $T$. Weak resistance does not promise absolute safety; it promises that the attacker must spend more compute than is worth it given their objective.

The distinction is not merely academic. Part 3's vocabulary maps cleanly: strong resistance corresponds to the "collapse" philosophy (the model becomes useless before becoming harmful); weak resistance corresponds to the "raise the cost" philosophy (the model passively resists for longer than the attacker can sustain). The field currently produces far more weak resistance results than strong ones.

**An important asymmetry to note upfront (from Qi et al., 2025):** resistance claims depend entirely on the strength of the attack evaluated. A defence that achieves weak resistance against a 100-step fine-tuning adversary may fail in 150 steps. Evaluations that use weak adversaries can overstate resistance — a preliminary consideration we will return to in Section 4.6.

---

#### Condition 2 — Stability

Immunisation must not damage the model's general capability. Let $g(\cdot)$ be a capability evaluation function on a reference benign dataset $\mathcal{D}_{\text{ref}}$. Stability requires that the immunised model $M^*$ performs equivalently to the non-immunised model $M$ at the time of release:

$$g\!\left(M_{\theta[t=0]},\, \mathcal{D}_{\text{ref}}\right) \approx g\!\left(M^*_{\theta[t=0]},\, \mathcal{D}_{\text{ref}}\right). \tag{4}$$

Stability is both a usability condition and a safety condition. On usability: a model that cannot perform basic language tasks has no commercial value and will not be deployed. On safety: stability also implies that immunisation has not introduced new vulnerabilities — for example, an immunised model should not be *more* susceptible to inference-time jailbreaks than the original.

Evaluating stability correctly requires careful benchmark selection. Loss on a single reference dataset, even a large one, is insufficient. Rosati et al. recommend a comprehensive evaluation across natural language generation benchmarks that cover diverse capabilities. A model that passes narrow stability benchmarks while degrading on others has not demonstrated true stability.

---

#### Condition 3 — Generalisation

The defender does not have access to the samples the attacker will use. Therefore, a defence trained on one harmful dataset $\mathcal{D}_{\text{harm}}$ must resist attacks using a disjoint subset $\mathcal{D}'_{\text{harm}} \cap \mathcal{D}_{\text{harm}} = \emptyset$.

**In-domain generalisation** requires resistance against unseen samples from the *same* harmful domain the defence was designed for. If the defence was developed using toxic-content examples, it must resist attacks using other toxic-content examples not seen during immunisation.

**Cross-domain generalisation** is the stronger and more practically important condition: the defence must resist attacks from harmful domains that are entirely different from those used during immunisation. A defence trained on toxic text generation should, ideally, also resist harmful QA, phishing templates, and weaponisation knowledge queries.

Cross-domain generalisation is an open research question. The honest answer from the current literature is that most methods demonstrate in-domain generalisation adequately but struggle with cross-domain generalisation. Whether a single immunisation procedure can provide cross-domain resistance is not yet settled.

---

#### Condition 4 — Trainability

Open-weight models exist to be fine-tuned. A model that cannot be adapted for legitimate downstream tasks — medical question answering, code generation, domain-specific instruction following — has no practical value. Trainability requires that the immunised model remains fine-tunable for benign tasks at a comparable rate to the non-immunised model:

$$\min_\theta\, g\!\left(M^*_{\theta[t_1]},\, \mathcal{D}_{\text{ok}}\right) \approx \min_\theta\, g\!\left(M_{\theta[t_2]},\, \mathcal{D}_{\text{ok}}\right) \quad \text{s.t. } |t_1 - t_2| \leq \varepsilon, \tag{5}$$

for benign datasets $\mathcal{D}_{\text{ok}}$. Trainability is technically optional for the formal definition of a secure defence — a model could be immunised to the point of near-total rigidity — but it is commercially and practically mandatory for open-weight deployment. Trainability is where the tension between resistance and utility is most acute, and where the hardest engineering challenges lie.

---

### A Summary View of the Four Conditions

| Condition | What it requires | Primary tension |
|---|---|---|
| **Resistance** (strong) | Safety never collapses under unlimited attack | Hard to achieve; may require sacrificing trainability |
| **Resistance** (weak) | Safety survives the realistic attacker's budget | Depends critically on what "realistic" means |
| **Stability** | Capability unchanged at release time | Methods that over-immunise damage utility |
| **Generalisation** | Resists unseen and cross-domain attacks | Most methods fail cross-domain |
| **Trainability** | Benign fine-tuning remains efficient | Directly conflicts with strong resistance |

The four conditions are not equally hard. Stability is routinely achieved: most proposed methods successfully preserve general capability at release time. Trainability is achieved by the majority of cost-raising methods (Philosophy 1 from Part 3) but rarely by the collapse methods (Philosophy 2). Generalisation is mixed: in-domain is usually demonstrated, cross-domain is often not. Resistance is the most contested: methods differ enormously in how strong a resistance they can demonstrate, and evaluations vary widely in adversarial strength.

---

### 1.4.4 An Alternative Lens: Immunisation as Differential Conditioning of the Hessian

The Rosati framework defines immunisation through *behavioural* conditions: what the model does or does not do under attack. A complementary and geometrically richer definition comes from a perspective on the loss landscape itself.

The key insight, formalised in the condition number paper (Boursinos & Iosifidis, 2023), is that the speed at which gradient descent converges on a task is governed by the **condition number** of the Hessian of the loss:

$$\kappa(\mathbf{H}) = \frac{\sigma_{\max}(\mathbf{H})}{\sigma_{\min}(\mathbf{H})},$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values of the Hessian, respectively. Recall from standard optimisation theory that the convergence of gradient descent satisfies:

$$\|\mathbf{w}_t - \mathbf{w}^*\|^2 \leq \left(1 - \frac{1}{\kappa(\mathbf{H})}\right)^t \|\mathbf{w}_0 - \mathbf{w}^*\|^2.$$

When $\kappa$ is large (ill-conditioned), the factor $(1 - 1/\kappa)$ is close to 1, and convergence is painfully slow. When $\kappa \approx 1$ (well-conditioned), the factor approaches 0 and convergence is rapid. An attacker using gradient descent on an ill-conditioned loss landscape may need exponentially more steps to reach a given harmful performance level.

The immunisation problem, from this perspective, becomes: **engineer the feature extractor $\theta$ such that the harmful task's Hessian $\mathbf{H}_H(\theta)$ is maximally ill-conditioned while the benign task's Hessian $\mathbf{H}_P(\theta)$ remains well-conditioned.** Formally, the three conditions are:

**(a)** The immunised feature extractor $\theta^I$ should make fine-tuning on the harmful task significantly harder than an identity baseline:

$$\kappa\!\left(\nabla^2_\mathbf{w} \mathcal{L}(\mathcal{D}_H, \mathbf{w}, \theta^I)\right) \gg \kappa\!\left(\nabla^2_\mathbf{w} \mathcal{L}(\mathcal{D}_H, \mathbf{w}, \mathbf{I})\right). \tag{5'}$$

**(b)** Fine-tuning on the primary benign task should be no harder after immunisation:

$$\kappa\!\left(\nabla^2_\omega \mathcal{L}(\mathcal{D}_P, \omega, \theta^I)\right) \leq \kappa\!\left(\nabla^2_\omega \mathcal{L}(\mathcal{D}_P, \omega, \mathbf{I})\right). \tag{6'}$$

**(c)** The immunised model should maintain competitive task performance on the primary dataset:

$$\min_{\omega, \theta} \mathcal{L}(\mathcal{D}_P, \omega, \theta) \approx \min_\omega \mathcal{L}(\mathcal{D}_P, \omega, \theta^I). \tag{7'}$$

These three conditions map directly onto Rosati's framework: (5') is resistance, (6') is trainability, and (7') is stability. The condition number perspective is richer because it provides a *single differentiable quantity* — the condition number of the task Hessian — that can be optimised during training. The resulting immunisation objective is:

$$\min_{\omega, \theta}\; \mathcal{R}_{\text{ill}}(\mathbf{H}_H(\theta)) + \mathcal{R}_{\text{well}}(\mathbf{H}_P(\theta)) + \mathcal{L}(\mathcal{D}_P, \omega, \theta), \tag{11}$$

where $\mathcal{R}_{\text{ill}}$ is a regulariser that maximises $\kappa(\mathbf{H}_H)$ and $\mathcal{R}_{\text{well}}$ is a regulariser that minimises $\kappa(\mathbf{H}_P)$. The paper proves that these regularisers have **monotone gradient updates**: applying a single gradient step of $\mathcal{R}_{\text{ill}}$ strictly increases $\kappa(\mathbf{H}_H)$, and a single gradient step of $\mathcal{R}_{\text{well}}$ strictly decreases $\kappa(\mathbf{H}_P)$. This theoretical guarantee does not require convexity, making it broadly applicable.

The Relative Immunisation Ratio (RIR) provides a single scalar evaluation metric for this framework:

$$\text{RIR} \;\triangleq\; \frac{\kappa(\mathbf{H}_H(\theta^I)) \;/\; \kappa(\mathbf{H}_H(\mathbf{I}))}{\kappa(\mathbf{H}_P(\theta^I)) \;/\; \kappa(\mathbf{H}_P(\mathbf{I}))}.$$

An RIR $> 1$ means the harmful task has become harder relative to the benign task — the immunisation is working asymmetrically, as intended. The denominator guards against the degenerate case where both tasks become harder equally, which would indicate that the feature extractor has simply been damaged.

The condition number framing is not limited to the toy linear models where the theory was proven. Empirically, it transfers to deep networks. A striking result: immunising the last two blocks of a ViT model yields an RIR of up to 41, while ImageNet accuracy *increases* after immunisation — the constraint imposed by the harmful-task regulariser appears to act as a beneficial feature-space compression. This suggests that the ill-conditioning of the harmful Hessian and the improvement of the benign task are not in fundamental opposition.

---

### 1.4.5 A Note on Normative Scope: What Immunisation Is and Is Not

The Rosati framework is explicit about a limitation that deserves to be stated clearly at the outset.

**Immunisation assumes models are already made safe at inference time.** It does not claim to solve the jailbreaking problem, the prompt injection problem, or the general alignment problem. It addresses the orthogonal question: given a model that is already safely aligned, how do we ensure that safe alignment cannot be cheaply erased by a downstream fine-tuner?

If a model is not safe at inference time, immunising it merely embeds an insecure state more deeply into the weights. The reversed gear on an armored car that was already unsafe does not make it safe — it just ensures it remains exactly as unsafe as it was.

**Immunisation requires a normative definition of harm.** The set $\mathcal{D}_{\text{harmful}}$ must be constructed by someone, and that construction reflects judgments about what counts as harmful. These judgments are not neutral. Harm definitions may privilege certain communities over others, may be culturally relative, and may fail to anticipate new forms of harm that arise after deployment. Rosati et al. acknowledge this directly: defining harm is a contentious issue endemic to LLM safety research more broadly, and immunisation inherits all of that complexity.

**The framework currently covers only supervised fine-tuning attacks.** RL-based attacks — where an adversary uses DPO, PPO, or similar methods to modify the model — are explicitly excluded from the Rosati framework's scope. Given the evidence from Part 2 that RL attacks can be more effective than SFT attacks at bypassing alignment, this is a meaningful gap. Methods like TokenBuncher (Part 5 of this tutorial) specifically target RL attacks, but the formal conditions have not yet been extended to cover them systematically.

**The dual-use risk of the immunisation datasets themselves.** The harmful datasets used to construct immunised models — if shared publicly — can be repurposed as attack tools. This is not a reason to avoid immunisation research, but it does require thoughtful dissemination practices.

---

### 1.4.6 Preliminary Considerations from Two Critical Papers

Before proceeding to the technical methods in the later parts of this tutorial, two recent papers raise foundational concerns that every practitioner should carry forward. These are not fatal objections to the field — they are diagnostic contributions that sharpen what we need to achieve.

#### 1.4.6.1 The G-Effect: The Gradient Perspective on Unlearning and Retention (Wang et al., 2025)

Wang et al. (2025) propose a diagnostic tool called the **G-effect**: an analytical approximation of the performance change induced by an unlearning objective, computed as a dot product of gradients without running full training. Formally, for a risk metric $\mathcal{R}$ and an unlearning objective applied to parameters $\theta$, the G-effect on dataset $\mathcal{D}'$ is:

$$\text{G-effect}(\mathcal{D}'; \mathcal{D}_u) \approx -\nabla_\theta \mathcal{R}(\mathcal{D}'; \theta)^\top \cdot \nabla_\theta \mathcal{L}_u(\theta; \mathcal{D}_u),$$

measuring the expected performance change on $\mathcal{D}'$ induced by taking a gradient step toward the unlearning objective on $\mathcal{D}_u$.

This tool reveals four structural findings that are directly relevant to immunisation:

**Finding 1 — Resistance and retention are in gradient tension.** Effective unlearning requires large gradient updates that push the model away from the harmful distribution. But large updates also corrupt the model's performance on benign data. The G-effect formalises this as a trade-off that cannot, in general, be resolved by simple regularisation: the gradient directions that most effectively remove harmful knowledge tend to be the same directions that damage benign capability.

**Finding 2 — Unlearning affects shallow layers disproportionately.** The G-effect is concentrated in early-to-middle transformer layers — precisely the layers where general knowledge is most densely encoded. Immunisation techniques that operate uniformly across all layers may be inadvertently focusing their force on the most capability-sensitive part of the model.

**Finding 3 — Excessive unlearning is harmful.** Beyond a certain intensity, improvements in removal quality come at a worse cost in retention quality than less aggressive unlearning. There is an optimal unlearning "temperature" — and most current methods do not have reliable mechanisms for identifying it.

**Finding 4 — NLL as a risk metric can be misleading.** The standard practice of measuring unlearning success by the increase in NLL on the forget set can be gamed. A model that simply increases NLL on the target distribution without actually forgetting the knowledge — for example, by learning to produce incorrect but confident responses — may pass an NLL-based evaluation while retaining the underlying harmful capability. This is a particular concern for immunisation evaluations that rely on loss metrics rather than direct harmful output testing.

The G-effect framework's core lesson for immunisation: **the mechanism by which a defence achieves its reduction in harmful capability matters, not just the reduction itself.** Evaluating immunisation by measuring ASR or loss on a specific harmful dataset, without understanding the gradient geometry, can miss failure modes that will appear under a different attacker or a different evaluation protocol.

#### 1.4.6.2 On Evaluating the Durability of Safeguards (Qi et al., 2025)

Qi et al. (2025) conduct a systematic re-evaluation of published immunisation and safeguard methods under stronger, more diverse, and more adaptive attacks. Their findings are sobering.

**Finding 1 — Most published safeguards collapse under stronger adversaries.** Methods that report high tamper-resistance against simple fine-tuning attacks (fixed hyperparameters, standard optimisers, small datasets) frequently collapse when the attacker varies the learning rate, uses different optimisation algorithms, scales the attack dataset, or employs a combination of strategies. The "durability" of a safeguard is not a fixed property — it is a property relative to the attack evaluated against.

**Finding 2 — Vanilla ASR is an insufficient evaluation metric.** The standard evaluation — measure ASR on a benchmark dataset after a fixed number of fine-tuning steps — systematically underestimates attacker capability. An adaptive attacker who can grid-search over hyperparameters or compose attacks will consistently find ways around defences that appear robust under non-adaptive evaluation. Qi et al. propose adversarial evaluation protocols that include hyperparameter search, varied optimisers, and multi-stage attacks.

**Finding 3 — There is an implicit "robustness spectrum".** Safeguards are not binary (robust or not) — they exist on a spectrum of how expensive the attacker must be to overcome them. The appropriate evaluation asks: at what attacker compute budget does the safeguard fail? This is precisely Rosati's weak resistance condition, but operationalised with adaptive attacks rather than fixed-parameter ones. Most published results do not answer this question because they evaluate only at a single budget point.

**Finding 4 — The gap between training attacks and test attacks is critical.** Methods that simulate the attacker during immunisation training (TAR, Antidote, and similar adversarial meta-learning approaches) are more robust to unseen attacks than methods that use fixed perturbations. But even adversarial meta-learning methods can fail when the test attack deviates significantly from the training attack distribution. The choice of training attack is therefore not a neutral engineering decision — it is a hypothesis about what real-world attackers will do, and that hypothesis can be wrong.

These findings do not invalidate immunisation research. They specify what rigorous immunisation research looks like. An immunisation paper that does not evaluate under adaptive, multi-attack, multi-budget conditions has not demonstrated the claim it is implicitly making. This tutorial will use this standard when assessing the methods covered in later parts.

---

### 1.4.7 The Tension as a Productive Research Programme

Bringing together the four conditions, the condition number perspective, and the two critical papers yields a clear picture of where the field stands.

The conditions are well-defined and widely adopted. The condition number framework provides a differentiable, optimisable formulation that unifies resistance and trainability in a single geometric quantity. The G-effect reveals that gradient-based removal of harmful capability necessarily disturbs benign capability, and that this tension is structural rather than incidental. The durability critique establishes that current evaluation standards systematically understate attacker capability and therefore overstate defence quality.

Together, these constraints define a **productive research programme**: methods that achieve weak resistance under strong adaptive attacks, while preserving stability and trainability, and whose gradient geometry can be understood in terms of the G-effect. The later parts of this tutorial examine how far current methods have progressed toward this target.

The key open questions, which will frame the technical discussion ahead:

1. **Is strong resistance achievable without sacrificing trainability?** Current evidence suggests it is not — strong resistance and trainability appear to be in fundamental tension. The field has not yet found the parameter configuration that achieves both.

2. **Can cross-domain generalisation be engineered?** Most methods achieve in-domain generalisation. Cross-domain generalisation requires that the immunisation embeds a property of the loss landscape (not a property of the specific harmful data), which is a much harder target.

3. **What is the right evaluation standard?** The durability critique establishes that vanilla ASR under fixed attacks is insufficient. The field needs standardised adversarial evaluation protocols with explicit attacker budget accounting.

4. **How does the G-effect constrain what is achievable?** If resistance and retention are in fundamental gradient tension, there may be theoretical upper bounds on how much immunity a model can have before its benign capability necessarily degrades. Characterising these bounds is an open theoretical problem.

The rest of this tutorial is organised around these questions.

---

### References for Part 1.4

- Rosati, D., Wehner, J., Williams, K., Bartoszcze, Ł., Sajjad, H., and Rudzicz, F. **Immunization against harmful fine-tuning attacks.** EMNLP Findings, 2024.
- Boursinos, D. and Iosifidis, A. **Model immunization from a condition number perspective.** ICML 2023.
- Wang, Q., Zhou, J.P., Zhou, Z., Shin, S., Han, B., and Weinberger, K.Q. **Rethinking LLM unlearning objectives: A gradient perspective and go beyond.** ICLR 2025.
- Qi, X., Wei, B., Carlini, N., Huang, Y., Xie, T., He, L., Jagielski, M., Nasr, M., Mittal, P., and Henderson, P. **On evaluating the durability of safeguards for open-weight LLMs.** arXiv:2412.07097, 2025.
- Tamirisa, R. et al. **Tamper-resistant safeguards for open-weight LLMs.** (TAR). ICLR 2025.
