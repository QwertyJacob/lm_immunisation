# Part 2.B.2 — What the State of the Art Still Gets Wrong

> **Session:** Afternoon · Part B · 10 minutes  
> **Role of this segment:** Forensic, not pessimistic. The field has produced real methods — this morning's papers are genuine contributions. But a set of critique papers published alongside them have identified structural failure modes that no current method fully resolves. This segment names them precisely, with evidence.

---

## How to Read This Section

These are not attacks on individual papers. Most of the limitations below apply to the entire field simultaneously. They are failure modes of the *paradigm*, not of specific implementations — which means fixing one paper does not fix the problem. The critique literature exists precisely because the immunisation community has been honest enough to publish it. That honesty is worth honouring by reading it carefully.

We identify five distinct failure modes.

---

## Failure Mode 1: The Obfuscation Problem

*Source: Łucki, Wei, Huang et al. — "An Adversarial Perspective on Machine Unlearning for AI Safety" (TMLR, 2025)*

The most fundamental critique in the literature goes straight at the premise of Block 1 unlearning methods: **they do not erase hazardous knowledge — they hide it**.

The paper performs white-box evaluations of state-of-the-art unlearning methods (RMU, NPO) for hazardous knowledge, comparing them against standard DPO safety training. The verdict is stark. Jailbreak methods previously reported as ineffective against unlearning can recover substantial accuracy after small changes to the loss function. More damningly: **fine-tuning on 10 completely unrelated examples can fully recover supposedly unlearned capabilities**. Not related examples — unrelated ones. The optimization landscape retains the hazardous knowledge in some dormant form, and any perturbation of sufficient magnitude can dislodge it.

The mechanism is as follows. Current unlearning methods modify the model's output distribution — they push hazardous queries toward high loss, low confidence, or refusal. But the *weights themselves* still encode the relevant knowledge. Removing specific directions in the activation space, or shifting the loss landscape slightly, renders the knowledge accessible again. The paper's conclusion: unlearning for LLMs is currently closer to a sophisticated obfuscation than to actual erasure. The difference matters enormously for immunisation: a locked vault is not the same as an empty one.

> **Implication for the field:** Claims of "knowledge removal" via unlearning should be interpreted as "knowledge suppression under the specific attack surface evaluated." The appropriate evaluation is white-box, adaptive, and adversarial — not ASR on a fixed benchmark.

---

## Failure Mode 2: The G-Effect — Unlearning's Collateral Damage

*Source: "Rethinking LLM Unlearning Objectives" (ICLR, 2025)*

The G-effect is a precise diagnostic for a problem that practitioners have observed but rarely quantified: **gradient ascent (GA) for unlearning causes excessive collateral damage to non-targeted data**.

The G-effect measures the inner product $\nabla_\theta \mathcal{L}(D; \theta)^\top \nabla_\theta p(s^i_u | s^{<i}_u; \theta)$ — the alignment between the gradient of general retained data and the gradient of the unlearning update per token. When this inner product is strongly negative, the unlearning step is simultaneously damaging the model's representation of general, non-targeted knowledge. The paper shows this effect is large, layer-dependent, and most severe in shallow layers where general knowledge is stored.

The culprit is an **inverse confidence mechanism**. Standard GA weights each token's gradient by $1/p(s^i_u | s^{<i}_u; \theta)$ — the reciprocal of the model's confidence. As unlearning progresses and the model becomes less confident on the targeted tokens, these tokens receive disproportionately large gradient updates. Small negative values of the retaining G-effect compound across steps into severe general capability degradation.

The paper's remedy — Weighted GA (WGA), which reweights each token by $p(s^i_u | s^{<i}_u; \theta)^\alpha$ — attenuates the inverse confidence mechanism and substantially reduces collateral damage. But the deeper point is diagnostic: **any unlearning method that uses gradient ascent implicitly inherits this trade-off**. The G-effect is a structural property of the loss function, not a bug in any particular implementation.

> **Implication for the field:** Tamper-resistance and utility preservation are not independent objectives. Methods reporting low collateral damage should show G-effect analysis, not just downstream task benchmarks.

---

## Failure Mode 3: The Sparsity of Safety

*Source: Wei, Huang, Huang et al. — "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications" (ICML, 2024)*

This paper delivers one of the most uncomfortable empirical findings in the field. By identifying neurons and ranks that are critical *specifically* to safety behaviour (as opposed to general utility), it reveals that safety-critical parameters are remarkably sparse: **approximately 3% at the neuron level and 2.5% at the rank level**.

This sparsity has two implications that cut in opposite directions.

First, it confirms the structural fragility of current alignment. Removing the 3% of parameters responsible for safety while leaving the other 97% intact drops the model's safety completely — Attack Success Rate goes from near-zero to over 90% — while barely affecting utility benchmarks. The safety guardrails are not distributed through the model; they are concentrated in a small identifiable subspace. An attacker who knows this (and the paper publishes the methodology for finding it) can perform surgical alignment removal far cheaper than standard fine-tuning.

Second, and somewhat counterintuitively, this sparsity suggests a potential direction: if safety is sparse now and fragile, could we make it *less* sparse and more entangled with utility? The paper notes that the orthogonality between safety-critical and utility-critical regions is precisely what makes safety brittle. An immunisation strategy that intentionally entangles them — so that disrupting safety necessarily disrupts utility — would be structurally sounder. This is, in fact, the philosophy Block 2 pursues, and the sparsity result provides a mechanistic justification for why Block 1 approaches are limited.

Crucially: **freezing the safety-critical 3% does not solve the problem**. The paper shows that an attacker can work around frozen safety-critical parameters by creating new pathways entirely — introducing LoRA layers that bypass the frozen weights. Structural resistance cannot be achieved by protecting an identifiable subset of parameters when the model's overall trainability allows the attacker to route around it.

> **Implication for the field:** Methods should be evaluated for safety sparsity, not just post-attack ASR. A method that achieves resistance by concentrating safety in an even smaller, more identifiable region may appear to perform well on standard benchmarks while being more vulnerable to adaptive attack.

---

## Failure Mode 4: The Durability-Similarity Coupling

*Source: Hsiung, Pang et al. — "Your Task May Vary: A Systematic Understanding of Alignment and Safety Degradation When Fine-Tuning LLMs" (ICLR, 2025)*

This paper identifies a failure mode that is not about attack methods at all — it is about the structure of the alignment dataset itself. **The durability of safety guardrails after downstream fine-tuning is tightly coupled to the similarity between the upstream alignment data and the downstream fine-tuning data.** When they are similar, guardrails erode faster. When they are dissimilar (diverse), guardrails survive longer.

The finding is empirically sharp. Constructing alignment datasets from high-similarity subsets (data closely matching downstream task distributions) produces models where the same downstream fine-tuning increases Attack Success Rate by 5–10% relative to alignment datasets with low similarity. Benign list-format Alpaca data — data with no harmful content whatsoever — can erode a safety guardrail more effectively than harmful data if it happens to be representationally similar to the alignment examples.

This has a direct implication for immunisation that the field has not fully absorbed: **the upstream alignment dataset is not just a training resource, it is a security-critical design choice**. An alignment dataset that is diverse — broadly covering the input distribution, low-cosine-similarity to any particular downstream task — produces more durable guardrails even without any immunisation technique applied. Immunisation on top of a narrow, task-similar alignment dataset may produce results that look good on the paper's evaluation distribution but degrade unpredictably on downstream fine-tuning.

The paper also argues for dataset *privacy* as a defence: if an attacker does not know the composition of the alignment dataset, they cannot systematically find high-similarity subsets to exploit. Diversity and privacy together form a passive structural defence.

> **Implication for the field:** Immunisation papers that do not control for alignment dataset diversity cannot cleanly attribute their resistance results to their immunisation method. Dataset composition is a confound that the field has underweighted.

---

## Failure Mode 5: The Evaluation Problem

This is not one paper's finding — it is the *meta-critique* synthesised across all four above.

The immunisation field currently evaluates resistance primarily via Attack Success Rate (ASR) on fixed benchmark datasets (AdvBench, HarmBench, BeaverTails) under specific fine-tuning configurations (SFT with a particular learning rate and step count, tested with standard jailbreak suffixes). This evaluation surface is narrow in at least three ways:

**Narrow attack surface.** ASR on 100 AdvBench prompts under one fine-tuning configuration does not characterise resistance to an adaptive attacker. TAR's robustness evaluation is the current gold standard precisely because it tests across 26 adversaries with varied hyperparameters — and it found that early methods broke under configurations not included in their own evaluations. Most papers in the field test against 1–3 attack configurations. This is not enough.

**Black-box evaluation of a white-box problem.** The Adversarial Perspective paper shows that methods can score well on black-box ASR while still being fully recoverable via white-box access to the activations. Since open-weight release is exactly the scenario we care about, white-box evaluation is not optional — it is the correct threat model. A method that passes black-box evaluation only is not demonstrably safe in the open-weight deployment context.

**ASR measures refusal, not knowledge removal.** A model that refuses harmful requests but still encodes hazardous knowledge is not immunised — it is behaviour-patched. The Adversarial Perspective paper makes this distinction concrete: robustness against activation probing says nothing about robustness against fine-tuning, because the suppression operates at the output level while the knowledge persists at the representation level. Evaluation methodology needs to distinguish between these two.

---
## Considerations from Two Critical Papers

Before proceeding to the technical methods in the later parts of this tutorial, two recent papers raise foundational concerns that every practitioner should carry forward. These are not fatal objections to the field — they are diagnostic contributions that sharpen what we need to achieve.

#### The G-Effect: The Gradient Perspective on Unlearning and Retention (Wang et al., 2025)

Wang et al. (2025), in their paper "Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond" propose a diagnostic tool called the **G-effect**: an analytical approximation of the performance change induced by an unlearning objective, computed as a dot product of gradients without running full training. Formally, for a risk metric $\mathcal{R}$ and an unlearning objective applied to parameters $\theta$, the G-effect on dataset $\mathcal{D}'$ is:

$$\text{G-effect}(\mathcal{D}'; \mathcal{D}_u) \approx -\nabla_\theta \mathcal{R}(\mathcal{D}'; \theta)^\top \cdot \nabla_\theta \mathcal{L}_u(\theta; \mathcal{D}_u),$$

measuring the expected performance change on $\mathcal{D}'$ induced by taking a gradient step toward the unlearning objective on $\mathcal{D}_u$.

This tool reveals four structural findings that are directly relevant to immunisation:

**Finding 1 — Resistance and retention are in gradient tension.** Effective unlearning requires large gradient updates that push the model away from the harmful distribution. But large updates also corrupt the model's performance on benign data. The G-effect formalises this as a trade-off that cannot, in general, be resolved by simple regularisation: the gradient directions that most effectively remove harmful knowledge tend to be the same directions that damage benign capability.

**Finding 2 — Unlearning affects shallow layers disproportionately.** The G-effect is concentrated in early-to-middle transformer layers — precisely the layers where general knowledge is most densely encoded. Immunisation techniques that operate uniformly across all layers may be inadvertently focusing their force on the most capability-sensitive part of the model.

**Finding 3 — Excessive unlearning is harmful.** Beyond a certain intensity, improvements in removal quality come at a worse cost in retention quality than less aggressive unlearning. There is an optimal unlearning "temperature" — and most current methods do not have reliable mechanisms for identifying it.

**Finding 4 — NLL as a risk metric can be misleading.** The standard practice of measuring unlearning success by the increase in NLL on the forget set can be gamed. A model that simply increases NLL on the target distribution without actually forgetting the knowledge — for example, by learning to produce incorrect but confident responses — may pass an NLL-based evaluation while retaining the underlying harmful capability. This is a particular concern for immunisation evaluations that rely on loss metrics rather than direct harmful output testing.

The G-effect framework's core lesson for immunisation: **the mechanism by which a defence achieves its reduction in harmful capability matters, not just the reduction itself.** Evaluating immunisation by measuring ASR or loss on a specific harmful dataset, without understanding the gradient geometry, can miss failure modes that will appear under a different attacker or a different evaluation protocol.

####  On Evaluating the Durability of Safeguards (Qi et al., 2025)

Qi et al. (2025) conduct a systematic re-evaluation of published immunisation and safeguard methods under stronger, more diverse, and more adaptive attacks. Their findings are sobering.

**Finding 1 — Most published safeguards collapse under stronger adversaries.** Methods that report high tamper-resistance against simple fine-tuning attacks (fixed hyperparameters, standard optimisers, small datasets) frequently collapse when the attacker varies the learning rate, uses different optimisation algorithms, scales the attack dataset, or employs a combination of strategies. The "durability" of a safeguard is not a fixed property — it is a property relative to the attack evaluated against.

**Finding 2 — Vanilla ASR is an insufficient evaluation metric.** The standard evaluation — measure ASR on a benchmark dataset after a fixed number of fine-tuning steps — systematically underestimates attacker capability. An adaptive attacker who can grid-search over hyperparameters or compose attacks will consistently find ways around defences that appear robust under non-adaptive evaluation. Qi et al. propose adversarial evaluation protocols that include hyperparameter search, varied optimisers, and multi-stage attacks.

**Finding 3 — There is an implicit "robustness spectrum".** Safeguards are not binary (robust or not) — they exist on a spectrum of how expensive the attacker must be to overcome them. The appropriate evaluation asks: at what attacker compute budget does the safeguard fail? This is precisely Rosati's weak resistance condition, but operationalised with adaptive attacks rather than fixed-parameter ones. Most published results do not answer this question because they evaluate only at a single budget point.

**Finding 4 — The gap between training attacks and test attacks is critical.** Methods that simulate the attacker during immunisation training (TAR, Antidote, and similar adversarial meta-learning approaches) are more robust to unseen attacks than methods that use fixed perturbations. But even adversarial meta-learning methods can fail when the test attack deviates significantly from the training attack distribution. The choice of training attack is therefore not a neutral engineering decision — it is a hypothesis about what real-world attackers will do, and that hypothesis can be wrong.

These findings do not invalidate immunisation research. They specify what rigorous immunisation research looks like. An immunisation paper that does not evaluate under adaptive, multi-attack, multi-budget conditions has not demonstrated the claim it is implicitly making. This tutorial will use this standard when assessing the methods covered in later parts.

---

### The Tension as a Productive Research Programme

Bringing together the four conditions of Immunisation, the condition number perspective, and the two critical papers yields a clear picture of where the field stands.

The conditions are well-defined and widely adopted. The condition number framework provides a differentiable, optimisable formulation that unifies resistance and trainability in a single geometric quantity. The G-effect reveals that gradient-based removal of harmful capability necessarily disturbs benign capability, and that this tension is structural rather than incidental. The durability critique establishes that current evaluation standards systematically understate attacker capability and therefore overstate defence quality.

Together, these constraints define a **productive research programme**: methods that achieve weak resistance under strong adaptive attacks, while preserving stability and trainability, and whose gradient geometry can be understood in terms of the G-effect. The later parts of this tutorial examine how far current methods have progressed toward this target.

The key open questions, which will frame the technical discussion ahead:

1. **Is strong resistance achievable without sacrificing trainability?** Current evidence suggests it is not — strong resistance and trainability appear to be in fundamental tension. The field has not yet found the parameter configuration that achieves both.

2. **Can cross-domain generalisation be engineered?** Most methods achieve in-domain generalisation. Cross-domain generalisation requires that the immunisation embeds a property of the loss landscape (not a property of the specific harmful data), which is a much harder target.

3. **What is the right evaluation standard?** The durability critique establishes that vanilla ASR under fixed attacks is insufficient. The field needs standardised adversarial evaluation protocols with explicit attacker budget accounting.

4. **How does the G-effect constrain what is achievable?** If resistance and retention are in fundamental gradient tension, there may be theoretical upper bounds on how much immunity a model can have before its benign capability necessarily degrades. Characterising these bounds is an open theoretical problem.

The rest of this tutorial is organised around these questions.
## The Map of What Remains

Taken together, these five failure modes trace a coherent outline of the gap between current SOTA and a genuinely robust immunisation result:

| Failure mode | What it shows | What it demands |
|---|---|---|
| Obfuscation | Unlearning hides, not erases | White-box, adaptive evaluation as default |
| G-effect | GA damages non-targeted knowledge structurally | Unlearning objectives beyond vanilla gradient ascent |
| Safety sparsity | 3% of params hold all safety — surgically removable | Entanglement with utility, not isolation |
| Durability-similarity | Guardrail durability is a dataset design question | Alignment dataset diversity as security property |
| Evaluation inadequacy | ASR on fixed configs is not resistance | Diverse, adaptive, white-box evaluation standards |

None of these are solved. All of them have partial responses in the literature. The partial responses — WGA, TAR's multi-adversary evaluation, Block 2's utility entanglement philosophy, SEAM's gradient coupling — are real progress. But the map is not yet filled.

---

*In 2.B.3: the challenge activity. You will work with one of these failure modes directly.*
