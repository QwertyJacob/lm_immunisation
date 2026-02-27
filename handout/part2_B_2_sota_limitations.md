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
