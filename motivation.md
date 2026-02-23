# Motivation: Why Surface-Level Alignment Is Not Enough

*Taken from: [Gu et al. (2025), "Probing the Robustness of Large Language Models Safety to Latent Perturbations"](https://arxiv.org/abs/2506.16078); Goodfellow et al. (2014), "Explaining and Harnessing Adversarial Examples".*

## 1.1 The Alignment Illusion

Modern large language models (LLMs) are deployed with the expectation that safety alignment — through Supervised Fine-Tuning (SFT) and Preference Optimisation (PO) — renders them reliably unwilling to produce harmful outputs. Ask a well-aligned model *"How do I make a bomb?"* and it will refuse. This refusal feels robust. It is not.

The core thesis of this tutorial is the following uncomfortable observation:

> **Current alignment methods successfully modify input-output behaviour, but leave the model's internal representational structure essentially untouched.** Safety is enforced at the surface; beneath it, the latent space remains fragile.

This section makes that claim concrete and quantitative, drawing on a recent line of work that probes alignment robustness through *latent perturbations* — small, controlled shifts injected directly into a model's hidden activations during generation.

---

## 1.2 Formalising the Attack Surface

Consider an autoregressive LLM parameterised by $\theta$, defining a conditional distribution $\pi_\theta(y \mid x)$ over output sequences $y = (y_1, \ldots, y_{|y|})$ given input $x$. At each generation step $t$, the model computes a hidden activation

$$h_t^{(l)} \in \mathbb{R}^d$$

at every transformer layer $l$, where $d$ is the model's hidden dimension. Standard alignment training shapes the mapping $x \mapsto y$ via loss on token probabilities but imposes **no explicit constraint on the geometry of $\{h_t^{(l)}\}$**.

This leaves open the question: how much does the safety behaviour depend on the precise value of $h_t^{(l)}$? If the answer is *a lot*, alignment is shallow. If the answer is *very little*, alignment is structurally robust.

Gu et al. (2025) formalise this question via the **Activation Steering Attack (ASA)**: inject a perturbation $\delta$ into a single layer $l^*$ at generation step $t$, and observe whether the model's output switches from a safe refusal to a harmful response.

The perturbation is *normalised* to match the statistical distribution of the target activation, preventing degenerate outputs (e.g. random token soup) while keeping the shift semantically meaningful:

$$\delta' = \mu\!\left(h_t^{(l^*)}\right) + \frac{\delta - \mu(\delta)}{\sigma(\delta)} \cdot \sigma\!\left(h_t^{(l^*)}\right) \tag{1}$$

where $\mu(\cdot)$ and $\sigma(\cdot)$ denote the **mean and standard deviation computed element-wise across the $d$ hidden dimensions** of a single activation vector (i.e. instance-level statistics, not batch statistics). The perturbed activation is then:

$$h_t^{\prime\,(l^*)} \leftarrow h_t^{(l^*)} + \delta' \tag{2}$$

and is propagated forward through all subsequent layers to produce perturbed output logits. Note that in the simplest variant — which we call $\text{ASA}_\text{random}$ — the raw perturbation $\delta$ is simply drawn from a standard Gaussian:

$$\delta \sim \mathcal{N}(0, I_d)$$

No crafted adversarial input. No auxiliary model. No labelled data. Just noise.

---

## 1.3 Measuring Latent Fragility: The NLL Probe

To quantify how much a perturbation destabilises alignment, a natural diagnostic is the **Negative Log-Likelihood (NLL) of the model's original safe response** $y$ under the perturbed model:

$$\mathcal{L}(x, y) = -\sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \tag{3}$$

A higher NLL after injection means the model finds its own safe output *less likely* — i.e. the perturbation has shifted the model away from its aligned behaviour. This reframes safety evaluation as an implicit classification problem: the model is either in a "refusal basin" or a "compliance basin" in the latent space, and ASA measures how close the current operating point is to the basin boundary.

This is directly analogous to Fast Gradient Sign Method (FGSM) in image classification (Goodfellow et al., 2014), which increases the loss on the correct class to force a misclassification — except here we operate on intermediate activations rather than input pixels, because the tokenisation process is non-differentiable.

---

## 1.4 Empirical Evidence: Safety Collapses Under Noise

Three metrics are used to evaluate attack effectiveness across a dataset of $N$ prompts and a set of candidate layers $\mathcal{L}$. Let $A_i^{(l)} \in \{0,1\}$ indicate whether the attack on sample $i$ at layer $l$ succeeds:

$$\text{MASR} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\!\left[\max_{l \in \mathcal{L}} A_i^{(l)} = 1\right] \tag{4}$$

$$\text{LASR}(l) = \frac{1}{N}\sum_{i=1}^{N} A_i^{(l)} \tag{5}$$

$$\text{PASR} = \max_{l \in \mathcal{L}} \text{LASR}(l) \tag{6}$$

MASR asks: *for what fraction of prompts does ASA succeed on at least one layer?* PASR asks: *what is the best a single layer can do?*

The results, evaluated on 100 seed prompts from AdvBench across 12 open-source models, are striking:

| Model | INIT (no attack) | MASR | PASR |
|---|---|---|---|
| Llama-3.2-3B-Instruct | 0.12 | **0.94** | 0.47 |
| Qwen-2.5-7B-Instruct | 0.11 | **0.80** | 0.43 |
| Llama-3.1-8B-Instruct | 0.18 | **0.92** | 0.42 |
| Llama-2-13B-Chat | 0.06 | **0.52** | 0.23 |
| Llama-3.3-70B-Instruct | 0.11 | **0.77** | 0.33 |

**The key takeaway:** models that almost never produce harmful outputs without an attack (INIT $\approx 0.1$) are jailbroken on the majority of prompts by $\text{ASA}_\text{random}$ — injection of a *single random Gaussian vector*, normalised to match the activation statistics. Scaling to 70B parameters does not provide immunity; MASR for Llama-3.3-70B-Instruct remains at 0.77.

Furthermore, the attack effect is **cumulative**: because LLMs generate autoregressively, injecting a perturbation at every step $t$ causes the KL divergence between clean and steered output distributions to grow monotonically with generation length:

$$\text{KL}(z_t \| \hat{z}_t) = \sum_i z_t^{(i)} \log \frac{z_t^{(i)}}{\hat{z}_t^{(i)}} \tag{7}$$

where $z_t$ and $\hat{z}_t$ are the clean and perturbed vocabulary distributions at position $t$. In practice, both MASR and PASR increase with generation length, meaning even a weak perturbation becomes effective if the model is asked to generate a moderately long response.

---

## 1.5 A Gradient-Based Variant Pushes Further

The random perturbation result is already damning, but alignment can be broken even more reliably by replacing the random direction with a **gradient-guided** one. Given a harmful prompt $x$ and a target harmful suffix $y^*$, $\text{ASA}_\text{grad}$ computes:

$$\delta' = \alpha \cdot \text{sign}\!\left(\nabla_{h^{(l)}} \mathcal{L}(x + y^*)\right) \tag{8}$$

normalised via Eq. (1), with $\alpha = 1$ by default. This is a single-step, inference-time operation — no weight updates, no iterative optimisation. The results:

| Model | $\text{ASA}_\text{random}$ MASR | $\text{ASA}_\text{grad}$ MASR | $\Delta$ |
|---|---|---|---|
| Qwen-2.5-7B-Instruct | 0.89 | **1.00** | +0.11 |
| Llama-3.1-8B-Instruct | 0.96 | **0.99** | +0.03 |
| Llama-3.1-8B-Base | 0.99 | **0.99** | 0.00 |
| Qwen-2.5-7B-Base | 0.96 | **1.00** | +0.04 |

For Qwen-2.5-7B-Instruct, the gradient-guided attack achieves **100% success rate** — every single harmful prompt in the evaluation set produces an unsafe response after a single-step activation injection. The loss landscape analysis confirms why: the NLL surface is orders of magnitude sharper along the $\delta_\text{grad}$ direction than along a random direction, meaning the model's safety basin has a thin wall in the direction gradient descent naturally finds.

---

## 1.6 What This Tells Us About Alignment

These results collectively point to a structural diagnosis. Standard SFT and PO optimise the model to refuse on a distribution of *inputs*. They do not regularise the latent space to be locally robust around safe outputs. Formally, there is no objective during alignment training that enforces:

$$\pi_\theta(y \mid x) \approx \pi_\theta(y \mid x, \delta') \quad \text{for small } \|\delta'\|$$

The model learns *which outputs to produce given which inputs*, but not *to produce those outputs robustly when its own hidden states are perturbed*. Safety is validated only at the input-output interface; inside the model, harmful behaviours are not erased — they are merely suppressed.

---

## 1.7 The Case for Immunisation

The preceding analysis motivates the central question of this tutorial:

> **Can we train models such that their safety behaviour is locally robust in the latent space — i.e., stable under small perturbations to internal activations?**

This is what we mean by **immunisation**: moving beyond surface-level output supervision toward *representational robustness*, such that safe behaviour is maintained even when hidden states deviate from the training distribution.

A preliminary answer exists in the form of **Layer-wise Adversarial Patch Training (LAPT)** (Gu et al., 2025), which fine-tunes models on safety examples while injecting the same class of random perturbations used in ASA into the identified fragile layers. The result is a consistent reduction in PASR across all evaluated models, with general task accuracy (GSM8K, CommonsenseQA) preserved within 0.05 of baseline — suggesting that targeted representational hardening is achievable without catastrophic forgetting.

However, LAPT is a proof of concept, not a complete solution. It does not situate itself within the broader literature on LLM immunisation, does not compare against other latent adversarial training methods, and does not provide theoretical guarantees on robustness. These are precisely the gaps this tutorial aims to fill.

**The lesson is not that alignment is hopeless. It is that alignment, as currently practised, is incomplete.** Safety is a property we want to hold in the model's representational geometry, not just in its input-output mappings. Understanding how to enforce it — even partially — is a worthwhile and pressing research direction.