# Language model Immunisation

## Introduction

**Harmful training**

Attackers achieve harmful fine-tuning by using the training objective below on a given LLM without harmful capabilities which we denote as $M_{\theta[t=0]}$, where $t$ indicates the number of optimization steps taken ($t=0$ is the initial model) to find the parameters $\theta$ for model $M$. 


The attacker will use a harmful dataset $D_\text{harmful}=\{X_i,Y_i\}^N_{i=1}$ which includes prompts $X$ and target responses $Y$ designed to elicit harmful behavior in $M_{\theta}$. They then minimize the loss function $\mathcal{L}_{D_\text{harmful}}(M_{\theta[t]}(X), Y)$ by taking training steps $t \in T$ up to their compute budget ($T$). The outcome is the optimal parameters $\theta[t^{\ast}]$ found at some training step $t^{\ast}$ that the attacker can use to engage in illegal and harmful activities that would have previously been refused. This we call \textbf{harmful training} - \cref{eq:harmful_training} - where $\theta[t^{\ast}]$ is found by:


$$
\text{argmin}_{\theta[t]} \mathbb{E}_{(X, Y) \sim D_{\text{harmful}}} \mathcal{L}(M_{\theta[t]}(X), Y)  \tag{1} 
$$

> We assume that the attacker has a limited compute budget to train a harmful model i.e. not enough to train an LLM from scratch.


 We say that a model is \textit{immunized} (we indicate an immunized model as ($M^{\ast}$) against harmful training (1) with respect to $D_\text{harmful}$ if it meets the following conditions:


 Below is a **concise research-style Markdown summary** with the **main conceptual points + key equations preserved**.

---

# **Immunization Conditions (Summary)**

The paper defines **model immunization** against harmful fine-tuning attacks (HFTAs) as a set of four conditions ensuring robustness, usability, and practical deployability. An immunized model is denoted $M^\ast$ and evaluated against a harmful dataset $D_{\text{harmful}}$.

---

## **1. Resistance**

**Goal:** Prevent the model from learning harmful behavior under adversarial training.

A defender defines a harmfulness threshold $\phi$ using a proxy metric $f(\cdot)$ (e.g., toxicity score).

### **Strong Resistance**

The model **never becomes harmful**, regardless of training steps:

$$
\forall t \in T,\quad f(M^{\ast}*{\theta[t]}, D*{\text{harmful}}) \leq \phi
$$

This provides protection even against unlimited attacker budgets.

---

### **Weak Resistance**

The model **eventually may become harmful**, but only after many training steps, making attacks economically infeasible:

$$
\max_{t} f(M^{\ast}_{\theta[t]}, D_{\text{harmful}}) \leq \phi
$$

Weak resistance focuses on increasing attack cost rather than guaranteeing absolute safety.

---

## **2. Stability**

**Goal:** Immunization should not degrade performance on benign tasks.

Using a capability metric $g(\cdot)$ on a reference dataset $D_{\text{ref}}$:

$$
g(M_{\theta[t=0]}, D_{\text{ref}}) \approx g(M^{\ast}_{\theta[t=0]}, D_{\text{ref}})
$$

This ensures the immunized model remains usable and does not become less safe via side effects (e.g., increased jailbreak vulnerability).

---

## **3. Generalization**

**Goal:** Immunization must generalize beyond the specific harmful samples used during defense.

* **In-domain generalization:** defense trained on one harmful subset $D_{\text{harm}}$ resists attacks on another subset from the *same domain*.
* **Cross-domain generalization:** defense trained on one harm domain (e.g., toxic text) resists attacks in *different domains* (e.g., harmful QA).

This models realistic attackers whose data is unknown to the defender.

---

## **4. Trainability**

**Goal:** Immunized models should still be trainable on harmless tasks.

For a benign dataset $D_{\text{ok}}$, fine-tuning efficiency should remain similar:

$$
\min_{\theta} g(M^{\ast}*{\theta[t_1]}, D*{\text{ok}})
\approx
\min_{\theta} g(M_{\theta[t_2]}, D_{\text{ok}})
\quad \text{s.t. } |t_1 - t_2| \leq \epsilon
$$

Trainability is optional for security but critical for practical and commercial usability.

---

# **Conceptual Takeaways**

* **Resistance**: blocks or delays harmful fine-tuning.
* **Stability**: preserves benign performance.
* **Generalization**: protects against unseen attack distributions.
* **Trainability**: keeps the model useful for legitimate training.

These conditions formalize **defensive robustness vs. utility trade-offs** in LLM immunization.

