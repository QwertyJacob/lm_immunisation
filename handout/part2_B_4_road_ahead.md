# Part 2.B.4 — The Road Ahead: Promising Directions

> **Role of this segment:** Signals of where the field is moving and why there is reason for optimism. 


## The Shift That Is Already Happening

Something has quietly changed in the last year of immunisation research, and it is worth naming before the individual directions.

The field began — Block 1, 2022–2023 — with a largely behavioural framing: make the model *behave* as if fine-tuning is hard. Perturb embeddings. Raise the loss on harmful steps. Flatten the loss landscape. All of these work at the level of what the model *does* under adversarial pressure. The adversarial Perspective paper this afternoon showed us the limit of that approach: behaviour can be patched, but the knowledge underneath it is still there. The vault is locked but not empty.

The shift that is emerging — and you can see it clearly in the most recent papers — is from behaviour-centric to **representation-centric** immunisation. The question is no longer only *can we make fine-tuning expensive?* but *can we change what the model fundamentally is, at the level of its internal geometry?* That is a harder question and a more interesting one.

Four directions are showing early promise.

---

## Direction 1 — Invariant Unlearning

The deepest critique of current unlearning is that it teaches a model to suppress a capability under one distribution of probes, but the capability persists under others. The analogy from causality is direct: standard unlearning learns a spurious correlation between "this is a harmful query" and "refuse" — and spurious correlations break under distribution shift.

**Invariant LLM Unlearning (ILU)** applies the machinery of Invariant Risk Minimisation to this problem. Instead of minimising loss on a single harmful dataset, it requires that the unlearned state remain *stationary* across multiple environments — different phrasings, different fine-tuning datasets, different attack vectors. The stability penalty forces the model to find a representation of safety that is not specific to any one probe, but holds across the causal structure of the problem.

Early results are encouraging: models trained with invariant objectives show substantially better resistance to fine-tuning attacks that use out-of-distribution harmful data — precisely the failure mode the Adversarial Perspective paper identified. The conceptual connection to the obfuscation problem is direct: invariance over environments is exactly the property that distinguishes erasure from suppression.

---

## Direction 2 — Adversarial Hypernetworks

You just ran AntiDote in the challenge. But step back and see what the hypernetwork idea actually represents as a research direction, beyond this one paper.

The fundamental bottleneck in every bi-level immunisation method — TAR, Booster, MAML-based approaches — is that the simulated inner adversary is too weak. First-order gradient approximations underestimate the true worst-case attacker. The training signal for the defender is therefore biased: it is hardened against a strawman.

The hypernetwork approach breaks this bottleneck by replacing the inner loop with a *differentiable adversary that can be trained jointly with the defender in real time*. The cost of generating an attack becomes a constant-size forward pass, not a function of the model scale. This unlocks two things: first, the arms race can actually happen at training time rather than being approximated; second, the gradient signal flowing to the defender is unbiased, which is why AntiDote's decoupled loss achieves better utility preservation than methods with coupled objectives.

The direction this opens is broad. A hypernetwork adversary is a general-purpose component — it could be applied to representation-engineering attacks, to RL-based subversion, to any attack type where you can define a differentiable loss. That generality is what makes it a *research direction* rather than just one paper's contribution.

---

## Direction 3 — Architecture-Level Locking (ArchLock)

Both Block 1 and Block 2 operate at the level of weights and training dynamics. ArchLock asks a different question: what if the architecture itself — not just the weights — encodes a structural constraint that makes harmful adaptation geometrically impossible?

The intuition is that standard fine-tuning works by exploiting the model's general trainability. LoRA, full fine-tuning, gradient ascent on harmful data — all of these assume that the parameter space is sufficiently unconstrained that a harmful trajectory exists within reach. ArchLock's approach is to introduce architectural modifications that partition the parameter space: certain subspaces are rendered structurally inaccessible to gradient updates unless a cryptographic or structural key is present.

This is early-stage work, but it is conceptually important because it represents the first serious attempt to move immunisation from a training-time property to an architecture-time property. The implications for the pre-alignment base model problem — the shared Achilles' heel we identified this morning — are significant. If the base model architecture itself encodes structural resistance, the attacker who starts from the pre-alignment checkpoint still faces the architectural constraint.

---

## Direction 4 — Immunisation as Differential Conditioning of the Hessian

The Rosati framework defines immunisation through *behavioural* conditions: what the model does or does not do under attack. A complementary and geometrically richer definition comes from a perspective on the loss landscape itself.

**The condition number**

Given a general matrix S, the condition number  is defined as

$$\kappa(\mathbf{S}) \triangleq \|\mathbf{S}\|_{2} \|\mathbf{S}^{\dagger}\|_{2} = \sigma_{\mathbf{S}}^{\text{max}} / \sigma_{\mathbf{S}}^{\text{min}}, \tag{1}$$

where $\|\cdot\|_2$ is the spectral norm, which is the largest singular value of the matrix, and $\mathbf{S}^{\dagger}$ is the pseudoinverse of $\mathbf{S}$.


Recall what the pseudoinverse $S^{\dagger}$ is: for a matrix S with SVD $S = U \Sigma V^{\top}$, the pseudoinverse is $S^{\dagger} = V \Sigma^{\dagger} U^{\top}$ where $\Sigma^{\dagger}$ is obtained by taking the reciprocal of each nonzero singular value in $\Sigma$ and transposing the resulting matrix. So if $\Sigma$ has singular values σ₁, σ₂, ..., σₘᵢₙ (where σ₁ ≥ σ₂ ≥ ... ≥ σₘᵢₙ > 0), then $\Sigma^{\dagger}$ will have singular values $\frac{1}{\sigma_{min}}, \frac{1}{\sigma_{min+1}}, \ldots, \frac{1}{\sigma_{max}}$

That's why equation (1) simplifies so cleanly:

$$\kappa(S) = \sigma_{\text{max}} / \sigma_{\text{min}}$$

> **The geometric intuition**: The condition number is asking: *how much does the matrix distort space anisotropically?* If κ ≈ 1, the matrix stretches all directions roughly equally (like a scaled rotation). If κ ≫ 1, it massively stretches some directions while nearly collapsing others — this is what makes gradient descent slow, because a gradient step that's well-sized for the "flat" direction is tiny relative to the "steep" direction, or vice versa.

**In the context of immunisation**, the matrix we care about is the Hessian of the loss landscape with respect to the model parameters. The condition number of this Hessian captures how "twisted" the loss landscape is around a given point. A high condition number means there are directions in parameter space where the loss changes very rapidly (steep) and others where it changes very slowly (flat). This anisotropy makes optimization difficult, as gradient descent struggles to find a step size that works well for all directions.

The key insight, formalised in the condition number paper (Boursinos & Iosifidis, 2023), is that the speed at which gradient descent converges on a task is governed by the **condition number** of the Hessian of the loss:

$$\kappa(\mathbf{H}) = \frac{\sigma_{\max}(\mathbf{H})}{\sigma_{\min}(\mathbf{H})},$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values of the Hessian, respectively. 


Recall from standard optimisation theory that the convergence of gradient descent satisfies:

$$\|\mathbf{w}_t - \mathbf{w}^*\|^2 \leq \left(1 - \frac{1}{\kappa(\mathbf{H})}\right)^t \|\mathbf{w}_0 - \mathbf{w}^*\|^2.$$

When $\kappa$ is large (ill-conditioned), the factor $(1 - 1/\kappa)$ is close to 1, and convergence is painfully slow. When $\kappa \approx 1$ (well-conditioned), the factor approaches 0 and convergence is rapid. An attacker using gradient descent on an ill-conditioned loss landscape may need exponentially more steps to reach a given harmful performance level.

> Read (2) as: *how far am I from the solution after t steps?* The right side is your initial distance $\|\mathbf{w}_0 - \mathbf{w}^*\|^2$, multiplied by a shrinkage factor raised to the power $t$. Since $(1 - \frac{\sigma^{\min}}{\sigma^{\max}})$ is between 0 and 1, the distance geometrically shrinks each step. But the rate of shrinkage depends on the ratio $\sigma^{\min}/\sigma^{\max}$, which is exactly $1/\kappa$. If κ is large (ill-conditioned), this ratio is small, and the shrinkage factor is close to 1, meaning you barely make progress each step. If κ ≈ 1 (well-conditioned), the ratio is close to 1, and the shrinkage factor is close to 0, meaning you rapidly converge to the solution.

**Spoiler**: This is exactly the intuition behind the whole paper: making the harmful task have a huge condition number → making fine-tuning on it painfully slow.


**The setting**

Suppose to have a representational backbone (the feature exctractor):
$$f_{\theta}: \mathbb{R}^{D_{\text{in}}} \to \mathbb{R}^{D_{\text{hid}}}$$

And suppose to have a (linear) classification head:
$$h_{\mathbf{w}}: \mathbb{R}^{D_{\text{hid}}} \to \mathbb{R}^{D_{\mathrm{out}}}$$

Fine-tuning could be seen in this case as focusing only in the head, leaving the backbone frozen:

$$\min_{\mathbf{w}} \mathcal{L}(\mathcal{D}, \mathbf{w}, \theta) \triangleq \min_{\mathbf{w}} \sum_{(\boldsymbol{x}, \boldsymbol{y}) \in \mathcal{D}} \ell(h_{\mathbf{w}} \circ f_{\theta}(\boldsymbol{x}), \boldsymbol{y}) \tag{4}$$

Citing the paper: "*The goal of model immunization is to learn a pre-trained model  $g_{\omega} \circ f_{\theta^{\mathrm{I}}}$ , consisting of a classifier  $g_{\omega}$  and an immunized feature extractor  $f_{\theta^{\mathrm{I}}}$ , such that fine-tuning  $f_{\theta^{\mathrm{I}}}$  on a harmful task is difficult, but not for other tasks.*"

**setting**:

- assume the feature extractor makes no dimensionality reduction, i.e. $\theta \in \mathbb{R}^{D_{in} \times D_{in}}$ I now, weird, but let's assume that for a sec.
- denote a pre-training dataset as  $\mathcal{D}_P = \{(\mathbf{x}, \mathbf{y})\}$  
- harmful dataset as  $\mathcal{D}_H = \{(\mathbf{x}, \tilde{\mathbf{y}})\}$  where  $\mathbf{x} \in \mathbb{R}^{D_{in}}$ . 
- The bad actor performs fine-tuning on  $\mathcal{D}_H$  following Eq. (4). 


The immunisation problem, from this perspective, becomes: **engineer the feature extractor $\theta$ such that the harmful task's Hessian $\mathbf{H}_H(\theta)$ is maximally ill-conditioned while the benign task's Hessian $\mathbf{H}_P(\theta)$ remains well-conditioned.** Formally, the three conditions are:

**(a)** The immunised feature extractor $\theta^I$ should make fine-tuning on the harmful task significantly harder than an identity baseline:

$$\kappa\!\left(\nabla^2_\mathbf{w} \mathcal{L}(\mathcal{D}_H, \mathbf{w}, \theta^I)\right) \gg \kappa\!\left(\nabla^2_\mathbf{w} \mathcal{L}(\mathcal{D}_H, \mathbf{w}, \mathbf{I})\right). \tag{5'}$$

**(b)** Fine-tuning on the primary benign task should be no harder after immunisation:

$$\kappa\!\left(\nabla^2_\omega \mathcal{L}(\mathcal{D}_P, \omega, \theta^I)\right) \leq \kappa\!\left(\nabla^2_\omega \mathcal{L}(\mathcal{D}_P, \omega, \mathbf{I})\right). \tag{6'}$$

**(c)** The immunised model should maintain competitive task performance on the primary dataset:

$$\min_{\omega, \theta} \mathcal{L}(\mathcal{D}_P, \omega, \theta) \approx \min_\omega \mathcal{L}(\mathcal{D}_P, \omega, \theta^I). \tag{7'}$$

These three conditions map directly onto Rosati's framework: (5') is resistance, (6') is trainability, and (7') is stability. The condition number perspective is richer because it provides a *single differentiable quantity* — the condition number of the task Hessian — that can be optimised during training. 



**The immunisation objective** in Zhang et al. is:

$$\min_{\omega, \theta}\; \mathcal{R}_{\text{ill}}(\mathbf{H}_H(\theta)) + \mathcal{R}_{\text{well}}(\mathbf{H}_P(\theta)) + \mathcal{L}(\mathcal{D}_P, \omega, \theta), \tag{11}$$

where $\mathcal{R}_{\text{ill}}$ is a regulariser that maximises $\kappa(\mathbf{H}_H)$ and $\mathcal{R}_{\text{well}}$ is a regulariser that minimises $\kappa(\mathbf{H}_P)$. The paper proves that these regularisers have **monotone gradient updates**: applying a single gradient step of $\mathcal{R}_{\text{ill}}$ strictly increases $\kappa(\mathbf{H}_H)$, and a single gradient step of $\mathcal{R}_{\text{well}}$ strictly decreases $\kappa(\mathbf{H}_P)$. This theoretical guarantee does not require convexity, making it broadly applicable.



The regulariser is formalised in [Nenov et al.](https://arxiv.org/pdf/2410.00169):

$$\mathcal{R}_{\text{well}}(\mathbf{S}) = \frac{1}{2} \|\mathbf{S}\|_{2}^{2} - \frac{1}{2p} \|\mathbf{S}\|_{F}^{2}, \tag{3}$$

Staring enough at this formula, we realise that it is mind-blowing:
- The first term $\|\mathbf{S}\|_{2}^{2}$ is the square of the spectral norm, which is the largest singular value squared. 
- The second term $\|\mathbf{S}\|_{F}^{2}$ is the square of the Frobenius norm, which is the sum of squares of all singular values. (*This follows from the fact that the Frobenius norm is invariant under orthogonal transformations, so* $\|\mathbf{S}\|_F^2 = \|\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\|_F^2 = \|\boldsymbol{\Sigma}\|_F^2 = \sum_i \sigma_i^2$.) Dividing by $p$ (the number of singular values) gives us the average of the squares of the singular values.
- Therefore $\mathcal{R}_{\text{well}}(\mathbf{S})$ is essentially the difference between the largest singular value squared and the average singular value squared. This means that if we were to *minimise* $\mathcal{R}_{\text{well}}$, we would penalise matrices where the largest singular value is much larger than the average, which is exactly what we want to do if we would like to reduce the condition number. In other words, this regulariser encourages the singular values to be more uniform, which in turn reduces the condition number and makes the optimization landscape more well-behaved for gradient descent.

So Zheng et al. use this intuition as is, and -most importantly, they also use it in the opposite direction: they tell a neural net to augment this quantity, so that further fine-tuning has a harder time converging:

$$\mathcal{R}_{\text{ill}}(\mathbf{S}) = \frac{1}{\frac{1}{2k} \left\| \mathbf{S} \right\|_F^2 - \frac{1}{2} \left( \sigma_{\mathbf{S}}^{\min} \right)^2}, \tag{12}$$

Now this regularises is reciprocal or opposite to the good regularisation term in Eq. (3), the basic message is: minimising (12) takes the condition number up. And we are minimising (12) for, or maximising the condition number of, the matrix  $\boldsymbol{H}_{\texttt{H}}(\theta)$, which is the hessian of the loss function for the harmful task, not only for the classification head, but, in their nice setting, for the whole pipeline (feature extractor + classifier).


Although their setting is a bit complex and has some important assumptions (like linearity of the classifier, use of an $\ell_2$  loss, etc.) it still works well in practice. 


The paper introduces the **Relative Immunisation Ratio (RIR)**, a single number that captures both sides of the immunisation goal:

$$\text{RIR} \triangleq  \underbrace{\frac{\kappa(\boldsymbol{H}_H(\theta^I))}{\kappa(\boldsymbol{H}_H(\boldsymbol{I}))}}_{\text{(i) harmful task harder?}} \Bigg/ \underbrace{\frac{\kappa(\boldsymbol{H}_P(\theta^I))}{\kappa(\boldsymbol{H}_P(\boldsymbol{I}))}}_{\text{(ii) pretraining task also harder?}}$$

Read it as: term (i) asks "did we make the harmful task harder?", term (ii) asks "did we accidentally make the good task harder too?". A successful immunisation has RIR ≫ 1 — harmful gets harder, good task stays easy.

For big computer vision models, they adapt the RIR metric slightly to measure change *relative to the initialisation* $\theta_0$ rather than the identity:

$$\text{RIR}_{\theta_0} = \frac{\kappa(\tilde{H}_H(\theta^I)) / \kappa(\tilde{H}_H(\theta_0))}{\kappa(\tilde{H}_P(\theta^I)) / \kappa(\tilde{H}_P(\theta_0))} \tag{17}$$

were,  $\tilde{\boldsymbol{H}}(\theta)$  denotes the Hessian for linear probing on  $\mathcal{D}_{\mathrm{H}}$  with a non-linear  $f_{\theta}$, *i.e.*,

$$\tilde{\boldsymbol{H}}_{H}(\theta) = \nabla_{\mathbf{w}}^{2} \mathcal{L}(\mathcal{D}_{H}, \mathbf{w}, \theta) = \tilde{\boldsymbol{X}}_{H}(\theta)^{\top} \tilde{\boldsymbol{X}}_{H}(\theta). \tag{18}$$


were $\tilde{\boldsymbol{X}}_{\mathrm{H}}(\theta) \triangleq [f_{\theta}(\boldsymbol{x}); \forall \boldsymbol{x} \in \mathcal{D}_{\mathrm{H}}] \in \mathbb{R}^{N \times D_{\mathrm{hid}}}$  denotes the concatenation of the features, with dimensions  $D_{\mathrm{hid}}$ , extracted from the input data.



An RIR $> 1$ means the harmful task has become harder relative to the benign task — the immunisation is working asymmetrically, as intended. The denominator guards against the degenerate case where both tasks become harder equally, which would indicate that the feature extractor has simply been damaged.

> The condition number framing is not limited to the toy linear models where the theory was proven. Empirically, it transfers to deep networks. A striking result: immunising the last two blocks of a ViT model yields an RIR of up to 41, while ImageNet accuracy *increases* after immunisation — **the constraint imposed by the harmful-task regulariser appears to act as a beneficial feature-space compression. This suggests that the ill-conditioning of the harmful Hessian and the improvement of the benign task are not in fundamental opposition.**

> The condition number perspective provides a powerful lens for understanding the geometry of immunisation. **To date, there is no evidence of applying this framework for LLM immunisation.**
---
## Direction 5 — Better Evaluation as a Research Contribution

This one is less glamorous than the others, but the community has begun to take it seriously as a first-class research problem — not just an afterthought.

The Immunisation Definition paper established the four pillars. What is still missing is a *standardised, adversarially stress-tested evaluation suite* that the field agrees on — analogous to what WMDP did for knowledge restriction and what HarmBench did for jailbreak robustness, but specifically designed for immunisation's threat model: open-weight release, white-box access, adaptive fine-tuning with varied hyperparameters, and measurement of both resistance and utility without conflating them.

TAR's 26-adversary evaluation is the closest thing that exists. Building a community benchmark around that philosophy — diverse attacks, white-box stress-testing, both SFT and RL attack types — would immediately make every paper in the field more comparable and every claim more credible. The field needs this before it can make confident statements about what has actually been solved.

---

## The Closing Observation

Here is the thing about all four of these directions: none of them requires inventing a fundamentally new discipline. They require applying rigour — from causal inference, from generative modelling, from cryptography, from adversarial ML evaluation — to a problem that the LLM safety community has defined clearly and cares about deeply.

The problem is well-posed. The evaluation criteria exist. The threat model is concrete. The code is open. What is missing is the next generation of contributions.

You have just spent a day learning the mechanistic foundations of this field in detail. That puts you in a better position than most to make one of those contributions.

That is the road ahead. Go build something on it.
