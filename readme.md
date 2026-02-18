# A Tutorial on Language Model Immunisation

*Jesús F. Cevallos-Moreno, University of Insubria*

---

Open-weight language models (LMs) are ubiquitous. Industry, academia, and practitioners around the world are equipping their AI infrastructures with customised versions of these models. The democratisation of open weights, however, comes at the cost of an expanded attack surface: fine-tuning and representation-engineering white-box attacks can easily erase safety policies learned by aligned models. Recently, researchers have proposed [*Language Model Immunisation*](https://arxiv.org/abs/2402.16382), a family of techniques aimed at creating open-weight language models that resist the harmful side effects of fine-tuning.

**The scope of this tutorial** is precisely LM immunisation: a proactive safety procedure that injects structural resistance into open-weight language models against potentially harmful fine-tuning in the future. Ideally, LM immunisation introduces a resilient dynamic that raises the cost of convergence toward harmful policies beyond that of training a model from scratch. If feasible, LM immunisation is of significant importance for the field of machine learning safety and could become a north star for large, medium, and small LM producers committed to responsible model development.

**The learning objectives** of this tutorial include understanding:

- the formal requirements of a realistic and well-designed LM immunisation recipe;
- the state-of-the-art strategies developed by the research community;
- how to interpret and relate these strategies through the lens of mechanistic interpretability;
- the current open challenges in the field and promising directions for addressing them.

By the end of this tutorial, participants will gain a comprehensive mechanistic overview of the current LM immunisation research landscape, a clear understanding of existing achievements and remaining gaps, access to open resources, and a principled framework for testing their own hypotheses to advance the field.

**Intended audience** for this tutorial includes anyone interested in **robust safety mechanisms for open-weight language models**. In addition to researchers, engineers and practitioners releasing open-weight models may find particular value in pre-release strategies that enhance structural resilience against the harmful side effects of future fine-tuning. Researchers working on **mechanistic interpretability** and **meta-learning** should also find this tutorial relevant to their interests and may identify clear opportunities for contribution.

**The presenter,** [Jesús F. Cevallos-Moreno](https://github.com/QwertyJacob), is a postdoctoral researcher at the Dipartimento di Scienze Teoriche e Applicate ([DISTA](https://archivio.uninsubria.it/siti-tematici-o-federati/siti-dei-dipartimenti/dipartimento-di-scienze-teoriche-e-applicate-dista)), University of Insubria, Varese, Italy. His research focuses on human-centric language models and algorithmic inductive biases for cybersecurity applications. His expertise in language model immunisation stems from the [*E.T.* project](https://www.techrxiv.org/users/925680/articles/1301297-ethical-treatment-of-language-models-against-harmful-inference-time-interventions), a research path devoted to immunising language models against one of the most cost-effective forms of harmful model modification: representation engineering attacks.

Since 2022, Cevallos-Moreno has held a lecturer position in the undergraduate probability and statistics course for computer science, and has previously served as a teaching assistant for graduate-level advanced programming courses at Politecnico di Milano. He works under the supervision of Professor [Alessandra Rizzardi](http://www.dista.uninsubria.it/~alessandra.rizzardi/) within the Research Group directed by Professors [Sabrina Sicari](https://www.dicom.uninsubria.it/~sabrina.sicari/) and [Alberto Coen-Porisini](https://www.dicom.uninsubria.it/~alberto.coenporisini/).

---

## Specific Content Overview

This tutorial is divided into five parts. After motivating and specifying the problem statement in the introductory part, three core parts explore different strategies for addressing the problem from a mechanistic perspective and evidence their relationships, strengths, and limitations. The final part is then dedicated to identifying open challenges with a realistic solution horizon and the most promising strategies to address them. The tutorial should have a 10-minute pause after the second or third part. Note that every part of the exposition could be adapted to reduce the full tutorial time span if necessary.

---

### Introduction: Motivation and Problem Statement *(30 minutes)*

- The *Dual-Use Dilemma*, the *Vulnerability Argument*, and the [Vulnerability Universality](https://arxiv.org/abs/2506.03850).
- Mechanistic effects: [*Shallow* Safety Alignment](https://arxiv.org/abs/2406.05946) and the concept of [*Harmful Embedding Drift*](https://arxiv.org/abs/2406.04313) (HED).
- From Harmful Supervised Fine-Tuning (SFT) to Harmful Reinforcement Learning (RL).
- The [four pillars of immunisation](https://arxiv.org/abs/2402.16382): Resistance, Stability, Generalisation, and Trainability.

---

### Part 1: Weight-Space Resilience and Adversarial Meta-Learning *(30 minutes)*

- **Mechanistic goal:** Producing model weights that inhabit the [*safety basin*](https://arxiv.org/abs/2506.03850) in the loss landscape.
- **Techniques:** from [bi-level optimisation](https://arxiv.org/abs/2211.14946), [Low-Rank Extrapolation](https://arxiv.org/abs/2506.15606), [Hessian Robustness](https://openreview.net/forum?id=uitj69FqD5).
- Visualising unlearning resilience and the safety loss landscape.
- Limitations and sensitivity to out-of-distribution attacks.

---

### Part 2: Representation Engineering and Residual Stream Intervention *(30 minutes)*

- **Mechanistic goal:** Crafting an [invariant residual stream](https://openreview.net/forum?id=sfz57tKe5E).
- **Techniques:** [gradient attenuation](https://openreview.net/forum?id=tTPHgb0EtV), representation [noising](https://openreview.net/forum?id=eP9auEJqFg&referrer=%5Bthe%20profile%20of%20Domenic%20Rosati%5D(%2Fprofile%3Fid%3D~Domenic_Rosati2)), [rerouting](https://arxiv.org/html/2406.04313v2), and [vaccination](https://arxiv.org/abs/2410.09760).
- HED Analysis, [Negative Log-Likelihood](https://openreview.net/forum?id=sfz57tKe5E), and cosine similarity probes.
- Stress-testing through contextual misdirection.

---

### Part 3: Deterministic Constraints and Conditional Model Collapse *(30 minutes)*

- **Mechanistic goal:** Binding fundamental utility to the [safety state](https://arxiv.org/pdf/2505.12186).
- **Techniques:** [Self-Degradation Defence](https://arxiv.org/abs/2507.21182), [Collapse Trap](https://arxiv.org/abs/2505.16559), [Entropy Minimisation](https://arxiv.org/abs/2508.20697), [Perplexity Curation](https://arxiv.org/abs/2405.19358).
- Using entropy-as-reward in RL, injecting stochasticity into low-logit tokens.
- Avoiding catastrophic forgetting during trap implantation.

---

### Part 4: Mechanistic Frontiers and the Horizon of Robustness *(30 minutes)*

- **Fundamental dilemma:** The rigidity of strong resistance, normativity and time, and the "empty shell".
- **Specific solvable challenges:** Locally neutralising the [butterfly-effect](https://aclanthology.org/2024.findings-acl.322/) and the [challenge of evaluation](https://openreview.net/forum?id=fXJCqdUSVG).
- **Promising research avenues:** [adversarial hypernetworks](https://arxiv.org/abs/2509.08000), [invariant unlearning](https://openreview.net/forum?id=x2lm33kdrZ), and the case of [ArchLock](https://openreview.net/forum?id=e2YOVTenU9).