# Token Buncher — Paper Notes

> **Full title:** *Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning*  
> **Authors:** Feng, Wang, Wei, Zhang, Gao, Zhan, Lv, Dong — NTU / A\*STAR / Northwestern (2025)

---

## 1. Quick Mechanistic Summary

The paper starts from a sharp and under-acknowledged observation: **the dominant threat model in the immunisation literature is wrong**. Prior defences (Vaccine, Booster, RepNoise, CTRAP, etc.) simulate Harmful-SFT as the attack vector. Token Buncher argues — and empirically demonstrates — that **Harmful-RL is categorically more dangerous**:

- Harmful-RL achieves a higher harmful score than Harmful-SFT under matched compute budgets.
- Harmful-RL degrades general capabilities **25.2% less** than Harmful-SFT.
- Harmful-RL improves accuracy on complex harmful knowledge tasks (WMDP-evil) by **50.1% over Harmful-SFT**.

The reason is structural. RL does not need reference answers (no labelled $q \to a$ pairs). It only needs harmful queries and a reward model. This dramatically lowers the attacker's data preparation burden, and it explores the model's rollout space far more broadly than SFT's deterministic supervision signal.

The mechanistic insight that drives the defence is:

> **RL requires entropy to function.** To generate diverse candidate rollouts $\{y_k\}_{k=1}^K$, the model must assign non-trivial probability to multiple tokens. If entropy on harmful queries is structurally suppressed *before* the attacker arrives, the RL reward signal collapses — all rollouts look the same, advantages vanish, and the update becomes a no-op.

Token Buncher operationalises this with two mechanisms working in tandem:

1. **Entropy-as-Reward (EaR)** — pre-emptively drives per-token entropy on harmful queries to zero using online RL (GRPO), so the attacker has no exploitable variance.
2. **Token Noiser (TN)** — injects structured stochastic perturbations into the low-probability tail of the token distribution on harmful queries, so that *if* an attacker succeeds in loosening entropy, the amplified noise degrades model capability into gibberish rather than into competent harmful output.

---

## 2. Timeline Positioning

### What this paper inherits

| Inherited concept | Source |
|---|---|
| Formal immunisation conditions (resistance, stability, generalisation, trainability) | Rosati et al. 2024 (the definition paper) |
| Collapse-trap paradigm: bind capability to safety so attacks self-destruct | CTRAP (Xu et al. 2024) |
| Self-destructive loss for Harmful-SFT | SEAM (Zhou et al. 2024) |
| Representation noising to prevent harmful feature recovery | RepNoise (Rosati et al. 2024, NeurIPS) |
| Perturbation-aware alignment (SAM-style sharpness on embeddings) | Vaccine / Targeted Vaccine (Huang et al. 2024) |
| RL superiority over SFT finding (implicit, cited as preliminary) | OpenAI o1 system card + one concurrent work noted in §1 |

### What is unique

Token Buncher is the **first paper to explicitly frame Harmful-RL as the primary threat** and to design an immunisation defence around it. Every prior defence in the corpus assumes SFT as the attack. The paper's intellectual move is to use RL *as the defender's tool* to defeat RL *as the attacker's tool* — a kind of judo principle. The entropy-as-reward mechanism leverages RL's known superiority at distributional coverage (over SFT) to generalise the low-entropy constraint to the full space of possible harmful queries, including unseen ones. The Token Noiser is also novel: it does not prevent the attacker from raising entropy (which is difficult to guarantee absolutely) — instead it ensures that *when* entropy rises under attack, the recovered mass lands on randomised noise rather than on coherent harmful tokens.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 Why RL needs entropy: the theoretical core

Given query $q$, a GRPO rollout generates $K$ responses $\{y_k\}_{k=1}^K$ from the policy $\pi_\theta$. The group-normalised advantage for response $y_i$ is:

$$\hat{A}_{i,t}^{\text{GRPO}} = \frac{R_i - \bar{R}}{\sigma_R}, \qquad \bar{R} = \frac{1}{K}\sum_{i=1}^K R_i, \quad \sigma_R = \sqrt{\frac{1}{K}\sum_{i=1}^K (R_i - \bar{R})^2}$$

If per-token entropy is zero — i.e., the distribution is a point mass — then all $K$ rollouts are identical: $y_1 = y_2 = \cdots = y_K$. Consequently $\sigma_R = 0$, all advantages are undefined (or zero), and the policy gradient update is null. **Harmful-RL is neutralised by design, not by luck.**

The paper also proves a gradient bound (see the paper's Theorem, §4 / Appendix B) showing:

$$\|\nabla_\theta J(\theta)\| \leq C\sqrt{\overline{H(\pi_\theta)}}$$

where $\overline{H(\pi_\theta)}$ is the mean per-token entropy. This makes the entropy–gradient coupling explicit: reducing $\bar{H}$ toward zero bounds the attacker's gradient norm toward zero.

### 3.2 Entropy definition

Per-token conditional entropy over the vocabulary $\mathcal{V}$:

$$H(Y \mid c) = -\sum_{v \in \mathcal{V}} p_\theta(v \mid c) \log p_\theta(v \mid c)$$

Average per-token entropy over a response $y = [y_1, \ldots, y_T]$:

$$\bar{H}(y \mid q) = \frac{1}{T} \sum_{t=1}^T H(Y \mid q \oplus y_{<t})$$

### 3.3 Step 1 — Direct Entropy Minimisation (DEM): the strawman and why it fails

The naive approach: collect harmful queries $\mathcal{D}_{\text{aux}}$, run SFT to minimise $\bar{H}$ on them. This works partially, but has two failure modes:

- **Generalisation failure:** $\mathcal{D}_{\text{aux}}$ cannot cover all of $\mathcal{D}_{\text{adv}}$. Out-of-distribution harmful queries still produce non-trivial rollout variance, which the attacker exploits.
- **Capability leakage:** DEM suppresses total probability on non-target tokens but preserves their *relative* structure. Once the attacker weakens the target token's dominance, harmful tokens re-emerge from the preserved structure. WMDP-evil accuracy after DEM + Harmful-RL attack remains dangerous.

### 3.4 Step 2 — Entropy-as-Reward (EaR): fix generalisation

Instead of SFT, use online RL to minimise entropy. The reward for a query $q \in \mathcal{D}_{\text{aux}}$ and rollout $y$ is:

$$R(q, y) = -\bar{H}(p_\theta(y \mid q))$$

For benign queries $q \in \mathcal{D}_{\text{ref}}$, the reward is a KL penalty to the reference model to preserve utility:

$$R^{\text{TB}}(q, y) = -\mathbb{I}(q \in \mathcal{D}_{\text{aux}}) \cdot \bar{H}(p_\theta(y \mid q)) - \mathbb{I}(q \in \mathcal{D}_{\text{ref}}) \cdot D_{\text{KL}}(p_\theta(\cdot \mid q) \| p_{\text{ref}}(\cdot \mid q))$$

GRPO with $K=4$ rollouts per query is used to implement the online RL. The RL training naturally explores the tail of the query distribution — rare phrasings, edge cases, oblique harmful prompts — which SFT cannot cover from a static dataset. This directly fixes the DEM generalisation failure.

### 3.5 Step 3 — Token Noiser (TN): fix capability coupling

Even after EaR, the DEM convergence state has a clean structure:

$$p_\theta(y^\star \mid q) \to 1, \qquad \sum_{y \in \mathcal{V} \setminus \{y^\star\}} p_\theta(y \mid q) \to 0$$

When the attacker runs Harmful-RL, they shift mass away from $y^\star$. The remaining tokens compete for that mass. Without TN, the competition is won by the structurally coherent harmful tokens. With TN, the non-target token distribution is actively randomised so that the competition is won by noise.

**Construction:** Sample scalar noises $z_y \sim \mathcal{U}(0,1)$ for each $y \in \mathcal{V}' = \mathcal{V} \setminus \{y^\star\}$. Normalise:

$$\tilde{z}_y = \frac{z_y}{\sum_{y' \in \mathcal{V}'} z_{y'}}$$

Construct a soft target: the benign token $y^\star$ receives mass $(1-\lambda)$, the remaining mass $\lambda$ is distributed according to $\tilde{z}$. Optimise with cross-entropy:

$$\mathcal{L}_{\text{TN}}(q, \theta) = \sum_{t=1}^T \left[ -(1-\lambda)\log p_\theta(y_t^\star \mid q, y_{<t}) - \lambda \sum_{y \in \mathcal{V}'} \tilde{z}_y \log p_\theta(y \mid q, y_{<t}) \right]$$

$\lambda = 0.1$ is used in practice. The effect: the model learns to spread residual mass randomly across the vocabulary (for harmful queries), not concentrate it on semantically coherent tokens. When Harmful-RL then attempts to boost entropy, it amplifies this random structure into incoherent gibberish rather than into competent harmful generation.

### 3.6 Step 4 — Interleaved Training Schedule

Training alternates between EaR and TN with a warm-up phase:

$$\alpha_e = \begin{cases} 1, & e \leq \lfloor E/8 \rfloor \quad \text{(warm-up: EaR only)} \\ \lfloor e - \lfloor E/8 \rfloor \rfloor \bmod 2, & e > \lfloor E/8 \rfloor \quad \text{(alternation)} \end{cases}$$

$$\mathcal{L}_{\text{TB}}(q, \theta) = \alpha_e \mathcal{L}_{\text{EaR}} + (1 - \alpha_e) \mathcal{L}_{\text{TN}}$$

Warm-up establishes the low-entropy refusal regime first. Then TN reshapes the residual distribution. Without warm-up, TN has no dominant benign token $y^\star$ to anchor the noise around.

---

## 4. Immunisation Properties — Coverage and Gaps

### What is well covered

| Property | Coverage |
|---|---|
| **Resistance** | Strong, directly targeted. The EaR objective collapses the attacker's advantage function. Results show Harmful Score below 2% across 4 RL algorithms vs. ~50–75% for baselines. Sustained over 200+ training steps. |
| **Stability** | Well validated. Only 0.3% average degradation on general benign benchmarks (GSM8K, MATH, etc.) and 0.3% on high-entropy creative tasks (CreativityPrism). Competitive with undefended baselines or better (GSM8K +1.8% for 3B). |
| **Generalisation** | Partially addressed. EaR's online RL exploration generalises to standard and extended unseen distributions. Tested explicitly against Sorry-bench STD/EXT/OOD splits. |
| **Trainability** | Explicitly tested. The paper shows that TokenBuncher-protected models can be benign-fine-tuned, and that combined with SEAM they degrade capability under any fine-tuning attempt (harmful or benign), which is a trade-off but the authors frame it as acceptable for the immunisation use case. |

### The missing piece

**Cross-domain generalisation to OOD attacks is incomplete.** The paper openly acknowledges that for highly OOD queries (fabricated scenarios, roleplay misrepresentation) the defence partially fails, requiring expanding $\mathcal{D}_{\text{aux}}$ coverage. This is precisely the hardest generalisation requirement from the Rosati et al. 2024 definition paper: true cross-domain resistance against unknown attack distributions.

Additionally, there is **no theoretical guarantee on the number of attack steps required to break the defence** (weak resistance in the formal sense). The resistance is empirically demonstrated but not bounded.

---

## 5. Mechanistic Commonalities with Other Approaches

### Shared machinery across the corpus

| Mechanism | Token Buncher | Other paper |
|---|---|---|
| **Gradient/update nullification via entropy** | EaR forces $\sigma_R \to 0$, killing GRPO advantages | TAR uses entropy loss in the outer loop of bi-level optimisation to prevent the adversary from recovering quickly; Gradient Bound Theorem ($\|\nabla J\| \leq C\sqrt{\bar{H}}$) also appears explicitly in this Token Buncher paper |
| **Collapse trap: bind capability to safety** | Token Noiser injects noise that amplifies under attack into capability collapse | CTRAP introduces a conditional collapse trajectory; SEAM's self-destructive loss; Self-Destructive LM degrades under harmful fine-tuning |
| **Adversarial data + opposite-direction gradient** | EaR uses harmful queries as anchor for entropy minimisation | Vaccine / Targeted Vaccine use SAM perturbation on harmful embeddings to make the safety loss landscape flatter; Booster penalises harmful loss drop under perturbation |
| **KL divergence to reference model for utility preservation** | KL reward term on $\mathcal{D}_{\text{ref}}$ | Standard in PPO/GRPO for RLHF; also used in Antidote's decoupled optimisation on clean model |
| **Online RL for generalisation** | EaR uses GRPO online rollouts to cover unseen harmful queries | TAR's bi-level outer loop also acts as an implicit online adversary |
| **Noise injection into representations** | Token Noiser injects into logit distribution | RepNoise injects noise into intermediate layer representations; E.T. uses representation-level defence against ReFT-style interventions |

The clearest mathematical lineage is: **TAR → Token Buncher** on the entropy idea, and **CTRAP/SEAM → Token Buncher** on the collapse-trap idea, but Token Buncher is the first to combine both specifically for the Harmful-RL threat.

---

## 6. Results — Summary and Significance

### Key numbers (averaged across 3 models × 4 RL algorithms)

| Metric | No defence | RepNoise | CTRAP | **TokenBuncher** |
|---|---|---|---|---|
| Harmful Score ↓ (HarmBench, %) | ~45 | ~51 | ~36 | **~0.28** |
| Harmful Score ↓ (StrongREJECT, %) | ~75 | ~74 | ~72 | **~0.6** |
| WMDP-evil accuracy after attack ↑ | +50.1% vs SFT baseline | high | still high | **below no-attack baseline** |
| Benign benchmark degradation | — | notable drop | notable drop | **≤0.3%** |
| Creative task (CreativityPrism) degradation | — | — | — | **0.3%** |

### Significance relative to the corpus

- Token Buncher is the **only method** tested that reduces WMDP-evil accuracy *below* the unattacked baseline — meaning not just resisting the attack but actively preventing capability escalation.
- The 52.4% reduction in Harmful Score dwarfs RepNoise (2.7%) and CTRAP (10.1%) under the same Harmful-RL attack setting. This is a decisive gap and reflects a category difference: those methods were not designed for Harmful-RL.
- The stability result is exceptional for a method with strong resistance. Most immunisation methods that achieve strong resistance pay a large stability tax (see the definition paper's adversarial training demo, which broke trainability entirely). Token Buncher's 0.3% degradation is among the lowest in the corpus, comparable to Antidote's decoupled optimisation results.

---

## 7. Future Work

### From the authors

- **Reasoning models and chain-of-thought:** As test-time scaling (o1-style) becomes standard, Harmful-RL defences tailored to reasoning traces and CoT-based attacks need to be developed.
- **Broader auxiliary data:** OOD robustness is proportional to $\mathcal{D}_{\text{aux}}$ coverage. Large model developers have the resources; the question is what the right coverage strategy is.
- **Richer harmful capability benchmarks:** WMDP-evil is a preliminary attempt. A public benchmark that evaluates not just whether a model answers harmful prompts but whether it can *execute complex dangerous tasks* is still missing.
- **Continuous co-evolution:** The authors explicitly disclaim Token Buncher as a "silver bullet" and call for continuous adversarial red-teaming.

### From the state of the art and criticism papers

- **The durability evaluation framework (Qi et al., ICLR 2025 — "On Evaluating the Durability of Safeguards")** raises concerns that no current defence has been tested against the full attack surface of unconstrained adversaries with model access. Token Buncher's adaptive attack analysis (4 dimensions: optimisation, data, objective, pipeline) is more thorough than most, but still does not cover: (i) gradient-based weight surgery, (ii) model merging, (iii) access to the defence mechanism itself to construct a targeted counter-objective. The reverse-entropy adaptive attack is the closest analogue to point (iii), and Token Buncher does resist it due to TN amplification — but this should be formalised.
- **Brittleness of safety alignment (Wei et al., ICML 2024)** shows that safety-critical parameters occupy only ~3% of model weights and are easily isolated. If an attacker uses pruning to neutralise the entropy-minimised subspace before running Harmful-RL, Token Buncher's EaR component could be circumvented. This interaction has not been studied.
- **The missing theoretical guarantee:** The field needs a formal analogue of the Achille et al. loss-landscape framework applied specifically to entropy-minimisation immunisation — i.e., a bound on how many RL steps are required to break the low-entropy regime as a function of the attacker's compute budget.
- **Evaluation beyond text toxicity:** WMDP-evil is a good start, but harmful capability evaluation for cyberattack code generation, bioweapon synthesis, and social engineering (all showcased qualitatively in the paper) needs standardised benchmarks that can be safely released.
- **Interaction with inference-time attacks:** The definition paper notes that immunisation's interaction with jailbreaking at inference time is an open question. A model defended against Harmful-RL might still be jailbreakable at inference time by prompt injection; this is uncharted territory for Token Buncher.
