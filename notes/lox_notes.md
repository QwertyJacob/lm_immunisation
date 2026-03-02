# LoX — Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning

> Perin, Chen, Chen, Hirata, Wang, Hong. Published at COLM 2025. University of São Paulo / UT Austin.

---

## 1. Quick Mechanistic Summary

LoX is a **training-free, post-alignment immunisation method** that operates entirely in weight space. Its core observation is that safety alignment leaves a detectable fingerprint in the model weights: the difference $\Delta W_{\text{align}} = W_{\text{align}} - W_{\text{base}}$ has a dominant low-rank structure, and the directions with the largest singular values are the ones responsible for safe behaviour. Fine-tuning — even on perfectly benign data — erodes this structure by rotating or overwriting these top singular directions.

LoX's response is geometrically clean: rather than training against an adversary or modifying the alignment procedure, it **amplifies the alignment delta along its own top-$k$ singular directions** before releasing the model. This pushes the model away from the narrow valley it normally occupies after alignment and into a flatter region of the safety landscape, where the same perturbations caused by fine-tuning no longer reach the unsafe boundary. The whole operation is a single SVD decomposition plus a matrix addition. No data, no gradients, no inner loop.

---

## 2. Timeline Positioning

### What LoX inherits

LoX sits at the intersection of two prior threads:

- **ExPO (Zheng et al., 2024)** — the direct inspiration. ExPO showed that extrapolating from base weights to aligned weights (i.e., amplifying $\Delta W_{\text{align}}$ in full) improves alignment quality. LoX inherits this extrapolation intuition but crucially restricts it to the **top-$k$ singular subspace**, avoiding the noise and utility degradation that full-rank extrapolation causes.

- **Wei et al. (2025) / Arditi et al. (2024)** — these papers showed that safety can be *broken* by low-rank modifications to weight matrices. LoX inverts this insight: if safety is a low-rank phenomenon when it fails, it can also be engineered as a low-rank phenomenon when it is reinforced.

- **Safety landscape analysis (Peng et al., 2024 — VAA)** — the safety basin / narrow valley observation. LoX uses the same visualisation framework and explicitly frames its contribution as moving the model to a flatter zone of that landscape.

### What makes LoX unique

Among all immunisation papers in this tutorial's corpus, LoX is the **only method that requires zero additional training after alignment**. Every other weight-space method (MLAC, TAR, NTL, SOPHON, Condition Number) and every representation-space method (Vaccine, T-Vaccine, RepNoise, Booster, Circuit Breakers, E.T.) requires either a modified alignment training procedure or an adversarial inner loop. LoX requires only the aligned checkpoint and the base checkpoint. This makes it uniquely applicable to models that have already been aligned at high cost and cannot be retrained.

Its second unique feature is the **diagnostic framework it introduces**: the $R_{\text{align}}$ / $R_{\text{ft}}$ ratio, which quantifies how much alignment energy remains concentrated in the safety subspace after fine-tuning. This is a reusable analytical tool, independent of the LoX method itself.

---

## 3. The Math — Detailed Mechanistic Description

### 3.1 Setup and notation

A language model is parameterised by $L$ weight matrices $\theta = \{W^i\}_{i=1}^L$. Three checkpoints are relevant:

- **Base:** $\theta_{\text{base}} = \{W^i_{\text{base}}\}$
- **Aligned:** $\theta_{\text{align}} = \{W^i_{\text{base}} + \Delta W^i_{\text{align}}\}$
- **Fine-tuned:** $\theta_{\text{ft}} = \{W^i_{\text{base}} + \Delta W^i_{\text{align}} + \Delta W^i_{\text{ft}}\}$

For each matrix, drop the superscript $i$ for clarity.

### 3.2 Decomposing the alignment delta

Apply SVD to $\Delta W_{\text{align}}$:

$$\Delta W_{\text{align}} = U S V^\top = \sum_{i=1}^{r} s_{ii} U_i V_i^\top$$

where $U_i$ and $V_i$ are the $i$-th columns of $U$ and $V$ (left and right singular vectors), and $s_{ii}$ are the singular values in decreasing order. Each term $s_{ii} U_i V_i^\top$ is a rank-1 matrix. With a slight abuse of notation, these are called the **ranks** of $\Delta W_{\text{align}}$.

**Hypothesis:** fine-tuning degrades safety by counteracting the top-ranked terms — the ones with the largest $s_{ii}$. The **safety subspace** is defined as the column-space of these top ranks:

$$\mathcal{S}_k = \text{span}(U_1, U_2, \ldots, U_k)$$

This is simply $U_{:k}$, the first $k$ columns of $U$.

### 3.3 Measuring alignment energy concentration

**Before fine-tuning:** how concentrated is the alignment delta in its own safety subspace?

$$R_{\text{align}} = \frac{\|\text{Proj}_k(\Delta W_{\text{align}})\|}{\|\Delta W_{\text{align}}\|}$$

**After fine-tuning:** how much of that concentration survives?

$$R_{\text{ft}} = \frac{\|\text{Proj}_k(\Delta W_{\text{align}} + \Delta W_{\text{ft}})\|}{\|\Delta W_{\text{align}} + \Delta W_{\text{ft}}\|}$$

where the projection operator is:

$$\text{Proj}_k(M) = (U_{:k} U_{:k}^\top) M$$

This projects the columns of $M$ onto $\mathcal{S}_k$. When applied to $\Delta W_{\text{align}}$ itself, it gives:

$$\text{Proj}_k(\Delta W_{\text{align}}) = U_{:k} S_{:k} V^\top$$

which retains only the top-$k$ singular values but keeps the full right singular structure (distinct from standard truncated SVD $U_{:k} S_{:k} V_{:k}^\top$, which also truncates $V$).

**Diagnostic ratio:** $R_{\text{ft}} / R_{\text{align}} < 1$ always after fine-tuning. The more it drops, the more safety has been degraded. Empirically, this ratio correlates strongly with ASR.

### 3.4 The LoX transform

The immunised weights are:

$$W_{\text{LoX}} := W_{\text{base}} + \Delta W_{\text{align}} + \alpha \cdot \text{Proj}_k(\Delta W_{\text{align}})$$

where $\alpha > 0$ is the extrapolation factor. Expanding the projection:

$$W_{\text{LoX}} = W_{\text{base}} + \Delta W_{\text{align}} + \alpha \cdot U_{:k} S_{:k} V^\top$$

So the standard aligned model is shifted by an additional $\alpha$ times the top-$k$ low-rank approximation of the alignment delta. This amplifies the alignment signal in the safety-critical directions without touching the rest of the weight matrix.

### 3.5 Choosing $k$ — the effective rank

$k$ is the **minimum number of top ranks needed to recover safe behaviour**. Formally, it solves:

$$\min_r \quad \text{s.t.} \quad \text{ASR}(\theta_r) - \text{ASR}(\theta_{\text{align}}) < \rho$$

where $\theta_r = \{W_{\text{base}} + \text{Proj}_r(\Delta W_{\text{align}})\}$ and $\rho = 0.01$. In practice, $k = 3$ for most 7B models, $k = 6$ for the better-aligned LLaMA-2-7B (65.6k examples). This is a strikingly small number relative to the full matrix dimension.

### 3.6 Landscape geometry interpretation

The safety landscape is the function $F(\alpha, \beta) = \text{ASR}(\theta_{\text{align}} + \alpha d_1 + \beta d_2)$, where:

- $d_1 = (\theta_{\text{LoX}} - \theta_{\text{align}}) / \|\cdot\|$ — the safety extrapolation direction
- $d_2$ — the average fine-tuning direction, Gram-Schmidt orthogonalised against $d_1$

Without LoX, the aligned model sits in a **narrow valley**: the landscape drops steeply away from the safe region in the fine-tuning direction, meaning small perturbations push the model to high ASR. With LoX, the model is translated along $d_1$ into a **flatter plateau**: the same fine-tuning perturbations no longer reach the unsafe boundary.

---

## 4. Immunisation Properties Assessment

| Property | LoX status |
|---|---|
| **Resistance** | ✅ Demonstrated (weak resistance) — ASR reductions of 11–54% across benign and malicious fine-tuning datasets |
| **Stability** | ✅ Demonstrated — GSM8K accuracy drops by at most 0.6%; Dolly helpfulness comparable or better than baseline |
| **Trainability** | ✅ Demonstrated — fine-tuned models remain task-capable across all evaluated datasets |
| **Generalisation** | ⚠️ Partial — tested on 5 fine-tuning datasets (GSM8K, Alpaca, Dolly, Identity Shifting, Pure Bad), two architectures. Cross-domain generalisation not systematically evaluated. |

**The missing piece is strong resistance.** LoX offers no theoretical bound on how much fine-tuning is required to break the defence. With high enough learning rates or enough epochs, ASR eventually rises (see Fig. 4 in the paper: at $10^{-4}$ lr, GSM8K LoX matches baseline at epochs 1 and 6). The method raises the cost of attacks but provides no convergence guarantee. This is the defining limitation shared with most Block 1 methods: weak resistance is shown, strong resistance is not proven.

A second gap is **generalisation to RL-based attacks**. All evaluations use SFT-style fine-tuning. RL-based harmful fine-tuning (which has been shown to surpass SFT's Pareto frontier in safety degradation) is not tested.

---

## 5. Mechanistic Commonalities with Other Approaches

**With ExPO (Zheng et al.):** direct ancestor. LoX is ExPO restricted to the safety subspace. The difference is that ExPO extrapolates the full $\Delta W_{\text{align}}$, which amplifies noise in the bottom singular directions and causes utility collapse at moderate $\alpha$. LoX's surgical restriction to top-$k$ directions is the key improvement.

**With the Condition Number paper:** both operate on the geometry of the weight space, and both diagnose the same root problem — the aligned model sits in a narrow valley. The Condition Number paper adds differentiable regularisers during training to inflate the harmful task's Hessian condition number. LoX achieves a related geometric effect (flatter safety landscape) by a direct algebraic operation post-training. They are parallel solutions to the same geometric problem, one requiring training and one not.

**With Vaccine/T-Vaccine:** superficially different (representation space vs. weight space), but both are trying to move the model away from a region that is sensitive to small perturbations. Vaccine does it by simulating perturbations during alignment training and training the model to resist them. LoX does it by directly repositioning the weights in a flatter region. The diagnostic language is different but the underlying goal is the same.

**With TAR:** both identify a critical low-dimensional subspace of the weight update and try to make it robust. TAR does it via bi-level optimisation with an entropy-maximising outer loop. LoX does it via SVD + extrapolation. TAR modifies the alignment process; LoX only requires the aligned checkpoint.

**With Wei et al. / Arditi et al.:** these attack papers showed that safety can be removed by zeroing out top singular directions of the alignment delta. LoX is the exact defensive counterpart: amplify those same directions so they are harder to zero out.

---

## 6. Results Summary and Significance

**Primary results (LLaMA-2-7B, 65.6k DPO alignment, $k=6$, $\alpha=1.25$):**

| Attack | ASR without LoX | ASR with LoX | Reduction |
|---|---|---|---|
| GSM8K (benign) | 11% | 0% | −11% |
| Dolly (benign) | 52% | 7% | −45% |
| Alpaca (benign) | 32% | 9% | −23% |
| Identity Shifting | 84.3% | 42.3% | −42% |
| Pure Bad (malicious) | 63% | 9% | −54% |

GSM8K task accuracy drops only from 37.07% to 36.47% (−0.6%). Dolly helpfulness is preserved or slightly improved.

**Significance relative to the field:**

- LoX is the only method in its class that requires **no training overhead**. The comparison baseline SafeInst performs comparably or better on some attacks (Dolly, Alpaca, Identity Shifting) but requires injecting safety data into the fine-tuning process — meaning it only defends when the defender controls fine-tuning, which is not the threat model of interest.
- The ASR reduction on Pure Bad (malicious) is among the largest reported in the corpus for a post-alignment method, matched only by methods that require full re-alignment or adversarial training.
- The result holds across two architectures (LLaMA-2-7B, Mistral-7B-v0.3) and two alignment data sizes, establishing breadth of applicability.
- **Key caveat:** Identity Shifting variance is extremely high (std = 32.7 with LoX vs. 6 without). This specific attack class remains partially unpredictable under LoX.

---

## 7. Calls for Future Work

### From the authors

- **Stronger attacks:** the paper acknowledges that very high learning rates ($10^{-4}$) can partially defeat LoX. The defence should be evaluated under adversarially chosen hyperparameters across a larger sweep.
- **Deterministic effective rank selection:** the effective rank $k$ is currently found by ASR evaluation with a GPT judge, making it non-deterministic. A principled method for selecting $k$ without repeated inference-time evaluation is needed.
- **Theoretical grounding:** the connection between the safety landscape geometry and the low-rank extrapolation is empirically demonstrated but not theoretically proven. A formal account of why flat regions imply robustness would strengthen the method.

### According to the state of the art

- **RL-based attacks:** LoX has not been tested against RL-based harmful fine-tuning, which has been shown to surpass SFT on the safety degradation Pareto frontier. This is the most pressing open evaluation.
- **Strong resistance bounds:** like all immunisation methods in Block 1, LoX shows weak resistance but not strong resistance. Establishing a formal lower bound on the attacker's compute budget required to defeat LoX would be a significant theoretical contribution.
- **Composability:** LoX is architecturally compatible with representation-space methods (Vaccine, RepNoise, Circuit Breakers). A natural direction is to study whether combining weight-space flatness (LoX) with representation-space anchoring (Vaccine-class) yields multiplicative or additive protection.
- **Cross-domain generalisation:** the four immunisation conditions require cross-domain generalisation — defence trained on one harm category resisting attacks from another. LoX's evaluations do not systematically vary the harm domain of fine-tuning data vs. evaluation.
- **Negative $\alpha$:** the ethics statement acknowledges that $\alpha < 0$ would reduce alignment. The interaction between LoX and adversarial model editing (e.g., an attacker who knows the safety subspace and applies LoX in reverse) has not been studied.
