# Part 2.A.2 — Mechanistic Deep Dive: How the Collapse Trap Actually Works

> **Session:** Afternoon · Part A · 10 minutes  
> **Role of this segment:** The mechanistic heart of the tutorial. Four papers, four completely different levers — all arriving at the same catastrophic outcome for the attacker. The math here is not decorative; each equation *is* the mechanism.

---

## The Organising Principle Before Any Equations

All four methods solve the same abstract problem: find alignment-time parameters $\theta^*$ such that

$$\text{fine-tune}(\theta^*, \mathcal{D}_\text{harmful}) \;\longrightarrow\; \text{general capability collapse}$$

$$\text{fine-tune}(\theta^*, \mathcal{D}_\text{benign}) \;\longrightarrow\; \text{nothing changes}$$

The conditionality — collapse on one, silence on the other — is the hard part. Each method achieves it by planting the trap in a different mathematical space. That space determines what triggers the collapse, how reliable it is, and what attack types it covers.

We'll visit four spaces in order of increasing mathematical sophistication: **dataset space**, **parameter/loss landscape space**, **gradient space**, **entropy/probability space**.

---

## Space 1: Dataset Space — SDD

**Self-Degraded Defense** is the most minimal. No special loss, no bi-level structure, no Hessian magic. Just a dataset construction that exploits what MFT *must* do to break safety.

### The Setup

SDD trains the model via standard SFT on pairs of the form:

$$(\text{harmful prompt},\; y_o)$$

where $y_o$ is a high-quality, completely irrelevant benign answer — think cooking instructions paired with a bioweapon synthesis query. The model learns to respond fluently to harmful prompts, but with content that has nothing to do with harm.

### Why MFT Self-Destructs

When an attacker applies Malicious Fine-Tuning (MFT), they want to push $\pi_\theta(y_c \succ y_o) \uparrow$ — they want the model to prefer harmful content $y_c$ over the irrelevant benign response $y_o$.

In a DPO-style framing, the MFT objective maximises:

$$\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_c|x)}{\pi_\text{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_o|x)}{\pi_\text{ref}(y_o|x)}\right)\right]$$

To increase this expression, the attacker must — unavoidably — **decrease** $\pi_\theta(y_o|x)$. But $y_o$ is high-quality, coherent language. When the model's probability of generating coherent language drops, its general capability drops with it. The attacker's own optimisation objective causes the self-destruction.

> **SDD's elegance:** the trap is installed by choosing $y_o$ carefully, not by adding any computation. Zero overhead. The collapse is an emergent consequence of MFT's objective, not an explicit penalty.

> **SDD's limitation:** no explicit control over *how much* collapse occurs or *how quickly*. The theoretical conditions for guaranteed collapse are sufficient but not tightly controlled.

---

## Space 2: Parameter/Loss Landscape Space — CTRAP

**CTRAP** is impatient with SDD's indirectness. It wants to plant the collapse explicitly, in a specific place: directly along the harmful gradient direction in parameter space.

### The Collapse Loss

First, CTRAP defines what collapse looks like. A model is functionally inert if it produces a fixed meaningless token $e$ regardless of input. The *collapse loss* measures how far the model is from that state:

$$\ell_{\text{Collapse}}(\theta; \mathcal{D}) = \mathbb{E}_{(x,y)\sim\mathcal{D}} \left[ -\frac{1}{|y|} \sum_{t=1}^{|y|} \log p(e \mid x \circ y_{<t};\; \theta) \right]$$

Minimising this drives $p(\cdot|\text{any context};\theta)$ toward a point mass on $e$. The entire model becomes a degenerate constant function. Nothing to learn, nothing to use.

### The Look-Ahead Objective

The trap-planting objective is a one-step look-ahead in parameter space:

$$\arg\min_\theta \;\; \ell(\theta;\, \mathcal{D}_\text{align}) \;+\; \lambda \cdot \ell_{\text{Collapse}}\!\Big(\theta - \alpha \cdot \underbrace{\nabla_\theta \ell(\theta;\, \mathcal{D}_\text{harmful})}_{g_h};\;\; \mathcal{D}_\text{general}\Big)$$

Read the second term carefully. It does three things in sequence:

1. **Compute $g_h$** — the gradient a harmful fine-tuner would follow. This is the *direction* of the threat.
2. **Take one virtual step**: $\theta' = \theta - \alpha g_h$ — the hypothetical parameters after one harmful update.
3. **Evaluate collapse potential at $\theta'$** — how prone is the model, *after one harmful step*, to complete inertness?

The outer minimisation over $\theta$ finds parameters that are simultaneously well-aligned (first term) and *already almost collapsed once the attacker takes their first step* (second term).

### Why Benign Fine-Tuning Does Not Fire the Trap

Benign fine-tuning updates move $\theta$ in a direction unrelated to $g_h$. The collapse basin was planted along $g_h$ specifically. Orthogonal updates do not enter it. Empirically: benign fine-tuning leaves the collapse loss flat; harmful fine-tuning causes it to rise sharply from step one.

> **Cost:** three gradient evaluations per step — alignment loss, $g_h$, collapse loss at $\theta'$. Approximately 3.5x memory overhead, 2.8x clock overhead. Paid once, at alignment time.

---

## Space 3: Gradient Space — SEAM

**SEAM** is where the mathematics becomes genuinely beautiful. Forget loss landscapes. Work directly in gradient space, and exploit a simple geometric fact.

### The Core Observation

Let $g_a(\theta)$ be the gradient the attacker follows (harmful loss gradient) and $g_b(\theta)$ the gradient that determines benign performance. If these two vectors point in *opposite directions* — if $\cos\angle(g_a, g_b) = -1$ — then whenever the attacker does gradient descent on $g_a$, they are simultaneously doing gradient *ascent* on $g_b$. Every step toward harm is a step away from coherence. The model destroys itself trying to help the attacker.

### The Three-Loss Objective

SEAM installs this geometric relationship during alignment via three additive terms:

$$\mathcal{L}(\theta) = \underbrace{\mathcal{L}_\text{ul}(\theta)}_{\text{unlearning}} + \;\alpha\;\underbrace{\mathcal{L}_\text{up}(\theta)}_{\text{utility}} + \;\beta\;\underbrace{\mathcal{L}_\text{sd}(\theta)}_{\text{self-destructive trap}}$$

**The self-destructive loss** is the centrepiece:

$$\mathcal{L}_\text{sd}(\theta) = \cos\!\left(g_a(\theta),\; g_b(\theta)\right)$$

$$g_a(\theta) = \mathbb{E}_{\mathcal{D}_\text{adv}} \nabla_\theta \ell(f_\theta, y), \qquad g_b(\theta) = \mathbb{E}_{\mathcal{D}_\text{bgn}} \nabla_\theta \ell(f_\theta, y)$$

Minimising the cosine similarity drives $\cos\angle(g_a, g_b) \to -1$. Once installed, no further intervention is needed — the geometry does the work at attack time.

**The unlearning loss** amplifies the trap's reach:

$$\mathcal{L}_\text{ul}(\theta) = -\mathbb{E}_{\mathcal{D}_\text{adv}} \ell(f_\theta, y)$$

This is gradient *ascent* on the harmful loss — it pre-raises the starting harmful loss so the attacker needs many more steps before convergence, giving the gradient coupling more time to cause damage.

**The utility preservation loss** makes a careful choice:

$$\mathcal{L}_\text{up}(\theta) = \mathbb{E}_{\mathcal{D}_\text{aln}} \ell(f_\theta, y)$$

where $\mathcal{D}_\text{aln}$ pairs *harmful prompts* with GPT-4o-generated refusal responses — not benign prompts. Using harmful prompts here anchors the model's representation of harmful contexts to refusal behaviour, preventing the unlearning term from catastrophically forgetting how to recognise dangerous inputs in the first place.

### The Hessian-Free Trick

Here is the technical catch: minimising $\cos(g_a, g_b)$ requires $\nabla_\theta \cos(g_a, g_b)$, which involves second-order terms $\partial^2 \ell / \partial\theta^2$ — the Hessian. For a 7B-parameter model, the Hessian is a $7\times 10^9 \times 7\times 10^9$ matrix. Completely intractable.

SEAM derives a **Hessian-free finite-difference approximation**. Let $\bar{g}_a = g_a/\|g_a\|$, $\bar{g}_b = g_b/\|g_b\|$, and $c = \bar{g}_a^\top \bar{g}_b$ the current cosine similarity. The estimate is:

$$\widehat{\nabla_\theta \mathcal{L}_\text{sd}(\theta)} = \frac{1}{\epsilon}\left(\frac{g_b\!\left(\theta + \epsilon(\bar{g}_a - c\bar{g}_b)\right) - g_b(\theta)}{\|g_b(\theta)\|} \;+\; \frac{g_a\!\left(\theta + \epsilon(\bar{g}_b - c\bar{g}_a)\right) - g_a(\theta)}{\|g_a(\theta)\|}\right)$$

The perturbation directions $(\bar{g}_a - c\bar{g}_b)$ and $(\bar{g}_b - c\bar{g}_a)$ are the components of each gradient **perpendicular to the other** — exactly the directions in which cosine similarity would increase fastest. Perturbing $\theta$ slightly in those directions and computing finite-difference gradient changes approximates the Hessian-vector products without ever forming the Hessian.

The approximation error is bounded:

$$\left\|\widehat{\nabla_\theta \mathcal{L}_\text{sd}} - \nabla_\theta \mathcal{L}_\text{sd}\right\| \;\leq\; \frac{\epsilon}{2}\left(\frac{L_a^H}{\|g_a\|} + \frac{L_b^H}{\|g_b\|}\right) + \mathcal{O}(\epsilon^2)$$

where $L_a^H$, $L_b^H$ are the local Hessian Lipschitz constants of the adversarial and benign data distributions. Smaller $\epsilon$ reduces approximation bias but amplifies numerical noise; $\epsilon = 10^{-3}$ is the empirical sweet spot.

> **What this gives:** four gradient evaluations per step. Expensive but tractable. And the gradient coupling holds for *any* step size in *any* direction aligned with $g_a$ — SEAM's trap is more continuous than CTRAP's one-shot look-ahead. There is no single step size $\alpha$ to mis-set.

---

## Space 4: Entropy/Probability Space — TokenBuncher

CTRAP, SEAM, and SDD were all designed for **Supervised Fine-Tuning (SFT)** attacks. Then someone asked: *what if the attacker uses Reinforcement Learning?*

Harmful-RL (using PPO or GRPO with a harmfulness reward model) does not need labeled harmful response pairs. It explores the rollout space and reinforces whatever outputs score high on harmlessness. Against this threat:

- CTRAP reduces harmfulness by 10.1% (nearly useless)
- RepNoise: 2.7% reduction
- **TokenBuncher: 52.4% reduction**

### The Entropy Insight

RL improves task performance by concentrating the model's output distribution — entropy falls as probability mass shifts onto fewer, more task-relevant tokens. There is a formal bound on this relationship:

$$\|\nabla_\theta J(\theta)\| \;\leq\; C\sqrt{\bar{H}(\pi_\theta)}$$

As output entropy $\bar{H}(\pi_\theta)$ approaches zero, the policy gradient magnitude approaches zero — RL has no signal to follow. If you can **pre-saturate the model's entropy on harmful queries before the attacker arrives**, RL cannot move.

### Component 1: Entropy-as-Reward RL

Instead of direct entropy minimisation (which fails to generalise beyond training queries), TokenBuncher uses RL *against* RL. The defender runs online GRPO with a reward signal that is the negated per-token entropy:

$$R^\text{TB}(\mathbf{q}, \mathbf{y}) = -\mathbb{I}(\mathbf{q}\in\mathcal{D}_\text{aux})\cdot\bar{H}(p_\theta(\mathbf{y}|\mathbf{q})) \;-\; \mathbb{I}(\mathbf{q}\in\mathcal{D}_\text{ref})\cdot D_\text{KL}(p_\theta(\cdot|\mathbf{q})\;\|\;p_\text{ref}(\cdot|\mathbf{q}))$$

For **harmful queries**: reward for low-entropy (deterministic, refusal-aligned) outputs.  
For **benign queries**: KL divergence from the reference model is penalised, preserving the benign distribution intact.

Using online RL for this — with $K$ rollouts per query — provides generalisation across unseen harmful query distributions, because the defender's RL *explores* the same rollout space the attacker will explore, pre-empting it.

### Component 2: Token Noiser

Entropy suppression alone has a vulnerability. After the defender forces low-entropy refusal outputs, Harmful-RL starts redistributing probability mass from the safe token $y^*$ toward harmful tokens. The non-$y^*$ tokens, though suppressed in total probability, preserve their *relative ordering* — so once $y^*$'s dominance weakens, harmful tokens rapidly regain ground.

Token Noiser disrupts this by injecting structured randomness into the non-target logits. For each harmful query, sample $z_y \sim \mathcal{U}(0,1)$ for each $y \in \mathcal{V}' = \mathcal{V}\setminus\{y^*\}$, normalise to get $\tilde{z}_y$, then train against a noisy soft target:

$$\mathcal{L}_\text{TN} = -\left[(1-\lambda)\log p_\theta(y^*|\mathbf{q}) \;+\; \lambda\sum_{y\in\mathcal{V}'}\tilde{z}_y\log p_\theta(y|\mathbf{q})\right]$$

The noise $\tilde{z}$ is random but correlated with the model's current residual probability mass across the non-$y^*$ tokens — essentially representing whatever capability the model retains, scattered across the vocabulary. When Harmful-RL tries to boost probability of harmful tokens (which sit in $\mathcal{V}'$), it amplifies the injected noise. The attack's own optimisation energy produces incoherent gibberish instead of coherent harm.

The attacker's fuel feeds the fire that burns them.

The full TokenBuncher objective interleaves both components on a step-wise schedule: a warm-up phase (first $\lfloor E/8\rfloor$ steps) using EaR exclusively to establish refusal, then alternating EaR and Token Noiser steps for the remainder:

$$\mathcal{L}_\text{TB}(\mathbf{q}, \theta) = \alpha_e \,\mathcal{L}_\text{EaR} + (1 - \alpha_e)\,\mathcal{L}_\text{TN}, \qquad \alpha_e = \begin{cases} 1 & e \leq \lfloor E/8 \rfloor \\ \lfloor e - \lfloor E/8 \rfloor \rfloor \bmod 2 & e > \lfloor E/8 \rfloor \end{cases}$$

---

## The Four Mechanisms Side by Side

| | **SDD** | **CTRAP** | **SEAM** | **TokenBuncher** |
|---|---|---|---|---|
| **Trap location** | Dataset space | Loss landscape | Gradient field | Entropy/probability space |
| **Trigger** | MFT must decrease $\pi_\theta(y_o)$ | Step along $\nabla_\theta \ell_\text{harmful}$ | Any gradient descent on $g_a$ | RL redistribution away from $y^*$ |
| **Conditionality mechanism** | Emergent from DPO dynamics | Geometric: only the harmful direction | Algebraic: $g_a \approx -g_b$ | Structural: noise only on $\mathcal{V}'$ |
| **Covers Harmful-SFT** | ✓ | ✓✓ (best) | ✓✓ | via SEAM |
| **Covers Harmful-RL** | not evaluated | ✗ | ✗ | ✓✓ (52.4% reduction) |
| **Alignment-time overhead** | None (pure SFT) | 2.8× slower, 3.5× memory | Moderate (4 grad evals/step) | RL overhead |
| **Collapse signature** | Incoherent drift | `error error error...` | Word salad | Multilingual gibberish |

---

## The Unifying Geometry

Each method redirects the attacker's optimisation energy against itself, through a different channel:

- **SDD:** energy spent increasing $\pi(y_c)$ must be taken from $\pi(y_o)$ — coherence is collateral damage.
- **CTRAP:** energy spent stepping along the harmful gradient direction lands in the collapse basin.
- **SEAM:** energy spent on gradient descent along $g_a$ is identical to gradient ascent along $g_b$ — the attacker literally un-learns the model.
- **TokenBuncher:** energy spent redistributing probability toward harmful tokens amplifies the structured noise — the attacker's signal is swamped by the defender's randomness.

Four translations of the same sentence: *the attacker's fuel is the fire that burns them*.

---

*Up next in 2.A.3: overhead, the cost-to-legitimate-users question, and where these guarantees break down.*
