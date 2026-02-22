# A Taxonomy of Immunisation Methods
We could divide these 18 papers along other axes — by attack type targeted (SFT vs. RL vs. ITI), by where in the pipeline they intervene (weight-space vs. representation-space), or by whether they offer strong vs. weak resistance. But we will divide by their philosophy, which is the *fundamental philosophical commitment*. In this sense, there is a **strategic bifurcation** in what "safety" means:

We basically have two groups, the first of which says: *a safe model is one that stays safe even under attack*, while the second says: *a safe model is one that becomes useless before it becomes harmful*.

This matters enormously from the perspective of the pvery definition of immunisation, which requires both resistance *and* trainability.  While the first group of methods try to keep stability and trainability, the second group of methods are the ones that most aggressively sacrifice trainability — and arguably general utility — in exchange for a hard guarantee.

As we will see, there is an asymmetry in grout size — 14 vs. 4 — and this asymmetry is itself meaningful. While the second group is younger, and more radical research direction, the field has been predominantly occupied with the first group, where the hard engineering challenge is preserving trainability and utility while raising resistance. The second group sidesteps that challenge by accepting a more extreme trade-off, and it's only recently gaining serious traction.

## Block 1 — *Raising the Cost of Harmful Convergence*

This block groups all methods whose core logic is: **prevent the harmful fine-tuning from succeeding, or make it so expensive that it's not worth attempting.** The model that comes out the other side of immunisation still works normally; it's just been placed in a region of parameter/representation space that is hostile to harmful optimization.

The approaches within this block are diverse in *mechanism* but unified in *intent*:

- **Adversarial meta-learning approaches** (MLAC/Self-Destructing Models, TAR, NTL, SOPHON) operate at the weight-space level, using bi-level optimisation or gradient geometry to push parameters into a "safety basin" that is a local minimum for harmful tasks.

- **Representation/noising approaches** (Vaccine, T-Vaccine, RepNoise, Circuit Breakers, Booster, Antidote, E.T.) operate at the activation/residual-stream level, ensuring that harmful perturbations in latent space either drift back to a safe region or are actively rerouted/neutralised.

- **Geometric/loss-landscape approaches** (Condition Number, LoX) characterise the difficulty of harmful learning in terms of the Hessian's condition number or low-rank subspace geometry, and engineer the model to maximise that difficulty on harmful tasks while minimising it on benign ones.

The common thread: **the attacker can try to fine-tune, and the model will passively resist**. The model's safety and utility coexist; the goal is resistance without sacrificing trainability.

| # | Paper |
|---|-------|
| 1 | Non-Transferable Learning (NTL) |
| 2 | Self-Destructing Models (MLAC) |
| 3 | CTRL |
| 4 | Vaccine |
| 5 | Circuit Breakers |
| 7 | [E.T.](https://www.techrxiv.org/users/925680/articles/1301297-ethical-treatment-of-language-models-against-harmful-inference-time-interventions) / [LAPT](https://arxiv.org/pdf/2506.16078)|
| 8 | TAR |
| 9 | Booster |
| 10 | [LoX](https://arxiv.org/pdf/2506.15606) |
| 11 | VAA |
| 12 | ILU |
| 13 | [Condition Number](https://arxiv.org/pdf/2505.23760) |
| 17 | Targeted Vaccine |
| 19 | [AntiDote](https://arxiv.org/pdf/2509.08000) |


## Block 2 — *Conditional Model Collapse (Fail-Safe Traps)*

This block groups all methods whose core logic is radically different: **don't bother resisting the attack; instead, plant a booby trap so that if harmful fine-tuning is attempted, the model destroys its own general capabilities.** Safety is guaranteed not by making harmful learning hard, but by making a successfully attacked model functionally useless.

The methods here all share the same philosophical skeleton, even if they differ in how the trap is triggered and what the collapse looks like:

- **CTRAP and SDD** embed a collapse loss during alignment that forces the model to predict a fixed "error" token for any input if pushed in a harmful direction — the model becomes an incoherent shell.
- **TokenBuncher** targets the specific threat of harmful reinforcement learning by minimising response entropy on harmful queries, removing the exploration signal that RL relies on to converge — starving the attack of its fuel.
- **Self-destructive LM variants** pursue similar goals through related mechanisms.

The common thread: **the attacker may succeed in removing the safety guardrails, but finds nothing useful underneath.** The trap turns the model's general intelligence against the attacker.

| # | Paper |
|---|-------|
| 14 | CTRAP |
| 15 | SEAM (Self-Destructive LM) |
| 18 | SDD |
| 20 | Token Buncher |


### A side note on paper 2 (Self-Destructing Models):

This paper is literally called *Self-Destructing Models*, and CTRAP, SDD, and SEAM all cite it as their direct conceptual ancestor. In that *lineage* sense, one would say it belongs with Block 2.

However, when we will look at what MLAC **actually does mechanistically**, we'll notice it sits firmly in Block 1. It uses bi-level meta-learning to push the model's parameters into a region that is a local minimum for harmful tasks — making harmful fine-tuning expensive and inefficient. The model doesn't collapse; it just becomes a bad starting point for harmful adaptation. There is no trap, no triggered degradation, no conditional inertness. The adversary who persists with enough data and compute can still get there.

The Block 2 papers took the *name and the spirit* of self-destruction but completely changed the *mechanism*: instead of raising cost, they engineer a conditional catastrophe — if you push, the model becomes useless for everyone.

So while CTRAP is the conceptual grandfather of Block 2, it is technically a Block 1 paper — which is actually quite an interesting position for it to occupy. It's the bridge between the two blocks.
