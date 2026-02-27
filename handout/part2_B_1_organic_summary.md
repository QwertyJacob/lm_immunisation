# Part 2.B.1 — The Road So Far: An Organic Summary

> **Session:** Afternoon · Part B · 5 minutes  
> **Role of this segment:** Step back. Survey the terrain covered since this morning. Not a list of papers — a narrative of how a research quest crystallised, what it has achieved, and where the unresolved edge is. The setup for everything that follows in 2.B.

---

## Where It All Began: A Problem That Shouldn't Exist

There is something philosophically uncomfortable at the heart of this field that is worth naming before we go further.

We spend enormous effort aligning language models — RLHF, constitutional AI, safety fine-tuning, red-teaming pipelines. The goal is to produce a model that behaves well. Then we release it as open weights, and within days someone fine-tunes out everything we worked to install. The safety wasn't structural. It was a thin coating, and fine-tuning is a solvent.

This is the **Vulnerability Argument** in its starkest form: *no matter how safe a model appears at inference time, if that safety can be easily removed, the model is fundamentally unsafe.* It's not a new observation — it was implicit in early alignment critiques — but it became concrete and urgent when the combination of open-weight releases and accessible fine-tuning infrastructure made the attack trivially cheap. Alignment without structural resistance is optimism, not safety.

The question immunisation sets out to answer is: *can we make the safety structural?* Not just behavioural, not just a token-level habit, but woven into the model's geometry in a way that fine-tuning cannot easily undo.

---

## The Crystallisation: How a Quest Became a Field

The quest didn't arrive fully formed. It crystallised in stages.

**2022 — The proof of concept.** MLAC showed that meta-learned adversarial censoring could make a BERT-era classifier a bad starting point for harmful adaptation. Small model, narrow task, approximate gradients. But the concept was on the table: *learn not to learn harmful tasks*. The vocabulary existed. The framework existed. The scale did not yet.

**2023–2024 — The problem becomes urgent.** Two discoveries changed everything. First, shallow alignment: safety training in LLMs mostly shifts the distribution over the first few output tokens. The deep generative capacity for harm is never removed — it's patched over. This means alignment is one good fine-tuning run away from erasure. Second, harmful embedding drift: fine-tuning on user data causes the residual stream representations of harmful prompts to visibly migrate away from where the aligned model placed them. The geometry of safety is legible and fragile. These two findings turned immunisation from an interesting theoretical exercise into a practical necessity.

**2024–2025 — The ecosystem forms.** The field split into two philosophical branches. Block 1 — make harmful fine-tuning expensive: Vaccine, T-Vaccine, RepNoise, Booster, TAR, Circuit Breakers, Condition Number, LoX. Block 2 — make the destination worthless: SDD, CTRAP, SEAM, TokenBuncher. Eighteen papers across three years, covering weight space, representation space, gradient space, entropy space. The breadth is real. The depth in each mechanism is real.

---

## The Road Done: What We Can Now Say

By the end of today, we can make a set of statements that the field could not make in 2022:

**We have formal definitions.** The four pillars — Resistance, Stability, Generalisation, Trainability — give us a vocabulary for evaluating defences that goes beyond anecdotal benchmarks. A paper that achieves resistance at the cost of trainability has not solved the problem; it has traded one failure mode for another.

**We have two principled strategies.** Not a zoo of heuristics, but two coherent philosophical camps with distinct mechanisms, distinct failure modes, and distinct performance profiles. The asymmetry in group size (fourteen versus four) is not evidence that Block 1 is better — it reflects that Block 2 is newer and harder to get right.

**We have attack coverage beyond SFT.** TokenBuncher's response to Harmful-RL is not a minor extension. It opens a new dimension of the problem. RL-based attacks don't need labeled harmful data, don't follow fixed gradient trajectories, and outperform SFT on capability preservation. The field now has at least one answer to this threat.

**We have practical tools.** T-VAC and SEAM are not just papers — they are notebooks you ran today. The gap between theory and implementation has narrowed substantially in three years.

---

## The Open Edge

And yet. The open edge is real, and we should look at it clearly before the challenge and before the hopeful close.

The field has not solved the problem. It has **characterised the difficulty and made partial progress**. The things it has not yet done:

- Demonstrated strong resistance against adaptive attackers who know the defence and optimise specifically against it, not just against naïve fine-tuners.
- Solved the pre-alignment base model problem: any defence applied at alignment time is bypassed by attackers starting from the pre-alignment checkpoint.
- Produced a defence that is simultaneously fully resistant, fully stable, fully generalisable, and fully trainable — the four pillars are still in tension with each other, and no method occupies all four vertices of that space.
- Developed evaluation methodology rigorous enough that the field can trust its own benchmarks. ASR on BeaverTails under one fine-tuning configuration is not a sufficient characterisation of resistance.

These are not reasons for pessimism. They are the open map. They are precisely what makes this a live research problem and not a solved one — which is to say, they are the reason you are here.

---

*In 2.B.2 we will go into the current SOTA's limitations with more precision, drawing on the critique papers. Then the challenge. Then the reason there is reason for hope.*
