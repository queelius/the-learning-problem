---
title: "The Incomputability of Simple Learning"
date: 2025-12-19
draft: false
tags: ["machine-learning", "philosophy", "AI", "bitter-lesson", "bayesian-inference", "solomonoff-induction"]
categories: ["AI"]
description: "An exploration of why the simplest forms of learning may be incomputable, and what that means for the intelligence we can build."
series: ["the-learning-problem"]
series_weight: 2
---

Karpathy's recent ["Animals vs Ghosts"](https://karpathy.bearblog.dev/animals-vs-ghosts/) piece has been rattling around in my head. In it, he surfaces a tension that deserves more attention: the author of "The Bitter Lesson" — the text that's become almost biblical in frontier AI circles — isn't convinced that LLMs are bitter lesson pilled at all.

The bitter lesson, in brief: methods that leverage computation scale better than methods that leverage human knowledge. Don't build in structure; let the model learn it. Don't encode heuristics; let scale find them. The lesson is "bitter" because it means a lot of clever human engineering ends up being wasted effort, steamrolled by dumber approaches with more compute.

LLM researchers routinely ask whether an idea is "sufficiently bitter lesson pilled" as a proxy for whether it's worth pursuing. And yet Sutton, the lesson's author, looks at LLMs and sees something thoroughly entangled with humanity — trained on human text, finetuned with human preferences, reward-shaped by human engineers. Where's the clean, simple algorithm you could "turn the crank" on and watch learn from experience alone?

This got me thinking about why that clean algorithm is so elusive. And I've come to suspect the answer is uncomfortable: **the simplest forms of learning may be incomputable, or at least intractable, in ways that force us into approximations that fundamentally shape the resulting intelligence.**

---

## The Library of All Programs

Here's one way to see the problem.

Imagine the space of all possible programs, like Borges' Library of Babel. In that famous story, a library contains every possible book — every combination of characters up to 410 pages. Most are gibberish. A tiny fraction are meaningful. The library "contains" all knowledge, all literature, all truth.

And it is utterly useless. Because you can't find anything.

Now consider program space. It contains:
- The perfect predictor for any phenomenon
- The optimal policy for any environment
- The ideal world model
- The shortest program that explains any dataset
- Every possible mind

The space is *complete*. The answer to any question is already "in there." The perfect intelligence exists, in some abstract sense, as a point in this vast combinatorial space.

And it is useless, for the same reason as Borges' library. The problem isn't generation — you can enumerate programs, in principle. The problem is **indexing**. Navigation. Search through a space so vast that random exploration is hopeless.

This reframes what learning is.

**Learning isn't synthesis; it's search.** We're not *creating* intelligence, we're *navigating* to it in the space of possible minds.

---

## The Bayesian Substrate

Here's the formal version of the same idea.

All learning, at some level, is inference. You have hypotheses about the world, you observe evidence, you update your beliefs. Bayes' theorem tells you how to do this optimally — weight hypotheses by prior probability, update on likelihood, normalize.

Solomonoff induction is just Bayesian inference with a particular choice of prior and model class: consider all computable hypotheses, weight them by algorithmic simplicity (shorter programs are more probable), update on observed data. It's provably optimal in a certain sense.

It's also provably incomputable.

The incomputability comes from two places. First, the model class is too large — all possible programs. Second, the prior itself (Kolmogorov complexity) is uncomputable. You can't, in general, determine the length of the shortest program that produces a given output.

But notice what Solomonoff induction *is*: it's a prescription for navigating program space. The prior is a *map* — it tells you where to look, which regions are more likely to contain the program you want. Short programs first, then longer ones.

The map is perfect. And the map is unreadable.

---

## No Free Lunch

Here's why you can't escape this.

The No Free Lunch theorems say something that sounds almost nihilistic: averaged over all possible problems, no learning algorithm beats random guessing. Every algorithm that does well on some problems must do poorly on others. The wins and losses exactly cancel.

But there's a constructive reading of NFL. It tells you that **to do well on specific problems, you must assume some patterns are more likely than others.** You need priors. You need inductive biases. You need a map.

The question isn't whether to have biases — you can't avoid them. The question is where they come from:

- **Evolved biases**: Animal brains, shaped by billions of years of selection, embody priors about physics, other agents, cause and effect. These are maps drawn by evolution. Consider working memory: Miller's famous "7 ± 2" chunks. That's not an accident—it's a bottleneck that forces compression. You can't remember every detail, so you must extract patterns, build models, find regularities. Memory limits automatically implement Occam's razor: only simple enough hypotheses fit through the bottleneck.

- **Derived biases**: Sometimes we can work out from first principles what patterns to expect. Physics gives us conservation laws. Information theory gives us compression. These are maps drawn by understanding.

- **Discovered biases**: Meta-learning, neural architecture search, learned optimizers. Maybe compute can discover its own maps. These would be maps drawn by search.

- **Handcrafted biases**: Transformers, attention mechanisms, positional encodings. These are maps drawn by human intuition and trial-and-error.

Each is a different way of constraining the search through program space. Each says: look here, not there. This region is more likely to contain what you want.

---

## Unprincipled Maps

Here's where it gets uncomfortable.

The transformer architecture *is* an inductive bias. It encodes assumptions about what functions are likely to be useful. Attention says "relevant information can be anywhere in context." Positional encoding says "order matters, but in this specific way." The whole thing carves out some subspace of possible programs and says: search here.

But we don't have a probability density over this space.

In proper Bayesian inference, your prior is a probability distribution. You can quantify uncertainty. You can know when you're extrapolating beyond your prior's support. You can update coherently as evidence arrives. The math works out.

With neural networks, we have none of this. We have point estimates (trained weights) instead of posteriors. We have an implicit prior (the architecture plus initialization plus optimizer) that we can't write down as a probability measure. We're doing something *shaped like* inference — hypothesis space, updates, generalization — but with unquantified priors and no principled uncertainty.

We have a map. But we can't read it. We don't know what territory it claims to describe. We can't tell when we've wandered off the edge.

Maybe this is fine. Maybe the implicit prior of "transformer trained on internet text" happens to be close enough to useful that it works in practice. But it's worth noticing how far we are from the clean formalism that would let us say *why* it works, or predict *when* it will fail.

We're navigating by a map we don't understand through territory we can't see.

---

## Approximate Maps

So we're stuck between theoretical optimality (incomputable) and principled uncertainty (intractable). What do we actually do?

We approximate. And each approximation is a different way of drawing a map.

**Pretraining on human text.** Karpathy calls this "our crappy evolution" — a hack to avoid the cold start problem. And I think that's exactly right, but it's worth dwelling on *why* it works.

Human text has extraordinarily high signal-to-noise ratio. Not by accident — by construction. Every sentence you read represents effort, intention, selection. Someone chose those words over alternatives. The corpus isn't raw reality; it's reality filtered through billions of human decisions about what's worth saying.

Pretraining works because we're not starting from scratch. We're using human text as a proxy for "useful programs look like things that predict this." It narrows the search space dramatically. It's a map, albeit one drawn by the collective motion of human minds rather than any principled analysis.

Is this bitter lesson pilled? It doesn't feel like it. It feels more like... inheriting the distilled results of human cognition rather than rediscovering them from scratch. Sweet lesson pilled, maybe.

**Verifiable rewards.** Karpathy makes another sharp observation:

> Software 1.0 easily automates what you can **specify**.
> Software 2.0 easily automates what you can **verify**.

Verification is what makes search tractable. If you can cheaply check whether a program is good, you can do local search, hill-climbing, reinforcement learning. You can navigate. Without verification, you're back to wandering blind.

This creates what Karpathy calls the "jagged frontier." Tasks with clean verification — math problems, code that compiles, games with win conditions — progress rapidly. Tasks without clean verification — creative work, strategic reasoning, anything requiring taste or judgment — advance more slowly, relying on generalization and hope.

But verification is a human-shaped constraint. What can be verified depends on what humans have figured out how to check. We can verify proofs because we built proof checkers. We can verify code because we built compilers and test suites. These are human artifacts — tools we created, metrics we defined.

So when we optimize for "verifiable rewards," we're really optimizing for problems where humans have already solved the verification problem. That's a strong selection effect. The map is drawn by the shape of human formal methods.

---

## Different Maps, Different Minds

Here's another way the approximations matter.

Karpathy distinguishes between "animals" and "ghosts" — two different points in the space of possible intelligences, reached by different optimization pressures.

**Animal intelligence** was found by evolution's search:
- Optimize for survival and reproduction in a physical world
- Deeply embodied, continuous, always-learning
- Social — huge compute dedicated to modeling other agents
- Shaped by adversarial multi-agent self-play where failure means death
- The map: billions of years of selection pressure in physical reality

**LLM intelligence** was found by a very different search:
- Statistical imitation of human text (pretraining)
- Task completion and human preference (finetuning, RLHF)
- Disembodied, fixed weights, context-window-bounded
- No continuous self, no embodied stakes
- The map: human documents + verifiable benchmarks + preference data

These aren't points on a spectrum. They're different regions of mind-space, reached by different search algorithms using different maps.

The maps encode different assumptions. Evolution's map says: programs that survive and reproduce in physical reality are good. Our map says: programs that predict human text and solve verifiable problems are good.

No surprise, then, that the intelligences found are different. LLMs are shape-shifters, statistical imitators, spiky and capable in some domains, brittle in others. Animals are general, embodied, robust, optimized for not dying across countless scenarios.

Ghosts and animals. Different search processes, different maps, different destinations.

---

## Other Maps?

So here's the question I keep circling back to:

Are there other maps? Other search strategies? Ways of navigating program space that are:

- **Simpler** than curated datasets and hand-designed rewards
- **More principled** than architecture intuitions we can't formalize
- **More tractable** than Solomonoff induction

Candidates people talk about:

**Curiosity / Information Gain.** Seek states that reduce uncertainty about the world. The map says: programs that learn efficiently are good. But this requires having a world model good enough to notice what's surprising — which is itself a hard problem. The map requires a map.

**Prediction Error Minimization.** Active inference, free energy frameworks. The map says: programs that minimize surprise are good. But pure surprise-minimization leads to degenerate solutions. The agent that closes its eyes and predicts darkness has minimized surprise perfectly. The map needs constraints.

**Empowerment.** Maximize the channel capacity between your actions and future states. Keep options open. The map says: programs that maintain influence over the future are good. Elegant, but computing empowerment is intractable in complex environments. The map is unreadable.

**Boundary Maintenance.** This one's interesting because it inverts the question. Instead of asking "what reward signal produces intelligence?", it asks "what computational structure *is* intelligence?"

One answer: intelligence is the maintenance of a self/non-self boundary, a region of low entropy in a high-entropy universe. Life itself as a self-maintaining boundary. The "map" isn't a search strategy but a definition — intelligence is whatever maintains its own existence as a coherent computational structure.

I don't know if any of these lead anywhere. Each has implementation challenges that push you back toward approximations, toward the same messy heuristics we're already using. Maybe the incomputability is fundamental. Maybe any tractable learning algorithm necessarily picks up biases from wherever you make it tractable.

But maybe not. The space of possible maps is itself vast. We've explored only a tiny region.

---

## Questions I'm Left With

I don't have conclusions. I have questions:

**Is the incomputability fundamental?**

Is there a theorem lurking here — something like "any learning algorithm that is both general and tractable must incorporate domain-specific structure"? Or are there paths to simpler learning that we just haven't found yet?

**What are we actually approximating?**

When we train transformers on human text, we're approximating *something*. But what? Is there a well-defined target we're approaching, or is it turtles all the way down — approximations of approximations with no ground truth?

**Can you navigate from "ghost" to "animal"?**

Karpathy speculates that maybe you can finetune ghosts "more and more in the direction of animals." But optimization pressure shapes deep structure. Can you undo the shape-shifting, sycophantic, human-imitation core of an LLM? Or are ghosts and animals different basins of attraction in mind-space, unreachable from each other?

**What maps are we not seeing?**

Pretraining on text is one map. Verifiable rewards are another. But the space of possible maps is large. What are we not exploring because we're path-dependent on what's worked so far? What would a truly different search strategy look like?

**What does "simple" even mean?**

The bitter lesson says simple algorithms + scale beats complex engineering. But "simple" is slippery. Solomonoff induction is conceptually simple — and incomputable. Evolution is mechanistically simple — and requires billions of years. Is there a notion of simplicity that's both meaningful and achievable?

---

## Coda

The space of all programs contains every possible mind. The perfect learner is in there, somewhere, as a point in that vast combinatorial space. The Library of Babel is complete.

And it is useless.

Because finding something in an infinite library is as hard as writing it from scratch. Search is the bottleneck. Navigation is the problem. And navigation requires maps — priors, biases, assumptions about where to look.

The bitter lesson tells us what would work in principle: simple algorithm, lots of compute, scale indefinitely. But the simplest algorithms are incomputable. So we approximate — with human data, with verifiable rewards, with architectural intuitions we can't formalize.

Each approximation is a tradeoff. Each draws a different map. Each shapes the intelligence we find in ways we're only beginning to understand.

Maybe LLMs are ghosts — not animals, not the platonic ideal, but something new. A different region of mind-space, reachable by the maps available to us. Statistical echoes of humanity, shape-shifters trained on our documents, useful and strange.

Or maybe they're waypoints. Stepping stones toward something we don't have words for yet. Points on a trajectory through mind-space that we're only beginning to trace.

I don't know. But I think the question "what kind of learning is actually possible?" deserves more attention than it gets. Not "what benchmarks can we hit?" but "what are the fundamental constraints on minds, and how do our methods navigate them?"

The bitter lesson is a direction, not a destination. And the path there — if there is a path — runs through territory we don't have maps for yet.

---

*The Library contains everything. The hard part was never writing the books.*
