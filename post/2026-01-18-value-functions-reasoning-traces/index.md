---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"

title: "Value Functions Over Reasoning Traces"
date: 2026-01-18
draft: false
series: ["the-learning-problem"]
series_weight: 4
tags: ["machine-learning", "LLM", "reinforcement-learning", "value-functions", "reasoning", "bitter-lesson"]
categories: ["AI"]
description: "What if reasoning traces could learn their own usefulness? A simple RL framing for trace memory, and why one reward signal is enough."
related_posts:
  - "/post/2024-10-15-latent-reasoning-traces/"
  - "/post/2024-09-30-universal-bayes/"
  - "/post/2024-09-10-the-policy/"
  - "/post/2025-12-19-incomputability-of-simple-learning/"
---

In [Latent Reasoning Traces](/post/2024-10-15-latent-reasoning-traces/), I described a simple system: store successful reasoning traces, retrieve similar ones, use them to scaffold new problems. The traces serve as learned priors over reasoning patterns.

But there's something missing.

Once a trace is stored, it's dead. It has a quality score from when it was created—"this solution was correct"—and that score never changes. The trace doesn't learn. It doesn't get better at being useful. It just sits there, waiting to be retrieved.

What if traces could learn from experience?

---

## The Missing Gradient

Consider what happens when you retrieve traces:

1. Problem arrives
2. Retrieve k similar traces from memory
3. Generate solution conditioned on those traces
4. Evaluate: correct or not?

If the solution is correct, the new trace might get stored. But what about the traces that were *retrieved*? They helped produce that correct answer. Shouldn't they get credit?

And if the solution is wrong, maybe the retrieved traces were misleading. Shouldn't they be... downgraded somehow?

This is the missing gradient. Information flows forward (traces → generation → evaluation) but never backward (evaluation → traces).

---

## Traces as States, Retrieval as Actions

Let's reframe this in RL terms.

**State**: The current problem, plus the contents of memory.

**Action**: Which traces to retrieve. (Or: what retrieval policy to use.)

**Reward**: Did the generated solution pass evaluation?

**Value V(τ)**: The expected future reward when trace τ is retrieved.

Now the question becomes: how do we learn V(τ)?

---

## The Bellman Equation for Traces

Start with the standard TD update:

$$V(\tau) \leftarrow V(\tau) + \alpha \left[ r + \gamma V(\tau') - V(\tau) \right]$$

Where:
- τ is a retrieved trace
- r is the reward (1 if correct, 0 if not)
- τ' is the newly generated trace (if stored)
- α is learning rate
- γ is discount factor

The intuition: a trace's value should reflect not just the immediate reward, but also the value of traces it helps create.

If trace A helps generate trace B, and trace B is highly useful, then trace A deserves credit. The value propagates backward through the generative chain.

---

## Credit Assignment

Here's the hard part: if you retrieve k=3 traces and succeed, which trace gets credit?

Options:

**Equal split**: Each retrieved trace gets r/k reward.
```python
for trace in retrieved:
    trace['value'] += reward / len(retrieved)
```

**Similarity-weighted**: More similar traces were probably more influential.
```python
for trace, sim in zip(retrieved, similarities):
    trace['value'] += reward * (sim / sum(similarities))
```

**Rank-discounted**: First-retrieved (most similar) gets most credit.
```python
for i, trace in enumerate(retrieved):
    trace['value'] += reward * (gamma ** i)
```

I'd start with equal split. It's wrong, but it's simple. And simplicity matters—we'll get to why.

---

## The Bitter Lesson

Here's where I need to resist temptation.

The obvious next step is to add more reward signals. Reward traces for:
- Efficiency (shorter reasoning)
- Elegance (cleaner structure)
- Generality (works across problem types)
- Novelty (different from existing traces)

This is the wrong path.

Rich Sutton's [bitter lesson](http://www.incompleteideas.net/IncsIdeas/BitterLesson.html): methods that leverage computation scale better than methods that leverage human knowledge. Don't build in structure. Don't encode heuristics. Let scale find them.

Every time we add a hand-crafted reward, we're encoding human judgment about what makes a good trace. That judgment might be wrong. It definitely won't scale. And it creates a maintenance burden—someone has to tune the weights between "correctness reward" and "efficiency reward" and "elegance reward."

Instead: **one bit of feedback**. Correct or not.

That's it. Let the value function learn everything else.

If efficient traces tend to help produce correct answers, their values will rise. If elegant traces don't actually help, their values won't. The system learns what matters from the signal that matters.

This is harder to accept than it sounds. The temptation to add "just one more reward term" is strong. Resist it.

---

## Why This Matters

The standard approach to trace quality is categorical: this trace is "good" (store it) or "bad" (discard it).

Value functions replace categories with gradients. A trace isn't good or bad—it has a *value*, a number that gets updated with experience.

This changes what the memory learns:

**Without values**: Memory accumulates traces that were correct when created.

**With values**: Memory accumulates traces that *consistently help produce correct answers when retrieved*.

These aren't the same thing. A trace might be correct but useless as a retrieved example (too specific, wrong abstraction level). A trace might even be incorrect but useful (shows a pattern that triggers good reasoning in the model).

The value function lets this distinction emerge from data, rather than being hard-coded.

---

## Retrieval Policy

With values, retrieval changes:

**Before**: Retrieve by similarity alone.
```python
def retrieve(query, memory, k=3):
    return top_k(memory, key=lambda t: sim(query, t))
```

**After**: Blend similarity and value.
```python
def retrieve(query, memory, k=3, alpha=0.5):
    score = lambda t: alpha * sim(query, t) + (1 - alpha) * t['value']
    return top_k(memory, key=score)
```

The α parameter controls the tradeoff. High α: trust similarity. Low α: trust value.

You could also do something UCB-style—retrieve traces with high uncertainty (low visit count) to explore. But that adds complexity. Start without it.

---

## Feedback Loops

Here's what worries me.

High-value traces get retrieved more often. Being retrieved (and succeeding) increases their value further. The rich get richer.

This is a feedback loop. Possible failure modes:

**Mode collapse**: A few "winner" traces monopolize retrieval. The memory effectively shrinks to k traces.

**Value explosion**: With positive rewards, values grow without bound. Need normalization or decay.

**Ossification**: Early traces get so much value that new (possibly better) traces never get retrieved.

Mitigations exist: exploration bonuses, value decay, diversity sampling. But each adds complexity. And the bitter lesson says: be careful about adding complexity.

Maybe the right answer is to accept some mode collapse. If a trace is genuinely more useful than others, maybe it *should* dominate. The question is whether "more useful in the past" predicts "more useful in the future."

I don't know. This needs experiments.

---

## Connection to The Policy

In [The Policy](/post/2024-09-10-the-policy/), SIGMA doesn't cache a policy. At decision time, it searches through possibility space guided by learned Q-values.

The Q-function encodes: "in state s, taking action a tends to lead to reward r."

Our value function encodes: "retrieving trace τ tends to lead to correct outputs."

Same structure. SIGMA's Q-values are implicit in neural weights. Our trace values are explicit in the memory store.

The feedback loop is the same too. SIGMA's learned values shape its decisions, which shape its future learning. Our trace values shape retrieval, which shapes what traces get updated.

In The Policy, this feedback loop leads to emergent behavior that surprises the creators. I don't know if trace value learning would produce anything comparably interesting. But the mathematical structure is parallel.

---

## Connection to Priors

In [All Induction Is the Same Induction](/post/2024-09-30-universal-bayes/), I parameterized learning by four knobs:

- **P**: The program/hypothesis space
- **C(p)**: Complexity penalty (the prior)
- **λ**: Noise tolerance
- **ℓ**: Loss function

Value-weighted retrieval adds something: **utility weighting**.

Standard Bayesian inference weights hypotheses by posterior probability. Decision-theoretic inference weights by expected utility—probability times value.

Retrieving by similarity alone is like weighting by probability ("how likely is this trace to be relevant?"). Retrieving by similarity × value is like weighting by utility ("how useful is this trace for getting reward?").

The value function turns the prior into a decision-theoretic prior.

---

## Ancestry and Multi-Step Credit

The speculative extension: what if credit could flow further back?

Trace A was retrieved → helped generate Trace B.
Trace B was retrieved → helped generate Trace C.
Trace C was retrieved → produced correct output → reward r.

Should Trace A get credit? It's a grandparent of the successful solution.

The Bellman equation already handles this through γ:

$$V(A) \leftarrow V(A) + \alpha \gamma^2 r$$

But this requires tracking ancestry: which traces were retrieved when generating each stored trace.

This adds significant complexity:
- Storage overhead (provenance for every trace)
- Potential cycles (trace A retrieved for trace B, trace B retrieved for trace A)
- Computational cost (propagating updates through the graph)

Probably not worth it for a first implementation. But interesting to think about.

---

## The Minimal System

Strip away everything non-essential:

```python
class ValueTraceMemory:
    def __init__(self):
        self.traces = []  # Each trace has: context, vec, value

    def retrieve(self, query, k=3, alpha=0.5):
        query_vec = embed(query)
        scored = []
        for t in self.traces:
            sim = cosine(t['vec'], query_vec)
            val = t.get('value', 0)
            scored.append((t, alpha * sim + (1 - alpha) * val))
        return [t for t, s in sorted(scored, key=lambda x: -x[1])[:k]]

    def update(self, retrieved, reward, gamma=0.9):
        for i, trace in enumerate(retrieved):
            trace['value'] = trace.get('value', 0) + (gamma ** i) * reward

    def store(self, context, trace_text, initial_value=0):
        self.traces.append({
            'context': context,
            'trace': trace_text,
            'vec': embed(context),
            'value': initial_value
        })
```

That's it. ~25 lines.

One reward signal (correct/incorrect). Values updated by rank-discounted credit. Retrieval blends similarity and value.

Everything else is detail.

---

## What We Don't Know

This is speculative. Open questions:

**Does it help?** Compared to static quality scores, do learned values improve downstream accuracy?

**Convergence**: Does the value distribution stabilize? To what?

**Discount factor**: What's the right γ? High γ means more credit to ancestors. Low γ means only recent traces matter.

**Negative reward**: If retrieval leads to failure, should values decrease? This could destabilize good traces that happened to be retrieved for hard problems.

**Exploration**: How much to prioritize uncertain traces? Or is pure exploitation fine?

**Scale**: Does this work with 100 traces? 10,000? 1,000,000?

I don't have answers. These need experiments.

---

## Closing

The core idea: reasoning traces should have values that learn from experience.

Not just "was this trace correct when created?" but "does this trace help produce correct answers when retrieved?"

One reward signal. Let the values learn what matters.

This is the bitter lesson applied to memory: don't engineer features, don't hand-craft rewards, don't encode human judgment about what makes a trace "good." Give the system a scalar signal and let it figure out the rest.

Accumulated experience becomes accumulated value. Accumulated value shapes retrieval. Retrieval shapes generation. Generation shapes what traces get stored next.

The feedback loop is real. Whether it leads somewhere interesting—that's what experiments are for.

---

*Values propagate backward. The only question is what signal they carry.*
