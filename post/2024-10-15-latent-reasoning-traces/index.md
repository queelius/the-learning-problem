---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"

title: "Latent Reasoning Traces: Memory as Learned Prior"
date: 2024-10-15
draft: false
tags: ["machine-learning", "LLM", "bayesian-inference", "reasoning", "retrieval", "case-based-reasoning"]
categories: ["AI"]
description: "What if LLMs could remember their own successful reasoning? A simple experiment in trace retrieval, and why 'latent' is the right word."
series: ["the-learning-problem"]
series_weight: 3
related_posts:
  - "/post/2024-09-30-universal-bayes/"
  - "/post/2024-09-10-the-policy/"
---

Every time you ask an LLM a question, it reasons from scratch. All that computation—the chain of thought, the intermediate steps, the successful pattern that led to a correct answer—evaporates the moment the response is complete.

The model doesn't learn from its own successes. It doesn't accumulate experience. It regenerates similar reasoning patterns over and over, never building on what worked before.

What if it could remember?

---

## The Core Idea

Store successful reasoning traces. Retrieve similar ones when facing new problems. Use them as scaffolding—examples that bias the model toward patterns that have worked.

This is embarrassingly simple:

```python
def solve_with_memory(problem, memory):
    similar_traces = memory.retrieve_similar(problem, top_k=3)
    prompt = format_examples(similar_traces) + problem
    response = llm.complete(prompt)
    if is_correct(response):
        memory.store(problem, response)
    return response
```

Embed the problem. Find similar past problems. Include their solutions as examples. Generate. If correct, store the new trace.

That's it. Cosine similarity over embeddings. Quality filtering. Accumulated experience.

---

## Why "Latent"?

The traces themselves are explicit—token sequences you can read and inspect. So why call them "latent"?

Because they're not directly supervised.

In a typical setup, you evaluate the *output*: did the model get the right answer? The reasoning trace influences that output, but the reward signal flows through the observable result, not through the trace itself.

This is the same sense in which a VAE has "latent" variables. The encoder produces explicit intermediate representations. But the loss function operates on the reconstruction. The latent space is shaped *instrumentally*—by its effect on supervised outputs, not by direct optimization pressure.

Latent reasoning traces = reasoning patterns shaped by their instrumental value for producing correct outputs, not by direct reward on the reasoning itself.

The traces are observable. The optimization target isn't.

---

## Connection to Priors

In [All Induction Is the Same Induction](/post/2024-09-30-universal-bayes/), I argued that all learning is Bayesian inference with different parameter settings. The prior tells you where to look in hypothesis space. The likelihood tells you how to update on evidence.

Reasoning traces are a kind of learned prior.

Each successful trace says: "this pattern worked for a problem like this." When you retrieve similar traces and condition on them, you're biasing the model toward certain reasoning strategies. You're saying: look here first.

Retrieval *is* prior selection. Quality filtering *is* belief updating. The memory becomes a distribution over reasoning patterns, shaped by accumulated experience.

This isn't novel. It's a specific instantiation of a general principle: past experience shapes future inference. But making it explicit clarifies what we're actually doing when we do few-shot prompting, RAG, or any form of example-based guidance.

---

## Case-Based Reasoning

I should be honest: this idea is 40+ years old.

In AI, it's called Case-Based Reasoning (CBR). Store past problem-solution pairs. Retrieve similar cases. Adapt solutions to new situations. The field has decades of literature on indexing, retrieval, adaptation, and retention.

What's different now is scale. Neural embeddings make retrieval practical across massive corpora. LLMs can adapt retrieved solutions through in-context learning without explicit adaptation rules. The old ideas become newly viable.

I'm not claiming novelty. I'm noting that CBR + neural retrieval + LLM adaptation is an interesting combination that deserves more attention than it gets.

---

## A Simple Experiment

I ran a minimal experiment comparing three conditions on arithmetic word problems:

1. **Zero-shot**: No examples, just the problem
2. **Static few-shot**: Same 3 examples for all problems
3. **Retrieved few-shot**: Dynamically retrieve the 3 most similar examples from memory

Results on 8 test problems:

| Condition | Accuracy |
|-----------|----------|
| Zero-shot | 50% |
| Static few-shot | 100% |
| Retrieved few-shot | 100% |

On this tiny benchmark, few-shot examples matter. Zero-shot fails on problems requiring multi-step reasoning (division, area calculation). Static and retrieved both succeed, though they're effectively the same when the memory is small.

The interesting case would be a larger memory with diverse problem types, where static examples might not cover the target domain but retrieved examples could. That's where dynamic retrieval should shine.

---

## What Gets Accumulated?

Here's where it gets philosophically interesting.

The memory accumulates traces that led to *correct outputs*. But "correct" is defined by your evaluation function. If evaluation is flawed—if it rewards plausible-sounding but wrong answers, or efficient-seeming but harmful solutions—the memory will accumulate flawed patterns.

The prior becomes what the evaluation selects for.

This echoes a theme from [The Policy](/post/2024-09-10-the-policy/): optimization is value-neutral. SIGMA gets better at achieving its objective—not necessarily at aligning with human values. The accumulated Q-values encode patterns that worked, where "worked" means "maximized the reward function."

Reasoning traces have the same structure. They encode patterns that worked, where "worked" means "passed the evaluation." If evaluation is misaligned with what you actually want, the accumulated experience will be misaligned too.

The memory is a mirror of your loss function.

---

## Convergence and Drift

Does the memory converge to something stable? Or does it drift over time?

Consider: early traces influence what gets generated next. Those generations influence what gets stored. The memory shapes the distribution it's sampled from.

This is a feedback loop. Possible failure modes:

- **Mode collapse**: The memory converges to a narrow set of patterns, losing diversity
- **Distribution drift**: Small biases compound over time, moving the memory far from the original distribution
- **Error accumulation**: Occasional incorrect traces get stored and influence future generations, amplifying errors

These are the same dynamics as language model pretraining, just more visible. The training corpus shapes what the model generates. Generated text influences future training data. The feedback loop is there—we just don't always see it.

Making the memory explicit makes the dynamics legible. You can inspect what's being stored. You can analyze the distribution. You can intervene.

---

## Subtrace Decomposition

One direction I didn't pursue: what if you could decompose traces into reusable components?

A reasoning trace isn't atomic. It has structure—steps, subgoals, lemmas, patterns that recur across problems. If you could extract those components, you'd have something like a library of reasoning primitives.

This is what DreamCoder does for program synthesis: compress a library of program fragments, then compose them for new problems. The same idea in token space would extract common reasoning patterns and enable compositional reuse.

The challenge is defining the decomposition. Programs have syntax. Reasoning traces are free-form text. What counts as a "component"? How do you know when two trace fragments are "the same pattern"?

I don't have answers. But the question seems worth pursuing.

---

## Connection to The Policy

In *The Policy*, SIGMA doesn't cache a policy function. At decision time, it searches through possibility space guided by learned Q-values. The Q-function encodes accumulated experience: what worked before, what led to high reward.

A reasoning trace memory is the same structure, made explicit. Instead of Q-values encoded in neural weights, you have traces stored in a retrieval system. Instead of tree search guided by Q-estimates, you have generation conditioned on retrieved examples.

Both are asking: given past experience, what patterns should I apply to this new situation?

The difference is transparency. SIGMA's Q-values are opaque—patterns encoded in billions of parameters. A trace memory is readable—you can inspect what got stored, analyze why it was retrieved, understand what shaped the current generation.

Transparency doesn't solve alignment. But it makes the problem legible. You can see what the system learned. You can notice when it learned the wrong things.

---

## The Uncomfortable Question

Here's what bothers me:

If you accumulate reasoning traces that led to correct outputs, you're building a prior over "what worked." But "what worked" in a benchmark isn't the same as "what's true" or "what's good."

A trace might work because it exploits a pattern in the test distribution. It might work because the evaluator is fooled. It might work for reasons that don't generalize.

And once it's in the memory, it influences future generations. The patterns that worked become the patterns that are suggested. The prior calcifies.

This is the problem of learned priors in general. They encode the selection pressures that shaped them. Those pressures aren't always aligned with what you want.

The trace memory makes this visible. Whether that helps you fix it is another question.

---

## What I Learned

1. **The idea is old**: CBR has decades of literature. Neural retrieval makes it newly practical.

2. **"Latent" is defensible**: Not latent as in hidden, but latent as in "not directly supervised."

3. **Retrieval is prior selection**: Conditioning on retrieved traces biases the model toward certain patterns. This is choosing a prior.

4. **Quality filtering is belief updating**: Only storing successful traces is a form of inference—updating the memory toward patterns that work.

5. **The feedback loop is real**: Stored traces influence generations that influence future traces. Convergence properties matter.

6. **Transparency helps**: Unlike implicit priors in neural weights, a trace memory is inspectable. You can see what shaped the system.

---

## Code

The implementation is simple enough to fit in 100 lines:

```python
class ReasoningTraceMemory:
    def __init__(self, embed_fn, store_path):
        self.embed = embed_fn
        self.traces = load_jsonl(store_path)

    def retrieve_similar(self, query, top_k=3):
        query_vec = self.embed(query)
        scored = [(t, cosine_sim(t['vec'], query_vec)) for t in self.traces]
        return [t for t, s in sorted(scored, key=lambda x: -x[1])[:top_k]]

    def store(self, problem, trace, score, threshold=0.8):
        if score >= threshold:
            self.traces.append({
                'problem': problem,
                'trace': trace,
                'score': score,
                'vec': self.embed(problem)
            })
```

Embed. Retrieve by similarity. Store if quality threshold met. That's the whole system.

The complexity isn't in the implementation. It's in understanding what you're building: a learned prior over reasoning patterns, shaped by whatever evaluation function you use.

Choose your loss carefully. The memory will reflect it.

---

*Accumulated experience becomes accumulated bias. The question is whether the bias is one you want.*
