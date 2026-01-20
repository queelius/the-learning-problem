---
author:
  name: "Alex Towell"
  email: "queelius@gmail.com"
  url: "https://metafunctor.com"

title: "All Induction Is the Same Induction"
date: 2024-09-30
tags: ["machine-learning", "solomonoff-induction", "bayesian-inference", "information-theory", "philosophy"]
categories: ["machine-learning", "statistics"]
description: "Solomonoff induction, MDL, speed priors, and neural networks are all special cases of one Bayesian framework with four knobs."
draft: false
series: ["the-learning-problem"]
series_weight: 1
---

Solomonoff induction is the theoretical gold standard for prediction: average over all possible programs weighted by their simplicity. Shorter programs get higher weight. Given enough data, it converges to the true distribution faster than any other method.

Two problems: it's uncomputable, and it's brittle. One wrong prediction and a hypothesis is dead.

But here's the insight: **Solomonoff induction is just one setting of four knobs**. Turn those knobs differently and you get MDL, speed priors, or something that looks a lot like neural network training. Same framework, different parameters.

## The Four Knobs

Every induction method makes choices about:

**1. Which programs to consider** — All programs? Only short ones? Only fast ones?

**2. How to penalize complexity** — Length? Runtime? Both?

**3. How much noise to tolerate** — Must predictions be exact? Can we allow small errors?

**4. How to measure errors** — Binary right/wrong? Squared error? Log loss?

Write these as parameters and you have:

- **P**: the program space
- **C(p)**: complexity penalty (acts as prior)
- **λ**: noise tolerance (higher = stricter)
- **ℓ(o, x)**: loss function

The posterior weight for program p given data x₁, ..., xₙ is:

$$P(p \mid x_{1:n}) \propto C(p) \cdot \exp\left(-\lambda \sum_{i=1}^n \ell(p(i), x_i)\right)$$

This is just Bayes' rule. The exponential term is the likelihood. Different parameter choices give different induction strategies.

## Recovering Solomonoff

Classical Solomonoff induction:
- **P** = all programs
- **C(p)** = 2^{-|p|} (exponential preference for short programs)
- **λ** → ∞ (infinite strictness)
- **ℓ(o, x)** = 0 if o = x, else ∞ (binary match)

With λ → ∞, any program that makes a single wrong prediction gets exp(-∞) = 0 weight. Only programs that perfectly match the data survive. Among those, shorter programs dominate exponentially.

This gives the classic result: optimal prediction, but uncomputable (we can't enumerate all programs) and fragile (real data has noise).

## Making It Computable

Want computability? Restrict P:
- Only programs of length ≤ N
- Only programs that halt within T steps

Now everything is finite. We can actually run this. As N, T → ∞, we approach full Solomonoff induction.

This is the (N, T)-bounded approximation. It's no longer optimal, but it's *actually runnable*. The convergence guarantee: with large enough bounds, predictions get arbitrarily close to the full model.

## Speed Priors

Some short programs are slow. A program that solves a problem by brute force might be brief but take exponential time.

Levin's speed prior penalizes both length and runtime:

$$C(p) = \frac{2^{-|p|}}{t(p)}$$

Now a 10-character program that runs in 10⁹ steps gets the same weight as a 40-character program that runs in 1 step. We prefer *fast* simple explanations over slow simple ones.

This is still a valid setting of the four knobs. Same framework.

## Handling Noise

Real data is noisy. Strict Solomonoff kills any hypothesis that doesn't match perfectly.

Fix: use finite λ with a continuous loss function:

- **λ** = some finite constant
- **ℓ(o, x)** = (o - x)² (squared error)

Now a program that predicts 0.99 when the true value is 1.0 doesn't get zero weight—it just gets slightly penalized. Multiple small errors accumulate, but don't instantly disqualify.

This is essentially Bayesian regression over programs. The exp(-λ · loss) form is an exponential family likelihood.

## What Neural Networks Are Doing

A neural network implicitly defines:
- **P** = architectures expressible by the network structure
- **C(p)** = implicit prior from initialization and regularization
- **λ** = determined by learning rate and loss scaling
- **ℓ** = whatever loss function you chose

The network can't represent all programs—only those within its architecture. But within that space, gradient descent approximately finds the posterior mode.

Weight decay is a complexity penalty. Dropout regularizes. Early stopping prevents overfitting to noise. These are all ways of tuning the effective λ and C(p).

Neural networks aren't doing something fundamentally different from Solomonoff induction. They're doing a heavily resource-bounded version with implicit parameter choices.

## The Tradeoff Space

| Method | P | C(p) | λ | ℓ |
|--------|---|------|---|---|
| Solomonoff | all programs | 2^{-\|p\|} | ∞ | 0/∞ binary |
| (N,T)-bounded | length ≤ N, time ≤ T | 2^{-\|p\|} | ∞ | 0/∞ binary |
| Speed prior | all programs | 2^{-\|p\|}/t(p) | ∞ | 0/∞ binary |
| MDL | model class | 2^{-description length} | finite | log loss |
| Neural net | architecture | regularization | learning dynamics | task loss |

These aren't competing theories. They're points in the same parameter space.

## Why This Matters

**Theoretical clarity**: Arguments about which induction method is "best" become arguments about parameter choice. The framework itself isn't in dispute.

**Practical guidance**: If you want computability, restrict P. If you have noise, use finite λ. If you want fast predictions, penalize runtime in C(p). The tradeoffs are explicit.

**Biological plausibility**: Brains probably implement something like (N, T)-bounded induction with substantial noise tolerance. Limited working memory constrains P. Sensory noise requires finite λ. Neural architecture defines C(p) implicitly.

## The Bottom Line

Solomonoff induction isn't a separate algorithm from practical machine learning. It's an extreme parameter setting: all programs, maximum strictness, zero noise tolerance.

Relax those settings and you get computable, noise-tolerant methods. The math stays the same—just Bayesian inference with different priors and likelihoods.

Different problems call for different settings. The framework tells you what you're trading off.

---

*All induction is Bayesian inference over programs. The only question is which programs, with what penalties, under what noise model.*
