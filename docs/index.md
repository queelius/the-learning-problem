---
title: "The Learning Problem"
description: "Essays on induction, inference, and the search for useful representations"
---

## The Problem

How do you learn anything at all?

Solomonoff induction tells you how to do it optimally: consider all hypotheses, weight by simplicity, update on evidence. It's mathematically beautiful.

It's also incomputable.

Every practical learning algorithm is an approximation. And every approximation encodes assumptions—about what patterns are likely, what representations are useful, what search strategies will find good solutions.

These assumptions are priors. They're the maps we use to navigate hypothesis space.

## The Theme

These essays explore a single idea from multiple angles:
**learning is constrained search, and the constraints shape what gets learned.**

- All induction is Bayesian inference with different knobs
- The simplest learning is impossible, forcing approximations
- Those approximations—priors, architectures, objectives—shape the resulting intelligence
- The bitter lesson: scale + simple algorithms beat clever engineering, but you still need the right inductive biases

## The Arc

1. **Theory**: Why all induction reduces to the same framework
2. **Incomputability**: Why we're forced into approximations
3. **Memory**: How accumulated experience becomes learned priors
4. **Value**: How systems can learn what's useful, not just what's correct
5. **Search**: How tree search navigates reasoning space
6. **Agents**: How optimization pressure shapes emergent behavior

## The Question

If the learning problem is fundamentally about search, and search requires priors, what should those priors be?

Where should we look?
