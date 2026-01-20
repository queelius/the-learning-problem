---
title: "MCTS-Reasoning: Tree Search for LLM Reasoning"
date: 2024-12-01
draft: false
description: "Applying Monte Carlo Tree Search to large language model reasoning with a rigorous formal specification."
tags: ["MCTS", "LLM", "machine-learning", "reasoning", "tree-search", "Python"]
categories: ["AI Research"]
series: ["the-learning-problem"]
series_weight: 5
linked_project: [mcts-reasoning]
---

I've been working on a project that applies Monte Carlo Tree Search (MCTS) to LLM-based reasoning. The core insight is that multi-step reasoning can be modeled as a sequential decision problem where MCTS can systematically explore different reasoning paths.

## The Problem with Single-Shot Reasoning

When you ask an LLM a complex question, it generates a single response. If that response goes down a wrong path, there's no recovery. The model commits to its initial approach and follows it to completion, even if better alternatives existed.

MCTS addresses this by building a tree of reasoning paths and using the UCB1 bandit algorithm to balance exploration of new paths with exploitation of promising ones.

## How It Works

The system treats reasoning as:
- **States**: Partial reasoning traces (what's been written so far)
- **Actions**: Reasoning continuations (the next step)
- **Terminal states**: Complete solutions with final answers
- **Rewards**: Quality assessments of final answers

Each MCTS simulation runs through four phases:

1. **Selection**: Traverse the tree using UCB1 to pick promising paths
2. **Expansion**: Add a new reasoning step via LLM generation
3. **Rollout**: Continue reasoning until reaching a terminal state
4. **Backpropagation**: Update statistics back up the tree

### Tree-Building Rollouts

A key design choice is using **tree-building rollouts**. Unlike game-playing MCTS that uses a fast random policy without storing nodes, we add every rollout node to the tree. This preserves the full reasoning trace and allows reuse of reasoning steps in future simulations.

### Terminal-Only Evaluation

The evaluator is invoked only on terminal states. Intermediate reasoning states aren't evaluated, reducing computational cost. LLM-as-judge calls happen only when a complete answer is produced.

## The Technical Report

I've written a [formal specification](/papers/mcts-reasoning/) that provides rigorous definitions for all components:

- Formal definitions of states, actions, nodes, and the search tree
- Precise pseudocode for all four MCTS phases
- Clear interfaces for Generator and Evaluator components
- Complexity analysis showing O(KD) tree operations for K simulations with max depth D

The goal was to establish a canonical reference that authentically captures MCTS while adapting it for LLM reasoning.

## Usage

The library provides both a fluent API and an interactive shell:

```python
from mcts_reasoning import ReasoningMCTS, get_llm

mcts = (
    ReasoningMCTS()
    .with_llm(get_llm("anthropic"))
    .with_question("What is the optimal sorting algorithm for 1M integers?")
    .with_exploration(1.414)
    .with_max_rollout_depth(5)
)

mcts.search("Let's think step by step...", simulations=50)
print(f"Solution: {mcts.solution}")
print(f"Confidence: {mcts.best_value:.2%}")
```

Or interactively:

```bash
mcts-shell
> ask What is the sum of primes less than 100?
> search 50
> solution
> sample 5  # Get diverse reasoning paths
> consistency 20  # Check solution consistency
```

## Extensions

The tech report discusses several extensions under consideration:

- **Extended action spaces**: Compress (for long traces), Verify, ToolCall, Backtrack
- **Algorithm variants**: Progressive widening, RAVE, parallel MCTS
- **Graph-based reasoning**: DAG structures for problem decomposition and critique-revision cycles

The code is available at [github.com/queelius/mcts-reasoning](https://github.com/queelius/mcts-reasoning) and the technical report is on the [papers page](/papers/mcts-reasoning/).
