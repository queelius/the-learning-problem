---
title: "From A* to GPT: Rational Agents and the Representation Problem"
date: 2026-01-15
draft: false
tags: ["machine-learning", "reinforcement-learning", "llm", "rational-agents", "AI"]
categories: ["machine-learning", "AI"]
description: "The classical AI curriculum teaches rational agents as utility maximizers. The progression from search to RL to LLMs is really about one thing: finding representations that make decision-making tractable."
series: ["the-learning-problem"]
series_weight: 7
---

The classical AI curriculum teaches rational agents as utility maximizers. The progression from search to planning to reinforcement learning to probabilistic models is really about **one thing**: finding representations that make decision-making tractable. Large language models represent a new point in this design space—agents with massive pre-trained priors over rational behavior, where "learning" happens through in-context adaptation rather than gradient updates.

## The Rational Agent Framework

The foundational idea of artificial intelligence can be stated simply: an agent perceives the world, selects actions, and aims to maximize expected utility.

```
Agent = Percepts → Actions
Goal  = Maximize E[U(outcomes)]
```

The normative theory for this—Bayesian decision theory—tells us exactly *what* to compute: maintain rational beliefs via Bayes' rule, then choose actions maximizing expected utility under those beliefs. This is the gold standard. The challenge is making it tractable.

This framing unifies a surprisingly large range of AI systems. Chess engines evaluate board positions and select moves that maximize winning probability. Robotic navigation systems perceive obstacles and plan paths that minimize travel time while avoiding collisions. Dialogue systems generate responses that (ideally) maximize user satisfaction. Even large language models can be viewed through this lens—they select tokens that maximize the probability of generating coherent, helpful completions.

The elegance of this framework lies in its generality. Whether we're building a thermostat or a self-driving car, we can ask: What does this agent perceive? What actions can it take? What outcomes does it value?

The challenge, of course, is that the world is complex. State spaces are enormous—chess has roughly $10^{44}$ legal positions, Go has approximately $10^{170}$, and the real world has effectively infinite states. How do we make good decisions without exhaustive enumeration?

**The key insight**: The history of AI is largely about finding **representations** that make this tractable. Every major advance in AI can be understood as discovering or learning better ways to compress the complexity of the world into something a finite computational system can reason about.

## The Classical Progression

The standard AI curriculum walks through a sequence of increasingly sophisticated approaches to rational agency. Each step relaxes assumptions and adds new machinery to handle the resulting complexity. But viewed from above, they're all variations on the same theme: finding tractable representations for decision-making.

### Search

The simplest setting assumes a known world with deterministic transitions and a clear goal state. Given an explicit state graph—nodes representing configurations of the world, edges representing actions—we search for a path from start to goal.

A* search exemplifies the key idea: we can't explore everything, so we use a **heuristic** $h(n)$ that estimates the cost from any state to the goal. A good heuristic lets us focus computation where it matters, pruning vast portions of the search space.

The Manhattan distance heuristic for sliding tile puzzles. The straight-line distance for navigation. These hand-crafted functions encode human knowledge about problem structure, making intractable problems tractable.

**Limitation**: What if we don't know the transition model? What if outcomes are stochastic—the same action in the same state might lead to different results?

### MDPs and Planning

Markov Decision Processes introduce stochasticity. We now have transition probabilities $P(s'|s,a)$ and rewards $R(s,a)$. The goal shifts from finding a path to finding a **policy**—a mapping from states to actions that maximizes expected cumulative reward.

One approach is **expectimax tree search**: at each decision point, consider all actions, weight outcomes by their probabilities, and recursively compute expected values. This works—it's optimal—but it recomputes the same subproblems over and over. Every time you reach state $s$, you re-expand the entire subtree below it.

The key insight of dynamic programming: if the problem has **optimal substructure** (the Markov property), we can compute and cache the value of each state once. The representation that makes this tractable is the **value function**: $V(s)$ tells us the expected future reward from state $s$ under an optimal policy. The Bellman equations give us a recursive relationship:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

This is beautiful mathematics. Value iteration is essentially expectimax with memoization—we precompute the entire policy offline rather than searching online at decision time. It reduces planning over infinite horizons to a system of equations we can solve iteratively.

The trade-off: expectimax tree search is more general (it doesn't require enumerating all states upfront, handles non-stationary environments, and can focus computation on the current situation), while value iteration is more efficient when you have a complete model and will visit states repeatedly. This distinction matters later.

**Limitation**: Where do $P$ and $R$ come from? In the real world, we rarely have access to the true transition dynamics. And what if the state space is so large that we can't even enumerate it?

### Reinforcement Learning

Reinforcement learning drops the assumption that we know the environment's dynamics. Instead, we learn from experience—taking actions, observing outcomes, and updating our estimates of value.

Q-learning maintains estimates $Q(s,a)$ of the value of taking action $a$ in state $s$. Each experience $(s, a, r, s')$ generates an update:

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

With enough experience—visiting every state-action pair sufficiently often—these estimates converge to optimal values. The agent learns to act well without ever knowing the true transition probabilities.

**Limitation**: "Visiting every state-action pair" becomes impossible when state spaces are large. Tabular Q-learning doesn't generalize. Learning that ghosts are dangerous in one corner of the maze doesn't help us in another corner—unless we have the right representation.

### Function Approximation: The Key Insight

Here's where things get interesting. When state spaces are exponentially large, we can't store a separate Q-value for every (state, action) pair. We need to **generalize**.

The solution: approximate $Q(s,a) \approx \mathbf{w} \cdot \phi(s,a)$, where $\phi$ extracts **features** from the state-action pair and $\mathbf{w}$ is a learned weight vector.

Consider the Berkeley Pacman project. The maze has millions of possible configurations—Pacman's position, ghost positions, food locations, power pellet status. Tabular Q-learning is hopeless. But we can define features:

```python
features = {
    "bias": 1.0,
    "#-of-ghosts-1-step-away": count_nearby_ghosts(state, action),
    "eats-food": will_eat_food(state, action),
    "closest-food": distance_to_food(state, action) / max_distance
}
Q(s,a) = w · features  # Linear combination
```

Four features. That's it. These four numbers compress millions of possible states into a representation where learning is tractable AND—crucially—where knowledge generalizes. Learning that being near ghosts is bad in one part of the maze immediately applies everywhere.

**The punchline**: The quality of your features determines the quality of your agent. A brilliant feature representation can make an intractable problem trivial. A poor one makes even simple problems impossible.

But hand-crafting features is hard. It requires domain expertise, careful thought, and often many iterations. What if we could learn them automatically?

## Deep Learning: Learning the Representations

The breakthrough of the last decade: instead of hand-crafting features $\phi(s)$, learn them from data.

Deep Q-Networks (DQN) made this concrete. Instead of $Q(s,a) \approx \mathbf{w} \cdot \phi(s,a)$ with hand-crafted $\phi$, we have:

$$Q(s,a) \approx \text{neural\_network}(s, a)$$

The neural network takes raw state information—pixels from an Atari game screen—and outputs Q-values for each action. The hidden layers of the network learn features automatically. No human specifies what matters; the network discovers it through gradient descent on temporal difference errors.

The results were striking. DQN learned to play dozens of Atari games at human level or above, from raw pixels, with no game-specific engineering. The same architecture, the same learning algorithm, applied to Pong and Breakout and Space Invaders.

What did the network learn? Visualizing hidden layer activations reveals interpretable features: edges and textures in early layers, objects and spatial relationships in later layers. The network discovered what matters for playing these games—automatically.

Policy gradient methods take a different approach, learning the policy $\pi(a|s)$ directly rather than through value functions. Actor-critic algorithms, Proximal Policy Optimization (PPO), and their variants have achieved remarkable results in continuous control, robotics, and game-playing.

**The trade-off**:
- Hand-crafted features: interpretable, data-efficient, but limited by human insight
- Learned features: powerful and automatic, but hungry for data and often inscrutable

**Key point**: Deep RL is still the rational agent framework. We're still maximizing expected utility through sequential decision-making. We've just automated the most labor-intensive part: feature engineering.

## AlphaZero: Search Meets Learning

AlphaZero's breakthrough was combining online search with learned representations: it used Monte Carlo Tree Search (a form of expectimax) but aggressively pruned the search tree using learned value and policy networks. The networks provided a "warm start"—good default values and action priorities—while the search refined those estimates for the specific position at hand.

This combination unlocks something powerful: **rational action in novel situations**.

Consider what happens when AlphaZero encounters a board position it has never seen. Three mechanisms work together:

1. **Generalization via representations**: The network's learned features compress the state space. Even a novel position activates similar patterns to training positions—similar pawn structures, similar tactical motifs. The value estimate generalizes because the representation captures what *matters*, not the raw state.

2. **Exploration via world model**: With a model of the game (the rules), the agent can *imagine* future states. It doesn't need to have seen a position before—it can simulate what happens if it plays move A, then the opponent plays B, then it plays C. Search lets you reason about consequences in states you've never visited.

3. **Grounding via terminal states**: Search through an imagined future might reach positions with *known* values—checkmate, stalemate, or positions so advantageous the outcome is clear. These terminal states anchor the search in ground truth, not just learned estimates.

The result: AlphaZero could play brilliantly in positions no human or computer had ever seen, because it could (1) recognize structural similarity to familiar positions, (2) search forward through possibilities, and (3) ground that search in verifiable outcomes.

This wasn't just about learning better—it was about **acting rationally out of distribution at inference time**. The search is doing real reasoning, not just pattern matching. When the learned intuition is uncertain or wrong, the search can discover this and correct it, right now, for this specific situation.

The search also improved *training*. MCTS produces better move decisions than the raw network—it can "look ahead" and correct the network's mistakes. These search-improved decisions became training targets, bootstrapping the network to better performance.

This architecture—learned representations enabling efficient search, with verifiable endpoints grounding that search in truth—will reappear when we discuss LLMs.

## LLMs as Rational Agents

Now we arrive at the present moment. What happens when the "feature extractor" was pre-trained on most of human written knowledge?

### The Setup

A large language model can act as an agent. The state is described in text. The action is generated as text.

```
State description (text) → LLM → Action (text)
```

Consider a simple example:

```
You are in a 5x5 grid. You are at position (2,3).
There is food at (4,4). There is a ghost at (3,3).
Your goal is to eat the food without being caught.
Valid actions: north, south, east, west.
What action do you take?
```

A capable LLM responds: "I should move away from the ghost while heading toward the food. The ghost is directly north of me at (3,3), so going north would be dangerous. The food is to the northeast at (4,4). I'll move east to (3,3)—wait, that's where the ghost is. Let me reconsider. I'm at (2,3). Moving east takes me to (3,3), which is the ghost's position. Moving south takes me to (2,2), away from both ghost and food. Moving west takes me to (1,3). The safest path toward the food is probably to go south first, then east. I'll go south."

The model reasons about states, considers transitions, evaluates outcomes—and selects an action. Strip away the novelty, and this is still a rational agent—perceiving, reasoning, selecting actions to achieve a goal. The fundamental structure is unchanged from the 1960s. But *how* it achieves this is radically different.

### What's Different

**Pre-trained priors over rational behavior**: The LLM was trained on billions of tokens of humans writing about planning, reasoning, problem-solving, game-playing, decision-making under uncertainty. It has "seen" rational behavior in countless contexts. It's not learning from scratch—it arrives with massive prior knowledge about how agents pursue goals.

The representation $\phi(s)$ is language itself. And the LLM already knows how to extract meaning from natural language descriptions—how to parse "the ghost is at (3,3)" into a spatial relationship that matters for decision-making.

**In-context learning as adaptation**: Give the LLM a few examples in the prompt, and its behavior changes dramatically. This is learning without gradient updates. The model adapts its behavior based on context rather than weight modifications.

```
Example 1: State: ghost at (1,1), food at (2,2), you at (1,2). Action: east.
Example 2: State: ghost at (3,3), food at (1,1), you at (2,2). Action: west.
Now: State: ghost at (4,4), food at (1,1), you at (3,3). Action: ?
```

The model infers the pattern—move toward food, away from ghosts—and applies it. Few-shot learning, zero gradient descent.

**No explicit reward signal**: Classical RL requires explicit rewards: +10 for eating food, -500 for ghost collision, -1 per timestep. The LLM agent receives no such signal. "Utility" is implicit in the prompt ("your goal is to eat the food without being caught"). The model infers what you want from natural language.

### A Different Point in the Design Space

| Aspect | Classical RL | Deep RL | LLM Agent |
|--------|-------------|---------|-----------|
| Representations | Hand-crafted | Learned (task-specific) | Pre-trained (general) |
| Learning signal | Explicit reward | Explicit reward | Implicit (language) |
| Adaptation | Gradient updates | Gradient updates | In-context |
| Data efficiency | Low (tabular) / Medium (approx) | Low (needs much experience) | High (few-shot) |
| Generalization | Limited to features | Within distribution | Broad but shallow? |

This table reveals the trade-offs. LLM agents gain remarkable data efficiency—they can perform tasks zero-shot that would require millions of training examples for deep RL. But this comes from leveraging pre-training, not from fundamental sample efficiency gains.

### The Key Insight

LLMs are rational agents with a **massive prior**. They've seen so many examples of goal-directed behavior in text that they can often "do the right thing" without any task-specific training.

When you prompt GPT-4 to play chess, it doesn't search game trees like Stockfish. It pattern-matches to the countless chess games, analyses, and discussions in its training data. When you ask it to plan a route, it's not running Dijkstra's algorithm—it's drawing on a lifetime of human writing about navigation, directions, and spatial reasoning.

This is genuinely new. Previous agents either had hand-crafted knowledge (expert systems) or learned from scratch (RL). LLMs occupy a third position: they inherit vast knowledge from pre-training, then adapt through context rather than gradients.

But this comes with limitations:

**They may not actually plan**: When an LLM appears to reason through a problem, is it genuinely searching a space of possibilities, or pattern-matching to reasoning-like text? The answer likely varies by task and model, but there's evidence that much of what looks like planning is actually sophisticated retrieval and interpolation.

**Novel situations remain challenging**: Outside the distribution of training data, LLM agents can fail spectacularly. They excel at problems similar to what humans have written about; they struggle with genuinely novel challenges.

**No exploration-exploitation trade-off**: Classical RL agents balance exploiting known-good actions against exploring uncertain ones. LLM agents have no such mechanism built in. They produce confident outputs even when they shouldn't be confident.

**Implicit objectives can misalign**: When utility is specified in natural language, there's room for misunderstanding. "Help me win the game" admits many interpretations. The gap between what we say and what we mean—handled explicitly by reward functions in RL—becomes a source of potential misalignment.

### The Full Circle: RL on LLMs

Here's where the story comes full circle. The latest frontier in LLM development is... reinforcement learning.

RLHF (Reinforcement Learning from Human Feedback) was the first wave: train a reward model from human preferences, then use policy gradient methods to fine-tune the LLM to maximize that learned reward. This is how ChatGPT became ChatGPT—the base model could complete text, but RL made it *helpful*.

The newer wave uses **verifiable rewards**. For tasks where we can automatically check correctness—mathematical proofs, code that passes tests, factual questions with known answers—we can dispense with learned reward models entirely. The reward is binary and objective: did the proof check? Did the code compile and pass? Is the answer correct?

```python
Reward(response) = 1 if verify(response) else 0
```

Models trained this way show remarkable improvements in reasoning. They learn to check their work, to backtrack when stuck, to try multiple approaches. The RL signal teaches them *when* their reasoning has gone wrong, not just what good reasoning looks like.

**But here's the crucial point**: this RL is only tractable *because* of what the LLM already knows.

Consider the action space. A typical LLM generates from a vocabulary of 50,000+ tokens, producing responses hundreds of tokens long. The space of possible responses is astronomical—far larger than any Atari game, any robotic control task, any domain where RL has previously succeeded. Naive RL in this space would be hopeless.

What makes it work is that the pre-trained LLM has already compressed this space into something tractable. It already knows syntax, semantics, mathematical notation, logical structure. The RL doesn't need to discover that "2 + 2 = 4"—it needs to discover *when to apply* the knowledge the model already has. The pre-training provides the representation; the RL provides the objective.

This is the synthesis:

| Component | What it provides |
|-----------|------------------|
| Pre-training | Rich representations, world knowledge, reasoning patterns |
| RL fine-tuning | Objective signal, optimization toward verifiable goals |

Neither alone is sufficient. Pre-training without RL gives you a model that can mimic reasoning but doesn't reliably *achieve* goals. RL without pre-training faces an intractable search space. Together, they produce agents that combine broad knowledge with goal-directed optimization.

The classical curriculum—from search to MDPs to Q-learning to function approximation—wasn't a historical detour. It was laying the foundations for exactly this moment: applying reinforcement learning to the most powerful function approximators we've ever built.

The parallel to AlphaZero is striking. Remember the three mechanisms we identified: generalization via learned representations, exploration via a world model, and grounding via terminal states with known values. LLMs have all three—language as a compressed representation, chain-of-thought as a form of search, and verifiable outcomes (proofs, tests, factual verification) as checkable endpoints.

The difference is scale and generality. AlphaZero had a perfect world model (the rules of chess) and clear terminal states (checkmate). LLMs have an *approximate* world model (learned from text about how the world works) and verification is only available for some domains. But the architecture is the same: rich learned representations enable efficient search, and verifiable endpoints ground that search in truth.

This connects to one of the most important paradigms in current AI: **test-time compute** (also called inference-time scaling). Just as AlphaZero allocates more MCTS rollouts to difficult positions, LLMs can allocate more "thinking" to hard problems. Extended reasoning, chain-of-thought, and search-like exploration at inference time all trade compute for accuracy. The pre-trained representations make this tractable; the verifiable rewards tell us when to stop.

The parallel runs deeper. When we train LLMs with RL on verifiable rewards, we're explicitly training them to generate chains of thought that reach verifiable terminal states. The model learns to *imagine* reasoning steps—a latent form of search through the space of possible arguments—until it arrives at something checkable: a proof that verifies, code that passes tests, an answer that matches ground truth.

The "world model" here is language itself, and the model's learned representations of how reasoning works. It's approximate and imperfect, but it's rich enough to support meaningful search. And crucially, we can generate verifiable terminal states at scale through synthetic data:

- Take a working program, introduce a bug → finding that bug is now verifiable
- Generate math problems with known solutions → checking answers is automatic
- Create logical puzzles with provable solutions → verification is mechanical
- Build synthetic physical systems with known laws → [discovering those laws is verifiable](/post/2025-01-05-science-as-verifiable-search/)

This last point extends the thesis beyond math and code. Science itself is search through hypothesis space, and we can train the *process* of scientific inquiry—experiment design, hypothesis refinement, anomaly detection—by creating synthetic worlds where we control the ground truth. The agent sees only probes and measurements; we know the underlying laws. The strategies transfer to real domains.

This synthetic data trick is powerful: we can manufacture unlimited training signal by creating problems where we *already know* what success looks like. The model learns to search through reasoning space to reach these known-good endpoints, and that capacity transfers to problems we don't have answers for.

The latent chain of thought is the LLM's equivalent of AlphaZero's tree search—imagined exploration grounded by verifiable outcomes.

## Open Questions

Several fascinating questions emerge from this synthesis:

**When do LLM agents work? When do they fail?** Early evidence suggests they excel at tasks well-represented in training data and struggle with those requiring genuine search or novel reasoning. Mapping this boundary more precisely would be valuable.

**Can we combine strengths?** Perhaps LLMs for high-level reasoning and goal decomposition, classical planners for combinatorial search, deep RL for low-level control. The architecture of such hybrid systems remains underexplored.

**What is "learning" when it's in-context?** In-context learning updates no weights, yet changes behavior. Is this computation, or learning, or something else? Recent theoretical work suggests ICL can be viewed as implicit Bayesian inference or as running learned optimization algorithms in forward passes.

**What about grounding?** LLMs learn from text about the world, not from the world directly. Does this matter? When? The debate over whether language models can truly "understand" connects to deep questions in philosophy of mind and cognitive science.

## Conclusion

The rational agent framework—perceive, reason, act to maximize utility—remains the core abstraction in AI. What's changed over six decades is how we obtain the representations that make this tractable:

1. **Hand-crafted** (classical AI): Human experts specify features, heuristics, and knowledge representations. Interpretable and data-efficient, but limited by human insight and labor-intensive to develop.

2. **Learned from task experience** (deep RL): Neural networks discover features through gradient descent on task rewards. Powerful and automatic, but requiring massive experience and often producing inscrutable representations.

3. **Pre-trained from vast general data** (LLMs): Models acquire rich priors about the world and rational behavior from internet-scale text. Remarkably data-efficient at new tasks, but potentially superficial and prone to distributional failures.

None is universally superior. Search algorithms remain the right tool for combinatorial optimization. Deep RL excels when we can simulate environments cheaply. LLM agents shine when tasks can be described in natural language and resemble what humans discuss in text.

Understanding where each approach excels—and how they might be combined—is the work ahead. The Berkeley AI course teaches (1) and the foundations of (2). Connecting these foundations to (3) is what we've sketched here.

The rational agent framework is not obsolete. It's the lens through which we can understand what LLMs are doing—and what they're not. They are agents with massive priors, adapting through context rather than gradients, pursuing goals specified in language rather than reward functions. This is genuinely new. But it's also genuinely continuous with everything that came before.

## Further Reading

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach* (4th ed.). The canonical textbook covering rational agents, search, MDPs, and RL.
- Berkeley CS188: Introduction to Artificial Intelligence. Course materials at [ai.berkeley.edu](https://ai.berkeley.edu).
- Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." *Nature*. The DQN paper.
- Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." *arXiv*. The AlphaZero paper.
- Ouyang, L. et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS*. The InstructGPT/RLHF paper.
- Lightman, H. et al. (2023). "Let's Verify Step by Step." *arXiv*. Process reward models for mathematical reasoning.
- Wei, J. et al. (2022). "Emergent abilities of large language models." *TMLR*.
- Yao, S. et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*.
