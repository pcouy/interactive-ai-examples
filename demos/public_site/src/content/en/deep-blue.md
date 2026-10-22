---
title: Deep Blue · adversarial tree search
description: |
    Play tic-tac-toe or Connect Four against an agent that explores the tree of
    moves. Explore the agent's decision process.
---

## The game

The left part of the screen lets you play a game of tic-tac-toe or
Connect Four against an agent that uses tree search (the minimax algorithm),
or make 2 agents play against each other.

The right part of the screen lets you expand and explore the search tree that
allowed the agent to decide on a move, from the current game position.

The controls in the top area let you configure the search algorithm.

The "Max depth" parameter in the right part only controls the display of the
expanded tree. Clicking a game position in the tree displays only the subtree
from that position.

## Minimax

Minimax assumes that each player plays optimally. It expands moves to a depth limit, then assigns a value to the leaves. The value propagates back up to the root: the agent maximizes its score, and the opponent minimizes it. A terminal position has a known value: win, loss, or draw.

An end-of-game position is worth $+1$ for a win, $-1$ for a loss, and $0$ for a
draw.

$$ V(s) = \begin{cases} E(s) & \text{if } s \text{ is terminal} \\ \max_{a \in A(s)} V(s_a) & \text{if it is the agent's turn} \\ \min_{a \in A(s)} V(s_a) & \text{otherwise} \end{cases} $$

$s$ is a state, and $V(s)$ is its value. $E(s)$ is $+1$, $0$, or $-1$ (terminal
state). $s_a$ is the game state after playing move $a$ from position
$s$.

## Alpha-beta pruning

The full tree quickly becomes huge. Alpha-beta prunes the branches that cannot change the decision. It carries two bounds: the best value guaranteed to the maximizer and to the minimizer. The search is restricted to the window $[\alpha, \beta]$. The result remains identical to minimax, but the visited tree is smaller.

## The evaluation function

For a game like tic-tac-toe, alpha-beta pruning is enough to explore
the whole tree in a reasonable time. However, even for Connect Four (much
smaller than chess), it is no longer enough. So we limit the maximum
search depth to a few moves, to contain the computation time and
the memory space required.

At the leaves of the tree (when we explore to the maximum depth without
reaching an end-of-game position), the agent does not know the outcome of the game. An evaluation function estimates the position: alignments, threats, free cells. Its choice changes the style of play.

The `windows` choice (the default) corresponds to:
- Always 0 in non-terminal states for tic-tac-toe
- A "simple" evaluation function for Connect Four

The `rollout` evaluator plays a few random games from each
leaf to estimate the value.

The `custom` choice lets you try to develop your own evaluation
function.

The evaluation function is one of the main weaknesses of "Deep Blue-style"
methods. It requires expert knowledge of the game and its strategies, and
perhaps something even harder: you must be able to translate those strategies and
intuitions into code. It is long and difficult work, and it only works for
a single game. Tree search can work with many games, but the evaluation
function must be rebuilt from scratch for each game.

## The discount factor

The $\gamma$ parameter weights the value of child moves during backup. An immediate gain is then worth more than a distant one. With $\gamma = 1$, all gains keep the same weight.

Note that with $\gamma = 1$, the agent "gives up" as soon as the tree
search proves that you can force a win. With $\gamma < 1$, for
example $\gamma = 0.99$, the agent will always prefer a move that lets it
delay its loss, or win faster.

## Transposition table (TT)

The search tree can be reduced further by noticing that some
positions are equivalent, by symmetry or rotation, and by reusing the
values already computed for equivalent positions.

In addition to reducing the size of the search, this table also provides
a rudimentary version of reinforcement learning. By
using appropriate rules to update and reuse the stored values,
the agent can automatically learn values that replace the evaluation
function.

### What you can try

1. Change the search depth and watch the tree grow.
2. Compare minimax and alpha-beta: the move stays the same, but the number of visited nodes drops.
3. Enable the transposition table and browse its entries to see the positions already evaluated.
