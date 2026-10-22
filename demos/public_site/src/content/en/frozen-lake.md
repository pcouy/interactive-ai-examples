---
title: Frozen Lake · tabular Q-learning
description: Guide an agent that learns on its own to cross a frozen lake. Set exploration, learning, slip, and replay.
---

## *Gridworld* environments

*Gridworld* environments are often used to quickly test
intuitions in reinforcement learning or to illustrate algorithms.
In general, you can apply reinforcement learning
algorithms whenever a problem can be modeled by an agent in an
environment:

- The agent observes a state $s$ of the environment
- The agent chooses an action $a$
- The environment produces a new state $s'$ and a
  reward $r$ "in response" to the action

*Gridworlds* are environments in which each state is a cell.
The agent observes the identifier of the cell, and can choose among 4 actions
corresponding to the 4 directions. Some cells can, when the agent lands
on them, end the episode, give a reward, or both at the same time.

## The frozen lake

It is a classic *gridworld* environment.

An agent moves on a frozen grid. Each cell is solid (no
reward) or hides a hole (end of episode with no reward). The goal
cell contains a reward of 1 unit. If the agent falls, the game stops. It must find a safe path without knowing the map.

## Learning by trial and error

The agent does not receive the rules of the game. It acts, observes the reward, and updates
the value $Q(s,a)$ of the state-action pair. A positive reward reinforces
the action; a fall penalizes it. The arrows then show the best known
move. You can hover over cells with the mouse to see the
value that the agent assigns to each action from that cell.

## Exploration and exploitation

If it always chooses the best known action, the agent can get stuck. Epsilon sets the probability of choosing a random action. A high value favors discovery; a low value favors exploitation.

## Slip and update

The slip probability is zero by default. Increasing it adds a
probability that the chosen action leads the agent to a neighboring cell other
than the one "expected". This models environments that we cannot
observe perfectly (unlike the games that Deep Blue can play).


The agent tries to learn the "value" $Q$ of each action, which corresponds to the
expected future reward following that action choice. When the agent experiences
a "transition" $(s, a) \rightarrow (s', r)$ it can deduce
the value $Q(s, a)$ of action $a$ from state $s$:

$$ Q_\text{observed}(s,a) = r + \gamma \max_{a'} Q(s', a') $$

The $\gamma$ parameter is the same discount factor as the one presented on
the [page about tree search](/en/deep-blue). Its value $\gamma
< 1$ reflects a preference for a reward closer in the future.

At each step, the agent corrects its value with the error between the estimate for the
observed transition and the current estimate:

$$ \text{Error} = Q(s,a) - Q_\text{observed}(s, a)  $$

$$ Q(s,a) \leftarrow Q(s,a) + \alpha * \text{Error} $$

The learning rate $\alpha$ controls the size of the correction. A rate that is too large makes learning unstable; a rate that is too small makes it slow.

## Replay

The replay memory keeps past transitions. The agent relearns from a
random sample, which breaks the correlation between steps. The size of the
sample sets the number of transitions used.

This also helps propagate the value to earlier
states.

### What you can try

1. Increase epsilon and watch the agent explore new paths.
2. Lower the learning rate: the values stabilize more slowly.
3. Add some slip, then enable replay to stabilize learning.
