import { describe, expect, it } from 'vitest';
import { QLearningAgent, type Transition } from './q-learning.js';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rngFromSequence(values: number[]): () => number {
  let cursor = 0;
  return (): number => values[cursor++ % values.length];
}

describe('QLearningAgent action selection', () => {
  it('selectAction returns the greedy action when epsilon is 0', () => {
    const agent = new QLearningAgent<string, number>([0, 1], { epsilon: 0, rng: () => 1 });
    agent.update({ observation: 's', action: 1, reward: 1, nextObservation: 's', done: true });
    expect(agent.selectAction('s')).toBe(1);
  });

  it('selectAction explores when epsilon is 1 and follows the injected rng', () => {
    const actions = ['left', 'right', 'jump'] as const;
    // Each call consumes one draw for the epsilon test, then one draw for the index.
    const draws = [0, 0, 0, 0.5, 0, 0.999999];
    const agent = new QLearningAgent<string, (typeof actions)[number]>(actions, {
      epsilon: 1,
      rng: rngFromSequence(draws),
    });
    expect(agent.selectAction('s')).toBe('left'); // floor(0 * 3) = 0
    expect(agent.selectAction('s')).toBe('right'); // floor(0.5 * 3) = 1
    expect(agent.selectAction('s')).toBe('jump'); // floor(0.999999 * 3) = 2
  });

  it('selectAction with epsilon 0 matches greedyAction once one action is strictly best', () => {
    const make = (): QLearningAgent<string, number> =>
      new QLearningAgent<string, number>([0, 1, 2], { epsilon: 0, rng: mulberry32(7) });
    const a = make();
    const b = make();
    const observations = ['s0', 's1', 's2'];
    for (const observation of observations) {
      const transition: Transition<string, number> = {
        observation,
        action: 0,
        reward: 1,
        nextObservation: observation,
        done: true,
      };
      a.update(transition);
      b.update(transition);
    }
    for (let step = 0; step < 20; step += 1) {
      const observation = observations[step % observations.length];
      expect(a.selectAction(observation)).toBe(b.greedyAction(observation));
    }
  });
});

describe('QLearningAgent update', () => {
  it('applies the exact terminal update', () => {
    const agent = new QLearningAgent<string, number>([0, 1], { learningRate: 0.5 });
    const transition: Transition<string, number> = {
      observation: 's',
      action: 0,
      reward: 1,
      nextObservation: 'terminal',
      done: true,
    };
    agent.update(transition);
    expect(agent.valueOf('s', 0)).toBeCloseTo(0.5);
  });

  it('bootstraps from the next observation value when done is false', () => {
    const agent = new QLearningAgent<string, number>([0, 1], {
      learningRate: 0.25,
      discountFactor: 0.8,
    });
    agent.update({ observation: 's', action: 1, reward: 1, nextObservation: 't', done: true });
    expect(agent.valueOf('s', 1)).toBeCloseTo(0.25);
    expect(agent.maxValue('t')).toBe(0);

    // target = 0.5 + 0.8 * maxValue('s') = 0.5 + 0.8 * 0.25 = 0.7
    // Q('s', 0) = 0 + 0.25 * (0.7 - 0) = 0.175
    agent.update({ observation: 's', action: 0, reward: 0.5, nextObservation: 's', done: false });
    expect(agent.valueOf('s', 0)).toBeCloseTo(0.175);
  });
});

describe('QLearningAgent update result', () => {
  it('reports the exact values of a terminal update', () => {
    const agent = new QLearningAgent<string, number>([0, 1], {
      learningRate: 0.5,
      discountFactor: 0.9,
      initialValue: 0.2,
    });
    const transition: Transition<string, number> = {
      observation: 's',
      action: 0,
      reward: 1,
      nextObservation: 'terminal',
      done: true,
    };

    const result = agent.update(transition);

    expect(result.transition).toBe(transition);
    expect(result.previousValue).toBeCloseTo(0.2);
    expect(result.nextValue).toBe(0);
    expect(result.discountFactor).toBe(0.9);
    expect(result.bootstrap).toBe(0);
    expect(result.target).toBeCloseTo(1);
    expect(result.tdError).toBeCloseTo(0.8);
    expect(result.learningRate).toBe(0.5);
    expect(result.updatedValue).toBeCloseTo(0.6);
    expect(result.updatedValue).toBeCloseTo(agent.valueOf('s', 0));
  });

  it('reports the exact values of a bootstrapped update', () => {
    const agent = new QLearningAgent<string, number>([0, 1], {
      learningRate: 0.25,
      discountFactor: 0.8,
    });
    // Seed the next state with maxValue 0.25.
    agent.update({ observation: 'next', action: 0, reward: 1, nextObservation: 't', done: true });
    const transition: Transition<string, number> = {
      observation: 's',
      action: 1,
      reward: 0.5,
      nextObservation: 'next',
      done: false,
    };

    const result = agent.update(transition);

    expect(result.transition).toBe(transition);
    expect(result.previousValue).toBe(0);
    expect(result.nextValue).toBeCloseTo(0.25);
    expect(result.discountFactor).toBe(0.8);
    expect(result.bootstrap).toBeCloseTo(0.2);
    expect(result.target).toBeCloseTo(0.7);
    expect(result.tdError).toBeCloseTo(0.7);
    expect(result.learningRate).toBe(0.25);
    expect(result.updatedValue).toBeCloseTo(0.175);
    expect(result.updatedValue).toBeCloseTo(agent.valueOf('s', 1));
  });

  it('keeps the rates that were active during the update', () => {
    const agent = new QLearningAgent<string, number>([0], {
      learningRate: 0.1,
      discountFactor: 0.9,
    });
    const result = agent.update({
      observation: 's',
      action: 0,
      reward: 1,
      nextObservation: 't',
      done: true,
    });

    agent.learningRate = 0.9;
    agent.discountFactor = 0.1;

    expect(result.learningRate).toBe(0.1);
    expect(result.discountFactor).toBe(0.9);
    expect(result.updatedValue).toBeCloseTo(0.1);
    expect(agent.valueOf('s', 0)).toBeCloseTo(0.1);
  });
});

describe('QLearningAgent initial value', () => {
  it('reads the initial value for an unseen pair', () => {
    const agent = new QLearningAgent<string, string>(['first', 'second'], { initialValue: 0.5 });
    expect(agent.valueOf('unknown', 'first')).toBe(0.5);
    expect(agent.maxValue('unknown')).toBe(0.5);
  });

  it('uses the initial value in the update math', () => {
    const agent = new QLearningAgent<string, number>([0, 1], {
      learningRate: 0.5,
      initialValue: 0.5,
    });
    // target = 1, current = 0.5, updated = 0.5 + 0.5 * (1 - 0.5) = 0.75
    agent.update({ observation: 's', action: 0, reward: 1, nextObservation: 't', done: true });
    expect(agent.valueOf('s', 0)).toBeCloseTo(0.75);
  });

  it('lets the initial value change for unseen pairs', () => {
    const agent = new QLearningAgent<string, string>(['first'], { initialValue: 0.5 });
    agent.initialValue = -1;
    expect(agent.valueOf('unknown', 'first')).toBe(-1);
    expect(agent.maxValue('unknown')).toBe(-1);
  });

  it('keeps the learned value when the initial value changes', () => {
    const agent = new QLearningAgent<string, number>([0], { learningRate: 1, initialValue: 0.5 });
    agent.update({ observation: 's', action: 0, reward: 1, nextObservation: 't', done: true });
    agent.initialValue = 5;
    expect(agent.valueOf('s', 0)).toBeCloseTo(1);
    expect(agent.valueOf('unknown', 0)).toBe(5);
  });
});

describe('QLearningAgent table access', () => {
  it('returns 0 for unseen pairs', () => {
    const agent = new QLearningAgent<string, string>(['first', 'second']);
    expect(agent.valueOf('unknown', 'first')).toBe(0);
    expect(agent.maxValue('unknown')).toBe(0);
    expect(agent.qTable.size).toBe(0);
  });

  it('breaks an empty-table tie towards the first action when the rng is 0', () => {
    const agent = new QLearningAgent<string, string>(['first', 'second'], { rng: () => 0 });
    expect(agent.greedyAction('unknown')).toBe('first');
  });

  it('breaks an empty-table tie towards the last action when the rng is near 1', () => {
    const agent = new QLearningAgent<string, string>(['first', 'second'], {
      rng: () => 0.999999,
    });
    expect(agent.greedyAction('unknown')).toBe('second');
  });

  it('returns the strictly best action regardless of the rng', () => {
    const agent = new QLearningAgent<string, number>([0, 1, 2], { rng: () => 0.999999 });
    agent.update({ observation: 's', action: 1, reward: 1, nextObservation: 's', done: true });
    expect(agent.greedyAction('s')).toBe(1);
  });

  it('breaks a partial tie uniformly at random', () => {
    const agent = new QLearningAgent<string, number>([0, 1, 2], { rng: () => 0.999999 });
    agent.update({ observation: 's', action: 2, reward: 1, nextObservation: 's', done: true });
    agent.update({ observation: 's', action: 1, reward: 1, nextObservation: 's', done: true });
    expect(agent.greedyAction('s')).toBe(2);
  });

  it('reset clears the table', () => {
    const agent = new QLearningAgent<string, number>([0, 1]);
    agent.update({ observation: 's', action: 0, reward: 1, nextObservation: 's', done: true });
    expect(agent.qTable.size).toBe(1);
    agent.reset();
    expect(agent.qTable.size).toBe(0);
    expect(agent.valueOf('s', 0)).toBe(0);
  });

  it('keeps the table when a public option changes', () => {
    const agent = new QLearningAgent<string, number>([0, 1]);
    agent.update({ observation: 's', action: 0, reward: 1, nextObservation: 's', done: true });
    agent.learningRate = 0.9;
    agent.discountFactor = 0.5;
    agent.epsilon = 0.7;
    expect(agent.qTable.size).toBe(1);
    expect(agent.valueOf('s', 0)).toBeCloseTo(0.1);
  });

  it('uses object identity for keys', () => {
    const first = { id: 1 };
    const sameShape = { id: 1 };
    const agent = new QLearningAgent<{ id: number }, string>(['go'], { learningRate: 1 });
    agent.update({
      observation: first,
      action: 'go',
      reward: 1,
      nextObservation: first,
      done: true,
    });
    expect(agent.valueOf(first, 'go')).toBeCloseTo(1);
    expect(agent.valueOf(sameShape, 'go')).toBe(0);
    expect(agent.qTable.has(first)).toBe(true);
    expect(agent.qTable.has(sameShape)).toBe(false);
  });
});

describe('QLearningAgent construction', () => {
  it('throws on an empty action space', () => {
    expect(() => new QLearningAgent<string, number>([])).toThrowError(
      'QLearningAgent needs at least one action.',
    );
  });
});

describe('QLearningAgent policy action', () => {
  it('breaks an empty-table tie towards the first action every time', () => {
    const agent = new QLearningAgent<string, string>(['first', 'second', 'third'], {
      rng: () => 0.999999,
    });
    for (let step = 0; step < 10; step += 1) {
      expect(agent.policyAction('unknown')).toBe('first');
    }
  });

  it('does not consume the training rng', () => {
    const make = (): QLearningAgent<string, number> =>
      new QLearningAgent<string, number>([0, 1, 2], { epsilon: 0.5, rng: mulberry32(99) });
    const plain = make();
    const reading = make();
    const observations = ['s0', 's1', 's2'];
    for (let step = 0; step < 50; step += 1) {
      const observation = observations[step % observations.length];
      reading.policyAction(observation);
      expect(reading.selectAction(observation)).toBe(plain.selectAction(observation));
    }
  });
});

describe('QLearningAgent determinism', () => {
  it('produces the same action sequence for the same seed', () => {
    const make = (): QLearningAgent<string, number> =>
      new QLearningAgent<string, number>([0, 1], { epsilon: 0.5, rng: mulberry32(1234) });
    const a = make();
    const b = make();
    const observations = ['s0', 's1', 's2'];
    for (let step = 0; step < 50; step += 1) {
      const observation = observations[step % observations.length];
      expect(a.selectAction(observation)).toBe(b.selectAction(observation));
    }
  });
});

describe('QLearningAgent legal actions', () => {
  it('explores only within the legal set when epsilon is 1', () => {
    const actions = ['left', 'right', 'jump'] as const;
    const legal = ['left', 'right'] as const;
    // Each call consumes one draw for the epsilon test, then one for the index.
    const agent = new QLearningAgent<string, (typeof actions)[number]>(actions, {
      epsilon: 1,
      rng: rngFromSequence([0, 0, 0, 0.999999]),
    });
    for (let step = 0; step < 10; step += 1) {
      const chosen = agent.selectAction('s', legal);
      expect(legal).toContain(chosen);
      expect(chosen).not.toBe('jump');
    }
  });

  it('ignores a higher value outside the legal set', () => {
    const agent = new QLearningAgent<string, number>([0, 1, 2], { learningRate: 1 });
    agent.update({ observation: 's', action: 2, reward: 10, nextObservation: 's', done: true });
    expect(agent.maxValue('s')).toBeCloseTo(10);
    expect(agent.maxValue('s', [0, 1])).toBe(0);
  });

  it('makes greedyAction respect the legal set', () => {
    const agent = new QLearningAgent<string, number>([0, 1, 2], {
      learningRate: 1,
      rng: () => 0,
    });
    agent.update({ observation: 's', action: 2, reward: 10, nextObservation: 's', done: true });
    // Actions 0 and 1 are both unseen and tied, so the rng of 0 picks action 0.
    expect(agent.greedyAction('s', [0, 1])).toBe(0);
    expect(agent.greedyAction('s')).toBe(2);
  });

  it('makes policyAction respect the legal set', () => {
    const agent = new QLearningAgent<string, number>([0, 1, 2], { learningRate: 1 });
    agent.update({ observation: 's', action: 2, reward: 10, nextObservation: 's', done: true });
    // The legal order decides the deterministic tie-break.
    expect(agent.policyAction('s', [1, 0])).toBe(1);
    expect(agent.policyAction('s')).toBe(2);
  });

  it('throws for an empty legal set on every selection method', () => {
    const agent = new QLearningAgent<string, number>([0, 1]);
    const message = 'QLearningAgent needs at least one legal action.';
    expect(() => agent.selectAction('s', [])).toThrowError(message);
    expect(() => agent.greedyAction('s', [])).toThrowError(message);
    expect(() => agent.maxValue('s', [])).toThrowError(message);
    expect(() => agent.policyAction('s', [])).toThrowError(message);
  });
});
