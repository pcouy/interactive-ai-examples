import { describe, expect, it } from 'vitest';
import {
  FROZEN_LAKE_ACTIONS,
  type FrozenLakeCell,
  FrozenLakeCore,
  FrozenLakeEnv,
} from './frozen-lake.js';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TERMINALS_4X4 = [5, 7, 11, 12, 15];
const TERMINALS_8X8 = [19, 29, 35, 41, 42, 46, 49, 52, 54, 59, 63];

function terminalCells(env: FrozenLakeEnv): number[] {
  const cells: number[] = [];
  for (let index = 0; index < env.size * env.size; index += 1) {
    if (env.isTerminal(index)) {
      cells.push(index);
    }
  }
  return cells;
}

function train(core: FrozenLakeCore, ticks: number): void {
  for (let index = 0; index < ticks; index += 1) {
    core.tick();
  }
}

function runUntilEpisodes(core: FrozenLakeCore, count: number, maxTicks: number): number {
  let ticks = 0;
  while (core.stats.episodes < count && ticks < maxTicks) {
    core.tick();
    ticks += 1;
  }
  return ticks;
}

describe('FrozenLakeEnv maps', () => {
  it('exposes the four gym actions', () => {
    expect(FROZEN_LAKE_ACTIONS).toEqual(['left', 'down', 'right', 'up']);
  });

  it('parses the 4x4 map into the gym layout', () => {
    const env = new FrozenLakeEnv({ size: 4 });
    const expected: FrozenLakeCell[] = [
      'start',
      'frozen',
      'frozen',
      'frozen',
      'frozen',
      'hole',
      'frozen',
      'hole',
      'frozen',
      'frozen',
      'frozen',
      'hole',
      'hole',
      'frozen',
      'frozen',
      'goal',
    ];
    expect(env.map).toHaveLength(16);
    expect([...env.map]).toEqual(expected);
    expect(env.cellAt(0)).toBe('start');
    expect(env.cellAt(1)).toBe('frozen');
    for (const index of [5, 7, 11, 12]) {
      expect(env.cellAt(index)).toBe('hole');
    }
    expect(env.cellAt(15)).toBe('goal');
    expect(terminalCells(env)).toEqual(TERMINALS_4X4);
  });

  it('parses the 8x8 map into the gym layout', () => {
    const env = new FrozenLakeEnv({ size: 8 });
    expect(env.map).toHaveLength(64);
    expect(env.cellAt(0)).toBe('start');
    expect(env.cellAt(63)).toBe('goal');
    expect(terminalCells(env)).toEqual(TERMINALS_8X8);
  });
});

describe('FrozenLakeEnv deterministic movement', () => {
  it('moves on the 4x4 map and clamps at walls', () => {
    const env = new FrozenLakeEnv({ size: 4, slippery: false });
    expect(env.reset()).toBe(0);
    expect(env.step('right')).toEqual({ observation: 1, reward: 0, done: false });

    const down = new FrozenLakeEnv({ size: 4, slippery: false });
    expect(down.step('down')).toEqual({ observation: 4, reward: 0, done: false });

    const leftAtWall = new FrozenLakeEnv({ size: 4, slippery: false });
    expect(leftAtWall.step('left')).toEqual({ observation: 0, reward: 0, done: false });

    const upAtWall = new FrozenLakeEnv({ size: 4, slippery: false });
    expect(upAtWall.step('up')).toEqual({ observation: 0, reward: 0, done: false });
  });

  it('lands on the hole at cell 12 after three steps down', () => {
    const env = new FrozenLakeEnv({ size: 4, slippery: false });
    env.step('down');
    env.step('down');
    expect(env.step('down')).toEqual({ observation: 12, reward: 0, done: true });

    expect(env.step('right')).toEqual({ observation: 12, reward: 0, done: true });
    expect(env.observation).toBe(12);
    expect(env.done).toBe(true);
  });

  it('reaches the goal from cell 13 with reward 1', () => {
    const env = new FrozenLakeEnv({ size: 4, slippery: false });
    for (const action of ['down', 'down', 'right', 'right', 'down', 'left'] as const) {
      expect(env.step(action).done).toBe(false);
    }
    expect(env.observation).toBe(13);

    expect(env.step('right')).toEqual({ observation: 14, reward: 0, done: false });
    expect(env.step('right')).toEqual({ observation: 15, reward: 1, done: true });
    expect(env.step('left')).toEqual({ observation: 15, reward: 0, done: true });
  });

  it('returns to the start cell on reset after a terminal step', () => {
    const env = new FrozenLakeEnv({ size: 4, slippery: false });
    env.step('down');
    env.step('down');
    expect(env.step('down').done).toBe(true);

    expect(env.reset()).toBe(0);
    expect(env.observation).toBe(0);
    expect(env.done).toBe(false);
    expect(env.step('right')).toEqual({ observation: 1, reward: 0, done: false });
  });
});

describe('FrozenLakeEnv slip', () => {
  it('follows the fixed slip outcomes for a down move at cell 0', () => {
    const intended = new FrozenLakeEnv({ size: 4, slippery: true, rng: () => 0 }).step('down');
    expect(intended.observation).toBe(4);

    const slipLeft = new FrozenLakeEnv({ size: 4, slippery: true, rng: () => 0.5 }).step('down');
    expect(slipLeft.observation).toBe(0);

    const slipRight = new FrozenLakeEnv({ size: 4, slippery: true, rng: () => 0.999 }).step('down');
    expect(slipRight.observation).toBe(1);
  });

  it('never leaves the grid over many slippery steps', () => {
    const env = new FrozenLakeEnv({ size: 4, slippery: true, rng: mulberry32(11) });
    for (let index = 0; index < 1000; index += 1) {
      const action = FROZEN_LAKE_ACTIONS[index % FROZEN_LAKE_ACTIONS.length];
      const before = env.observation;
      const result = env.step(action);
      expect(result.observation).toBeGreaterThanOrEqual(0);
      expect(result.observation).toBeLessThan(env.size * env.size);
      const rowDelta = Math.abs(
        Math.floor(result.observation / env.size) - Math.floor(before / env.size),
      );
      const columnDelta = Math.abs((result.observation % env.size) - (before % env.size));
      expect(rowDelta + columnDelta).toBeLessThanOrEqual(1);
      if (result.done) {
        env.reset();
      }
    }
  });
});

describe('FrozenLakeCore training', () => {
  it('counts episodes, successes and steps at episode boundaries', () => {
    const core = new FrozenLakeCore({ size: 4, slippery: false, epsilon: 1, seed: 3 });
    const ticks = runUntilEpisodes(core, 5, 100000);
    expect(ticks).toBeLessThan(100000);
    expect(core.stats.episodes).toBeGreaterThanOrEqual(5);
    expect(core.stats.steps).toBe(ticks);
    expect(core.stats.successes).toBeLessThanOrEqual(core.stats.episodes);
    expect(core.rewardHistory).toHaveLength(core.stats.episodes);
    const successes = core.rewardHistory.reduce((sum, reward) => sum + reward, 0);
    expect(successes).toBe(core.stats.successes);
    expect(core.stats.successRate).toBeCloseTo(core.stats.successes / core.stats.episodes);
    expect(core.observation).toBe(0);
  });

  it('trains to a success rate of at least 0.5', () => {
    const core = new FrozenLakeCore({
      size: 4,
      slippery: false,
      learningRate: 0.5,
      discountFactor: 0.95,
      epsilon: 0.1,
      seed: 42,
    });
    train(core, 30000);
    expect(core.stats.successRate).toBeGreaterThanOrEqual(0.5);
  });

  it('reproduces the same reward history for the same seed', () => {
    const run = (): number[] => {
      const core = new FrozenLakeCore({ size: 4, slippery: false, seed: 17 });
      train(core, 500);
      return [...core.rewardHistory];
    };
    expect(run()).toEqual(run());
  });

  it('keeps the training run identical when the policy is read between ticks', () => {
    const ticks = 300;
    const quiet = new FrozenLakeCore({
      size: 4,
      slippery: false,
      seed: 7,
      learningRate: 0.5,
      discountFactor: 0.95,
      epsilon: 0.1,
    });
    const reading = new FrozenLakeCore({
      size: 4,
      slippery: false,
      seed: 7,
      learningRate: 0.5,
      discountFactor: 0.95,
      epsilon: 0.1,
    });
    for (let index = 0; index < ticks; index += 1) {
      quiet.tick();
      reading.tick();
      const policy = reading.greedyActions;
      expect(policy).toHaveLength(16);
    }
    expect([...reading.rewardHistory]).toEqual([...quiet.rewardHistory]);
    expect(reading.stats.steps).toBe(quiet.stats.steps);
    expect(quiet.rewardHistory.length).toBeGreaterThan(0);
  });

  it('reproduces a fresh run after reset', () => {
    const make = (): FrozenLakeCore =>
      new FrozenLakeCore({
        size: 4,
        slippery: false,
        seed: 3,
        learningRate: 0.5,
        discountFactor: 0.95,
        epsilon: 0.1,
      });
    const ticks = 400;
    const fresh = make();
    const restarted = make();
    train(restarted, 100);
    expect(restarted.stats.steps).toBe(100);

    restarted.reset();
    train(fresh, ticks);
    train(restarted, ticks);
    expect([...restarted.rewardHistory]).toEqual([...fresh.rewardHistory]);
    expect(restarted.stats).toEqual(fresh.stats);
    expect(fresh.rewardHistory.length).toBeGreaterThan(0);
  });
});

describe('FrozenLakeCore reporting', () => {
  it('reports a zero success rate before any episode', () => {
    const core = new FrozenLakeCore({ size: 4 });
    expect(core.stats).toEqual({ episodes: 0, successes: 0, steps: 0, successRate: 0 });
  });

  it('reports greedy actions and values for every cell', () => {
    const core = new FrozenLakeCore({ size: 4, slippery: false, seed: 1 });
    train(core, 500);

    expect(core.greedyActions).toHaveLength(16);
    expect(core.greedyValues).toHaveLength(16);
    for (let index = 0; index < 16; index += 1) {
      if (TERMINALS_4X4.includes(index)) {
        expect(core.greedyActions[index]).toBeNull();
        expect(core.greedyValues[index]).toBe(0);
      } else {
        expect(FROZEN_LAKE_ACTIONS).toContain(core.greedyActions[index]);
        expect(Number.isFinite(core.greedyValues[index])).toBe(true);
      }
    }
  });

  it('records one reward per episode and caps the history at 300', () => {
    const core = new FrozenLakeCore({ size: 4, slippery: false, epsilon: 1, seed: 9 });

    // Under the cap the history grows by exactly one entry per episode.
    let ticks = 0;
    while (core.stats.episodes < 100 && ticks < 100000) {
      core.tick();
      ticks += 1;
      expect(core.rewardHistory.length).toBe(core.stats.episodes);
    }
    expect(core.stats.episodes).toBe(100);
    expect(core.rewardHistory).toHaveLength(100);

    // Past the cap the history keeps exactly the 300 most recent episodes.
    while (core.stats.episodes <= 300 && ticks < 200000) {
      core.tick();
      ticks += 1;
    }
    expect(core.stats.episodes).toBeGreaterThan(300);
    expect(core.rewardHistory).toHaveLength(300);
    for (const reward of core.rewardHistory) {
      expect(reward === 0 || reward === 1).toBe(true);
    }
  });

  it('forwards hyperparameters to the agent through mutable proxies', () => {
    const core = new FrozenLakeCore({
      size: 4,
      learningRate: 0.2,
      epsilon: 0.3,
      discountFactor: 0.8,
    });
    expect(core.learningRate).toBe(0.2);
    expect(core.epsilon).toBe(0.3);
    expect(core.discountFactor).toBe(0.8);

    core.learningRate = 0.6;
    core.epsilon = 0.05;
    core.discountFactor = 0.9;
    expect(core.learningRate).toBe(0.6);
    expect(core.epsilon).toBe(0.05);
    expect(core.discountFactor).toBe(0.9);
  });
});

describe('FrozenLakeCore configuration', () => {
  it('clears the q-table and the stats on reset', () => {
    const core = new FrozenLakeCore({ size: 4, slippery: false, seed: 8 });
    train(core, 200);

    core.reset();
    expect(core.stats).toEqual({ episodes: 0, successes: 0, steps: 0, successRate: 0 });
    expect(core.observation).toBe(0);
    expect(core.rewardHistory).toHaveLength(0);
    expect(core.greedyValues.every((value) => value === 0)).toBe(true);
    for (let index = 0; index < 16; index += 1) {
      if (TERMINALS_4X4.includes(index)) {
        expect(core.greedyActions[index]).toBeNull();
      } else {
        expect(FROZEN_LAKE_ACTIONS).toContain(core.greedyActions[index]);
      }
    }
  });

  it('rebuilds the environment when the size changes', () => {
    const core = new FrozenLakeCore({ size: 4, slippery: false, seed: 2 });
    train(core, 50);
    expect(core.stats.steps).toBe(50);

    core.setSize(8);
    expect(core.stats).toEqual({ episodes: 0, successes: 0, steps: 0, successRate: 0 });
    expect(core.observation).toBe(0);
    expect(core.rewardHistory).toHaveLength(0);
    expect(core.greedyActions).toHaveLength(64);
    expect(core.greedyValues).toHaveLength(64);
    for (const index of TERMINALS_8X8) {
      expect(core.greedyActions[index]).toBeNull();
      expect(core.greedyValues[index]).toBe(0);
    }
    expect(core.greedyActions[0]).not.toBeNull();
  });

  it('flips slip without resetting training', () => {
    const core = new FrozenLakeCore({ size: 4, slippery: false, epsilon: 1, seed: 4 });
    train(core, 100);
    const stats = { ...core.stats };
    const history = [...core.rewardHistory];

    core.setSlippery(true);
    core.setSlippery(false);
    expect(core.stats).toEqual(stats);
    expect([...core.rewardHistory]).toEqual(history);
  });
});
