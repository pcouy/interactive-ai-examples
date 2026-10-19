import { describe, expect, it } from 'vitest';
import {
  FROZEN_LAKE_ACTIONS,
  type FrozenLakeCell,
  FrozenLakeCore,
  FrozenLakeEnv,
} from './frozen-lake.js';
import { QLearningAgent } from './q-learning.js';

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
const TERMINALS_6X6 = [7, 15, 18, 19, 27, 35];
const TERMINALS_8X8 = [19, 29, 35, 41, 42, 46, 49, 52, 54, 59, 63];

const CUSTOM_4X4 = ['SFFF', 'FFFF', 'FFFF', 'FFFG'];
const CUSTOM_6X6 = ['SHFFFF', 'FFFFFF', 'FFFFFF', 'FFFFFF', 'FFFFFF', 'FFFFFG'];
const CUSTOM_WALL_4X4 = ['SWFF', 'FFFF', 'FFFF', 'FFFG'];
const CUSTOM_CANDY_4X4 = ['SCFF', 'FFFF', 'FFFF', 'FFFG'];
// Candy next to the start and a hole one step further: an episode can collect
// the candy and then fall in the hole, for a return strictly between 0 and 1.
const CUSTOM_CANDY_HOLE_4X4 = ['SCFF', 'FHFF', 'FFFF', 'FFFG'];

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

  it('parses the 6x6 template map', () => {
    const env = new FrozenLakeEnv({ size: 6 });
    expect(env.map).toHaveLength(36);
    expect(env.cellAt(0)).toBe('start');
    expect(env.cellAt(35)).toBe('goal');
    expect(terminalCells(env)).toEqual(TERMINALS_6X6);
    // The template showcases a wall at cell 9 and a candy at cell 22.
    expect(env.cellAt(9)).toBe('wall');
    expect(env.isWall(9)).toBe(true);
    expect(env.cellAt(22)).toBe('candy');
  });

  it('accepts a custom map through desc', () => {
    const env = new FrozenLakeEnv({ size: 4, desc: CUSTOM_4X4 });
    expect(env.map).toHaveLength(16);
    expect(env.cellAt(0)).toBe('start');
    expect(env.cellAt(15)).toBe('goal');
    // The default 4x4 map has a hole at cell 5, the custom map does not.
    expect(env.cellAt(5)).toBe('frozen');
    expect(terminalCells(env)).toEqual([15]);
  });

  it('throws when a map row has the wrong length', () => {
    expect(
      () => new FrozenLakeEnv({ size: 4, desc: ['SFF', 'FFFF', 'FFFF', 'FFFG'] }),
    ).toThrowError(/must have exactly 4 characters/);
  });

  it('throws when a map has the wrong number of rows', () => {
    // A deleted row from the 6x6 template keeps the row width but not the count.
    expect(
      () =>
        new FrozenLakeEnv({
          size: 6,
          desc: ['SFFFFF', 'FHFWFF', 'FFFHFF', 'HHFFCF', 'FFFHFF'],
        }),
    ).toThrowError(/must have exactly 6 rows/);
  });

  it('throws when a map has no start cell', () => {
    expect(
      () => new FrozenLakeEnv({ size: 4, desc: ['FFFF', 'FFFF', 'FFFF', 'FFFG'] }),
    ).toThrowError(/exactly one start cell/);
  });
});

describe('FrozenLakeEnv deterministic movement', () => {
  it('moves on the 4x4 map and clamps at walls', () => {
    const env = new FrozenLakeEnv({ size: 4, slipProbability: 0 });
    expect(env.reset()).toBe(0);
    expect(env.step('right')).toEqual({ observation: 1, reward: 0, done: false });

    const down = new FrozenLakeEnv({ size: 4, slipProbability: 0 });
    expect(down.step('down')).toEqual({ observation: 4, reward: 0, done: false });

    const leftAtWall = new FrozenLakeEnv({ size: 4, slipProbability: 0 });
    expect(leftAtWall.step('left')).toEqual({ observation: 0, reward: 0, done: false });

    const upAtWall = new FrozenLakeEnv({ size: 4, slipProbability: 0 });
    expect(upAtWall.step('up')).toEqual({ observation: 0, reward: 0, done: false });
  });

  it('lands on the hole at cell 12 after three steps down', () => {
    const env = new FrozenLakeEnv({ size: 4, slipProbability: 0 });
    env.step('down');
    env.step('down');
    expect(env.step('down')).toEqual({ observation: 12, reward: 0, done: true });

    expect(env.step('right')).toEqual({ observation: 12, reward: 0, done: true });
    expect(env.observation).toBe(12);
    expect(env.done).toBe(true);
  });

  it('reaches the goal from cell 13 with reward 1', () => {
    const env = new FrozenLakeEnv({ size: 4, slipProbability: 0 });
    for (const action of ['down', 'down', 'right', 'right', 'down', 'left'] as const) {
      expect(env.step(action).done).toBe(false);
    }
    expect(env.observation).toBe(13);

    expect(env.step('right')).toEqual({ observation: 14, reward: 0, done: false });
    expect(env.step('right')).toEqual({ observation: 15, reward: 1, done: true });
    expect(env.step('left')).toEqual({ observation: 15, reward: 0, done: true });
  });

  it('returns to the start cell on reset after a terminal step', () => {
    const env = new FrozenLakeEnv({ size: 4, slipProbability: 0 });
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
  it('is deterministic when the slip probability is 0', () => {
    const env = new FrozenLakeEnv({ size: 4, slipProbability: 0, rng: () => 0.5 });
    expect(env.step('down')).toEqual({ observation: 4, reward: 0, done: false });
    expect(env.step('down')).toEqual({ observation: 8, reward: 0, done: false });
  });

  it('always picks a perpendicular move when the slip probability is 1', () => {
    // 'down' at cell 0 slips to 'left' (cell 0) for the first draw and
    // 'right' (cell 1) for the second.
    const first = new FrozenLakeEnv({ size: 4, slipProbability: 1, rng: () => 0 }).step('down');
    expect(first.observation).toBe(0);

    const second = new FrozenLakeEnv({ size: 4, slipProbability: 1, rng: () => 0.999 }).step(
      'down',
    );
    expect(second.observation).toBe(1);
  });

  it('slips below the threshold and keeps the move above it', () => {
    const slips = new FrozenLakeEnv({ size: 4, slipProbability: 0.5, rng: () => 0.4 }).step('down');
    expect(slips.observation).toBe(0);

    const keeps = new FrozenLakeEnv({ size: 4, slipProbability: 0.5, rng: () => 0.6 }).step('down');
    expect(keeps.observation).toBe(4);
  });

  it('clamps the stored slip probability to [0, 1]', () => {
    const env = new FrozenLakeEnv({ size: 4, slipProbability: 2 });
    expect(env.slipProbability).toBe(1);
    env.slipProbability = -1;
    expect(env.slipProbability).toBe(0);
  });

  it('never leaves the grid over many slippery steps', () => {
    const env = new FrozenLakeEnv({ size: 4, slipProbability: 1, rng: mulberry32(11) });
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

  it('matches gym with slipProbability 2/3 over many steps', () => {
    // Each iteration starts from cell 0 and takes 'down'. The intended move
    // lands on cell 4, the left slip clamps to cell 0 and the right slip
    // lands on cell 1, so the three observations are the three outcomes.
    const env = new FrozenLakeEnv({ size: 4, slipProbability: 2 / 3, rng: mulberry32(11) });
    const steps = 6000;
    let intendedCount = 0;
    let leftCount = 0;
    let rightCount = 0;
    for (let index = 0; index < steps; index += 1) {
      env.reset();
      const result = env.step('down');
      if (result.observation === 4) {
        intendedCount += 1;
      } else if (result.observation === 0) {
        leftCount += 1;
      } else {
        rightCount += 1;
      }
    }
    expect(intendedCount + leftCount + rightCount).toBe(steps);
    expect(intendedCount / steps).toBeCloseTo(1 / 3, 1);
    expect(leftCount / steps).toBeCloseTo(1 / 3, 1);
    expect(rightCount / steps).toBeCloseTo(1 / 3, 1);
  });
});

describe('FrozenLakeEnv walls', () => {
  it('parses W into a wall cell that is not terminal', () => {
    const env = new FrozenLakeEnv({ size: 4, desc: CUSTOM_WALL_4X4 });
    expect(env.cellAt(1)).toBe('wall');
    expect(env.isTerminal(1)).toBe(false);
    expect(env.isWall(1)).toBe(true);
    expect(env.isWall(0)).toBe(false);
    expect(terminalCells(env)).toEqual([15]);
  });

  it('blocks a move into a wall like a grid edge', () => {
    const env = new FrozenLakeEnv({ size: 4, desc: CUSTOM_WALL_4X4, slipProbability: 0 });
    expect(env.step('right')).toEqual({ observation: 0, reward: 0, done: false });
    expect(env.observation).toBe(0);
    expect(env.done).toBe(false);

    // A normal move still works.
    expect(env.step('down')).toEqual({ observation: 4, reward: 0, done: false });
    expect(env.step('right')).toEqual({ observation: 5, reward: 0, done: false });
  });
});

describe('FrozenLakeEnv candy', () => {
  it('grants the candy reward on the visit and keeps the episode running', () => {
    const env = new FrozenLakeEnv({ size: 4, desc: CUSTOM_CANDY_4X4, candyReward: 0.25 });
    const result = env.step('right');
    expect(result).toEqual({ observation: 1, reward: 0.25, done: false });
    expect(env.cellAt(1)).toBe('candy');
    expect(env.isTerminal(1)).toBe(false);
    expect(env.done).toBe(false);
  });

  it('grants the candy reward again on every visit', () => {
    const env = new FrozenLakeEnv({ size: 4, desc: CUSTOM_CANDY_4X4, candyReward: 0.25 });
    expect(env.step('right').reward).toBe(0.25);
    expect(env.step('left').reward).toBe(0);
    expect(env.step('right').reward).toBe(0.25);
    expect(env.step('left').reward).toBe(0);
    expect(env.done).toBe(false);
  });

  it('defaults the candy reward to 0.1', () => {
    const env = new FrozenLakeEnv({ size: 4 });
    expect(env.candyReward).toBe(0.1);
    env.candyReward = 5;
    expect(env.candyReward).toBe(1);
  });
});

describe('FrozenLakeCore training', () => {
  it('counts episodes, successes and steps at episode boundaries', () => {
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, epsilon: 1, seed: 3 });
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

  it('counts a success only when the episode ends on the goal', () => {
    // Scan seeds for a run that contains a candy-only episode: the agent
    // collects the candy at cell 1 and then falls in the hole at cell 5, so
    // the episode return is strictly between 0 and 1.
    let seed = 0;
    for (let candidate = 1; candidate <= 50; candidate += 1) {
      const probe = new FrozenLakeCore({
        size: 4,
        desc: CUSTOM_CANDY_HOLE_4X4,
        epsilon: 1,
        seed: candidate,
      });
      train(probe, 200);
      if (probe.rewardHistory.some((reward) => reward > 0 && reward < 1)) {
        seed = candidate;
        break;
      }
    }
    expect(seed).toBeGreaterThan(0);

    const core = new FrozenLakeCore({ size: 4, desc: CUSTOM_CANDY_HOLE_4X4, epsilon: 1, seed });
    train(core, 200);
    const history = [...core.rewardHistory];
    // The run is not vacuous: it contains candy-only episodes.
    expect(history.some((reward) => reward > 0 && reward < 1)).toBe(true);
    // Goal episodes have a return of at least 1 and candy-only episodes have
    // a return below 1, so the successes match the returns of at least 1.
    expect(core.stats.successes).toBe(history.filter((reward) => reward >= 1).length);
  });

  it('trains to a success rate of at least 0.5', () => {
    const core = new FrozenLakeCore({
      size: 4,
      slipProbability: 0,
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
      const core = new FrozenLakeCore({ size: 4, slipProbability: 0, seed: 17 });
      train(core, 500);
      return [...core.rewardHistory];
    };
    expect(run()).toEqual(run());
  });

  it('keeps the training run identical when the policy is read between ticks', () => {
    const ticks = 300;
    const quiet = new FrozenLakeCore({
      size: 4,
      slipProbability: 0,
      seed: 7,
      learningRate: 0.5,
      discountFactor: 0.95,
      epsilon: 0.1,
    });
    const reading = new FrozenLakeCore({
      size: 4,
      slipProbability: 0,
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
        slipProbability: 0,
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
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, seed: 1 });
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

  it('reports the Q values of all four actions and the cell type per cell', () => {
    const core = new FrozenLakeCore({ size: 4, desc: CUSTOM_WALL_4X4, seed: 1 });
    train(core, 1000);
    const values = core.actionValues(0);
    expect(values).toHaveLength(FROZEN_LAKE_ACTIONS.length);
    for (const value of values) {
      expect(Number.isFinite(value)).toBe(true);
    }
    // After training the four action values of the start cell are not all equal.
    expect(new Set(values).size).toBeGreaterThan(1);

    expect(core.cellTypeAt(0)).toBe('start');
    expect(core.cellTypeAt(1)).toBe('wall');
    expect(core.cellTypeAt(15)).toBe('goal');
    // A wall has no policy, like a terminal cell.
    expect(core.greedyActions[1]).toBeNull();
    expect(core.greedyValues[1]).toBe(0);
  });

  it('records one reward per episode and caps the history at 300', () => {
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, epsilon: 1, seed: 9 });

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
      initialValue: 0.4,
    });
    expect(core.learningRate).toBe(0.2);
    expect(core.epsilon).toBe(0.3);
    expect(core.discountFactor).toBe(0.8);
    expect(core.initialValue).toBe(0.4);

    core.learningRate = 0.6;
    core.epsilon = 0.05;
    core.discountFactor = 0.9;
    core.initialValue = 0.7;
    expect(core.learningRate).toBe(0.6);
    expect(core.epsilon).toBe(0.05);
    expect(core.discountFactor).toBe(0.9);
    expect(core.initialValue).toBe(0.7);
  });

  it('keeps the initial value when training resets', () => {
    const core = new FrozenLakeCore({ size: 4, initialValue: 0.5, seed: 1 });
    train(core, 50);
    core.reset();
    expect(core.initialValue).toBe(0.5);
    expect(core.greedyValues.every((value) => value === 0.5 || value === 0)).toBe(true);
  });

  it('forwards the slip probability to the environment', () => {
    const run = (slipProbability: number): number[] => {
      const core = new FrozenLakeCore({ size: 4, slipProbability, epsilon: 1, seed: 5 });
      train(core, 500);
      return [...core.rewardHistory];
    };
    expect(run(1)).not.toEqual(run(0));
  });
});

describe('FrozenLakeCore configuration', () => {
  it('clears the q-table and the stats on reset', () => {
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, seed: 8 });
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
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, seed: 2 });
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

  it('trains on the 6x6 map', () => {
    const core = new FrozenLakeCore({ size: 6, slipProbability: 0, epsilon: 1, seed: 6 });
    train(core, 2000);
    expect(core.greedyActions).toHaveLength(36);
    expect(core.greedyValues).toHaveLength(36);
    for (const index of TERMINALS_6X6) {
      expect(core.greedyActions[index]).toBeNull();
      expect(core.greedyValues[index]).toBe(0);
    }
    // The wall at cell 9 has no policy, like a terminal cell.
    expect(core.greedyActions[9]).toBeNull();
    expect(core.greedyValues[9]).toBe(0);
    expect(core.stats.episodes).toBeGreaterThan(0);
  });

  it('uses a custom map set through the desc option', () => {
    const core = new FrozenLakeCore({ size: 4, desc: CUSTOM_4X4, seed: 1 });
    expect(core.greedyActions).toHaveLength(16);
    // Cell 5 is a hole in the default map but frozen in the custom map.
    expect(core.greedyActions[5]).not.toBeNull();
    expect(core.greedyActions[15]).toBeNull();
  });

  it('keeps a custom map across reset and clears it on setSize', () => {
    const core = new FrozenLakeCore({ size: 4, seed: 1 });
    core.setCustomMap(CUSTOM_6X6);
    expect(core.greedyActions).toHaveLength(36);
    // Cell 1 is a hole in the custom map but frozen in the 4x4 preset.
    expect(core.greedyActions[1]).toBeNull();

    core.reset();
    expect(core.greedyActions).toHaveLength(36);
    expect(core.greedyActions[1]).toBeNull();

    // setSize drops the custom map and restores the preset for the new size.
    core.setSize(4);
    expect(core.greedyActions).toHaveLength(16);
    // Cell 5 is a hole in the 4x4 preset but frozen in the custom map.
    expect(core.greedyActions[5]).toBeNull();
    expect(core.greedyActions[1]).not.toBeNull();

    core.setSize(8);
    expect(core.greedyActions).toHaveLength(64);
    expect(core.greedyActions[19]).toBeNull();
  });

  it('derives the grid size from a custom 6x6 map', () => {
    const core = new FrozenLakeCore({ size: 4, seed: 1 });
    core.setCustomMap(CUSTOM_6X6);
    expect(core.greedyActions).toHaveLength(36);
    expect(core.greedyValues).toHaveLength(36);
    // Cell 1 is a hole in the custom map but frozen in the 6x6 preset.
    expect(core.greedyActions[1]).toBeNull();
    expect(core.greedyActions[35]).toBeNull();
    expect(core.greedyActions[2]).not.toBeNull();
  });

  it('rejects an invalid custom map and keeps the current one', () => {
    const core = new FrozenLakeCore({ size: 4, seed: 1 });
    expect(() => core.setCustomMap(['SFF', 'FFFF', 'FFFF', 'FFFG'])).toThrowError(
      /must have exactly 4 characters/,
    );
    // The rejected map is not stored, so the default 4x4 map stays active.
    expect(core.greedyActions).toHaveLength(16);
    expect(core.greedyActions[5]).toBeNull();
  });

  it('changes the slip probability without resetting training', () => {
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, epsilon: 1, seed: 4 });
    train(core, 100);
    const stats = { ...core.stats };
    const history = [...core.rewardHistory];

    core.setSlipProbability(1);
    core.setSlipProbability(0);
    expect(core.stats).toEqual(stats);
    expect([...core.rewardHistory]).toEqual(history);
  });

  it('forwards the candy reward to the environment without resetting training', () => {
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, seed: 1 });
    expect(core.candyReward).toBe(0.1);
    train(core, 50);
    const stats = { ...core.stats };

    core.candyReward = 0.5;
    expect(core.candyReward).toBe(0.5);
    expect(core.stats).toEqual(stats);
    expect(core.stats.steps).toBe(stats.steps);
  });
});

describe('FrozenLakeCore replay', () => {
  it('defaults replay to disabled with a batch size of 8 and a capacity of 1000', () => {
    const core = new FrozenLakeCore({ size: 4, seed: 1 });
    expect(core.replayEnabled).toBe(false);
    expect(core.replayBatchSize).toBe(8);
    expect(core.replayCapacity).toBe(1000);
    expect(core.replayBufferSize).toBe(0);
  });

  it('clamps the batch size setter to whole values of at least 1', () => {
    const core = new FrozenLakeCore({ size: 4, seed: 1 });
    core.replayBatchSize = 0;
    expect(core.replayBatchSize).toBe(1);
    core.replayBatchSize = -3;
    expect(core.replayBatchSize).toBe(1);
    core.replayBatchSize = 2.6;
    expect(core.replayBatchSize).toBe(3);
  });

  it('returns an empty last batch while replay is disabled', () => {
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, seed: 5 });
    expect(core.lastReplayBatch).toEqual([]);
    train(core, 50);
    expect(core.lastReplayBatch).toEqual([]);
  });

  it('samples a batch of min(replayBatchSize, steps) transitions', () => {
    const core = new FrozenLakeCore({
      size: 4,
      slipProbability: 0,
      seed: 5,
      replayEnabled: true,
      replayBatchSize: 8,
    });
    train(core, 3);
    expect(core.lastReplayBatch).toHaveLength(3);
    expect(core.replayBufferSize).toBe(3);
    train(core, 7);
    expect(core.lastReplayBatch).toHaveLength(8);
    expect(core.replayBufferSize).toBe(10);
  });

  it('ends the last batch with the current tick transition', () => {
    const core = new FrozenLakeCore({
      size: 4,
      slipProbability: 0,
      seed: 5,
      replayEnabled: true,
      replayBatchSize: 8,
    });
    for (let index = 0; index < 10; index += 1) {
      const before = core.observation;
      core.tick();
      expect(core.lastReplayBatch.at(-1)?.observation).toBe(before);
      expect(core.lastReplayBatch.at(-1)?.nextObservation).toBe(core.observation);
    }
  });

  it('matches a hand-built online reference when replay is disabled', () => {
    const seed = 21;
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, seed });
    train(core, 100);

    // Replicates the tick loop with a standalone env and agent, seeded the
    // same way as the core: env on the seed, agent on the seed plus one.
    const env = new FrozenLakeEnv({ size: 4, slipProbability: 0, rng: mulberry32(seed) });
    const agent = new QLearningAgent<number, (typeof FROZEN_LAKE_ACTIONS)[number]>(
      FROZEN_LAKE_ACTIONS,
      { rng: mulberry32((seed + 1) >>> 0) },
    );
    const history: number[] = [];
    let episodeReturn = 0;
    for (let index = 0; index < 100; index += 1) {
      const observation = env.observation;
      const action = agent.selectAction(observation);
      const result = env.step(action);
      agent.update({
        observation,
        action,
        reward: result.reward,
        nextObservation: result.observation,
        done: result.done,
      });
      episodeReturn += result.reward;
      if (result.done) {
        history.push(episodeReturn);
        episodeReturn = 0;
        env.reset();
      }
    }
    expect([...core.rewardHistory]).toEqual(history);
  });

  it('trains differently when replay is enabled', () => {
    const disabled = new FrozenLakeCore({ size: 4, slipProbability: 0, seed: 16 });
    const enabled = new FrozenLakeCore({
      size: 4,
      slipProbability: 0,
      seed: 16,
      replayEnabled: true,
    });
    train(disabled, 100);
    train(enabled, 100);
    expect([...enabled.rewardHistory]).not.toEqual([...disabled.rewardHistory]);
  });

  it('reproduces the same reward history for the same seed when replay is enabled', () => {
    const run = (): number[] => {
      const core = new FrozenLakeCore({
        size: 4,
        slipProbability: 0,
        seed: 29,
        replayEnabled: true,
      });
      train(core, 100);
      return [...core.rewardHistory];
    };
    expect(run()).toEqual(run());
  });

  it('clears the buffer on reset, setSize and setCustomMap', () => {
    const core = new FrozenLakeCore({ size: 4, slipProbability: 0, seed: 2, replayEnabled: true });
    train(core, 50);
    expect(core.replayBufferSize).toBeGreaterThan(0);

    core.reset();
    expect(core.replayBufferSize).toBe(0);
    expect(core.lastReplayBatch).toEqual([]);

    train(core, 50);
    expect(core.replayBufferSize).toBeGreaterThan(0);
    core.setSize(8);
    expect(core.replayBufferSize).toBe(0);

    train(core, 50);
    expect(core.replayBufferSize).toBeGreaterThan(0);
    core.setCustomMap(CUSTOM_6X6);
    expect(core.replayBufferSize).toBe(0);
  });

  it('caps the buffer at the configured capacity', () => {
    const core = new FrozenLakeCore({
      size: 4,
      slipProbability: 0,
      seed: 2,
      replayEnabled: true,
      replayCapacity: 3,
    });
    expect(core.replayCapacity).toBe(3);
    train(core, 10);
    expect(core.replayBufferSize).toBe(3);
    // The batch still ends with the newest transition at capacity.
    expect(core.lastReplayBatch).toHaveLength(3);
    expect(core.lastReplayBatch[2].observation).toBeDefined();
  });
});
