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
const TERMINALS_6X6 = [7, 9, 15, 18, 19, 27, 35];
const TERMINALS_8X8 = [19, 29, 35, 41, 42, 46, 49, 52, 54, 59, 63];

const CUSTOM_4X4 = ['SFFF', 'FFFF', 'FFFF', 'FFFG'];
const CUSTOM_6X6 = ['SHFFFF', 'FFFFFF', 'FFFFFF', 'FFFFFF', 'FFFFFF', 'FFFFFG'];

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
          desc: ['SFFFFF', 'FHFHFF', 'FFFHFF', 'HHFFFF', 'FFFHFF'],
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
});
