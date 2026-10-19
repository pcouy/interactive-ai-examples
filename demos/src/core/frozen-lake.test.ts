import { describe, expect, it } from 'vitest';
import { FrozenLakeCore } from './frozen-lake.js';

describe('FrozenLakeCore', () => {
  it('starts from a zeroed state', () => {
    const core = new FrozenLakeCore();
    expect(core.ticksElapsed).toBe(0);
    expect(core.value).toBeCloseTo(0.5);
  });

  it('advances the value inside the unit range', () => {
    const core = new FrozenLakeCore();
    for (let i = 0; i < 12; i += 1) {
      core.tick();
    }
    expect(core.ticksElapsed).toBe(12);
    expect(core.value).toBeGreaterThanOrEqual(0);
    expect(core.value).toBeLessThanOrEqual(1);
  });

  it('resets back to the initial state', () => {
    const core = new FrozenLakeCore();
    core.tick();
    core.reset();
    expect(core.ticksElapsed).toBe(0);
    expect(core.value).toBeCloseTo(0.5);
  });
});
