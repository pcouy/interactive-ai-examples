import { describe, expect, it } from 'vitest';
import type { Transition } from './q-learning.js';
import { ReplayBuffer } from './replay-buffer.js';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeTransitions(count: number): Transition<number, string>[] {
  return Array.from({ length: count }, (_, index) => ({
    observation: index,
    action: `a${index}`,
    reward: index,
    nextObservation: index + 1,
    done: index === count - 1,
  }));
}

function fill(
  buffer: ReplayBuffer<number, string>,
  transitions: Transition<number, string>[],
): void {
  for (const transition of transitions) {
    buffer.store(transition);
  }
}

describe('ReplayBuffer storage', () => {
  it('evicts the oldest entries first when the capacity is exceeded', () => {
    const buffer = new ReplayBuffer<number, string>({ capacity: 3 });
    const transitions = makeTransitions(5);
    fill(buffer, transitions);
    expect(buffer.size).toBe(3);
    // The two oldest entries are gone, so the batch holds the last three
    // transitions, with the newest one last.
    const batch = buffer.sample(100);
    expect(batch).toHaveLength(3);
    expect(batch[2]).toBe(transitions[4]);
    expect(new Set(batch.slice(0, 2))).toEqual(new Set(transitions.slice(2, 4)));
  });

  it('keeps every entry while the size stays at or below the capacity', () => {
    const buffer = new ReplayBuffer<number, string>({ capacity: 4 });
    const transitions = makeTransitions(3);
    fill(buffer, transitions);
    expect(buffer.size).toBe(3);
    const batch = buffer.sample(100);
    expect(batch).toHaveLength(3);
    expect(batch[2]).toBe(transitions[2]);
    expect(new Set(batch.slice(0, 2))).toEqual(new Set(transitions.slice(0, 2)));
  });

  it('empties with clear and stores again afterwards', () => {
    const buffer = new ReplayBuffer<number, string>({ capacity: 2 });
    fill(buffer, makeTransitions(2));
    buffer.clear();
    expect(buffer.size).toBe(0);
    const [transition] = makeTransitions(1);
    buffer.store(transition);
    expect(buffer.size).toBe(1);
    expect(buffer.sample(1)).toEqual([transition]);
  });
});

describe('ReplayBuffer sampling', () => {
  it('returns an empty batch for an empty buffer', () => {
    const buffer = new ReplayBuffer<number, string>();
    expect(buffer.sample(1)).toEqual([]);
    expect(buffer.sample(8)).toEqual([]);
  });

  it('always ends the batch with the newest transition', () => {
    const buffer = new ReplayBuffer<number, string>({ capacity: 10 });
    const transitions = makeTransitions(6);
    fill(buffer, transitions);
    for (const batchSize of [1, 2, 3, 8, 64]) {
      expect(buffer.sample(batchSize).at(-1)).toBe(transitions[5]);
    }
  });

  it('returns only the newest transition for a batch size of 1', () => {
    const buffer = new ReplayBuffer<number, string>({ capacity: 10 });
    const transitions = makeTransitions(6);
    fill(buffer, transitions);
    expect(buffer.sample(1)).toEqual([transitions[5]]);
  });

  it('returns the whole buffer with the last entry last when the batch size exceeds the size', () => {
    const buffer = new ReplayBuffer<number, string>({ capacity: 10 });
    const transitions = makeTransitions(4);
    fill(buffer, transitions);
    const batch = buffer.sample(100);
    expect(batch).toHaveLength(4);
    expect(batch[3]).toBe(transitions[3]);
    // The other entries are the remaining transitions, each drawn once.
    expect(batch.slice(0, 3).sort((a, b) => a.observation - b.observation)).toEqual(
      transitions.slice(0, 3),
    );
  });

  it('draws distinct entries without replacement', () => {
    const buffer = new ReplayBuffer<number, string>({ capacity: 20, rng: mulberry32(3) });
    const transitions = makeTransitions(20);
    fill(buffer, transitions);
    const batch = buffer.sample(8);
    expect(batch).toHaveLength(8);
    const nonLast = batch.slice(0, -1);
    expect(new Set(nonLast).size).toBe(nonLast.length);
    for (const transition of nonLast) {
      expect(transition).not.toBe(transitions[19]);
    }
  });

  it('produces identical batches for identical seeds', () => {
    const run = (): Transition<number, string>[] => {
      const buffer = new ReplayBuffer<number, string>({ capacity: 30, rng: mulberry32(11) });
      fill(buffer, makeTransitions(20));
      return buffer.sample(10);
    };
    expect(run()).toEqual(run());
  });

  it('does not mutate the buffer while sampling', () => {
    const buffer = new ReplayBuffer<number, string>({ capacity: 10, rng: mulberry32(5) });
    fill(buffer, makeTransitions(8));
    buffer.sample(4);
    expect(buffer.size).toBe(8);
    expect(buffer.sample(100)).toHaveLength(8);
  });
});

describe('ReplayBuffer configuration', () => {
  it('exposes the configured capacity and defaults it to 1000', () => {
    expect(new ReplayBuffer<number, string>({ capacity: 7 }).capacity).toBe(7);
    expect(new ReplayBuffer<number, string>().capacity).toBe(1000);
  });

  it('reports isEmpty before and after the first store', () => {
    const buffer = new ReplayBuffer<number, string>({ capacity: 2 });
    expect(buffer.isEmpty).toBe(true);
    const [transition] = makeTransitions(1);
    buffer.store(transition);
    expect(buffer.isEmpty).toBe(false);
    buffer.clear();
    expect(buffer.isEmpty).toBe(true);
  });

  it('throws when the capacity is not a positive integer', () => {
    const message = 'ReplayBuffer capacity must be a positive integer.';
    expect(() => new ReplayBuffer<number, string>({ capacity: 0 })).toThrowError(message);
    expect(() => new ReplayBuffer<number, string>({ capacity: 2.5 })).toThrowError(message);
  });
});
