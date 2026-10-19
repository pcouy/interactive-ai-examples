import type { Transition } from './q-learning.js';

/** Constructor options for {@link ReplayBuffer}. */
export interface ReplayBufferOptions {
  /**
   * Maximum number of stored transitions. Default 1000. It must be a positive
   * integer.
   */
  capacity?: number;
  /**
   * Random source used to draw a batch. Default `Math.random`. It must return a
   * number in `[0, 1)`. Inject a seeded source for deterministic runs.
   */
  rng?: () => number;
}

/**
 * Fixed-capacity experience buffer.
 *
 * `store` appends a transition and evicts the oldest entry when the capacity is
 * full. `sample` returns a newest-anchored batch: the newest transition is
 * always last, and the rest are drawn uniformly WITHOUT replacement from the
 * older entries, so the batch length is `min(max(1, batchSize), size)`.
 *
 * Inject `rng` for deterministic runs. It must return a number in `[0, 1)`.
 */
export class ReplayBuffer<O, A> {
  private readonly capacityLimit: number;

  private readonly rng: () => number;

  private readonly entries: Transition<O, A>[] = [];

  constructor(options: ReplayBufferOptions = {}) {
    const capacity = options.capacity ?? 1000;
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error('ReplayBuffer capacity must be a positive integer.');
    }
    this.capacityLimit = capacity;
    this.rng = options.rng ?? Math.random;
  }

  get size(): number {
    return this.entries.length;
  }

  /** Maximum number of stored transitions. */
  get capacity(): number {
    return this.capacityLimit;
  }

  /** True when the buffer holds no transition. */
  get isEmpty(): boolean {
    return this.entries.length === 0;
  }

  store(transition: Transition<O, A>): void {
    this.entries.push(transition);
    while (this.entries.length > this.capacityLimit) {
      this.entries.shift();
    }
  }

  /**
   * Returns a batch that always ends with the newest transition. The other
   * entries are drawn uniformly without replacement from the older entries,
   * so the batch holds min(max(1, batchSize), size) entries. Sampling does
   * not mutate the buffer.
   */
  sample(batchSize: number): Transition<O, A>[] {
    const size = this.entries.length;
    if (size === 0) {
      return [];
    }
    const last = this.entries[size - 1];
    const poolLength = size - 1;
    const extra = Math.min(Math.max(1, batchSize) - 1, poolLength);
    const indices = Array.from({ length: poolLength }, (_, index) => index);
    const chosen: Transition<O, A>[] = [];
    // Partial Fisher-Yates: each swap fixes one distinct index in place.
    for (let position = 0; position < extra; position += 1) {
      const swap = position + Math.floor(this.rng() * (poolLength - position));
      [indices[position], indices[swap]] = [indices[swap], indices[position]];
      chosen.push(this.entries[indices[position]]);
    }
    return [...chosen, last];
  }

  clear(): void {
    this.entries.length = 0;
  }
}
