import type { Transition } from './q-learning.js';

export interface ReplayBufferOptions {
  capacity?: number; // default 1000
  rng?: () => number; // default Math.random
}

export class ReplayBuffer<O, A> {
  private readonly capacity: number;

  private readonly rng: () => number;

  private readonly entries: Transition<O, A>[] = [];

  constructor(options: ReplayBufferOptions = {}) {
    this.capacity = options.capacity ?? 1000;
    this.rng = options.rng ?? Math.random;
  }

  get size(): number {
    return this.entries.length;
  }

  store(transition: Transition<O, A>): void {
    this.entries.push(transition);
    while (this.entries.length > this.capacity) {
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
