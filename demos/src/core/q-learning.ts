export interface Transition<O, A> {
  readonly observation: O;
  readonly action: A;
  readonly reward: number;
  readonly nextObservation: O;
  readonly done: boolean;
}

export interface QLearningOptions {
  learningRate?: number;
  discountFactor?: number;
  epsilon?: number;
  rng?: () => number;
}

const DEFAULT_LEARNING_RATE = 0.1;
const DEFAULT_DISCOUNT_FACTOR = 0.99;
const DEFAULT_EPSILON = 0.1;

export class QLearningAgent<O, A> {
  learningRate: number;

  discountFactor: number;

  epsilon: number;

  private readonly actions: readonly A[];

  private rng: () => number;

  private table: Map<O, Map<A, number>> = new Map();

  constructor(actions: readonly A[], options: QLearningOptions = {}) {
    this.actions = actions.slice();
    this.learningRate = options.learningRate ?? DEFAULT_LEARNING_RATE;
    this.discountFactor = options.discountFactor ?? DEFAULT_DISCOUNT_FACTOR;
    this.epsilon = options.epsilon ?? DEFAULT_EPSILON;
    this.rng = options.rng ?? Math.random;
  }

  selectAction(observation: O): A {
    if (this.rng() < this.epsilon) {
      const index = Math.floor(this.rng() * this.actions.length);
      return this.actions[index];
    }
    return this.greedyAction(observation);
  }

  update(transition: Transition<O, A>): void {
    const bootstrap = transition.done
      ? 0
      : this.discountFactor * this.maxValue(transition.nextObservation);
    const target = transition.reward + bootstrap;
    const current = this.valueOf(transition.observation, transition.action);
    const updated = current + this.learningRate * (target - current);
    const row = this.table.get(transition.observation);
    if (row) {
      row.set(transition.action, updated);
    } else {
      this.table.set(transition.observation, new Map([[transition.action, updated]]));
    }
  }

  valueOf(observation: O, action: A): number {
    return this.table.get(observation)?.get(action) ?? 0;
  }

  maxValue(observation: O): number {
    if (this.actions.length === 0) {
      return 0;
    }
    let best = -Infinity;
    for (const action of this.actions) {
      const value = this.valueOf(observation, action);
      if (value > best) {
        best = value;
      }
    }
    return best;
  }

  greedyAction(observation: O): A {
    let bestAction: A = this.actions[0];
    let bestValue = -Infinity;
    for (const action of this.actions) {
      const value = this.valueOf(observation, action);
      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }
    return bestAction;
  }

  get qTable(): ReadonlyMap<O, ReadonlyMap<A, number>> {
    return this.table;
  }

  reset(): void {
    this.table.clear();
  }
}
