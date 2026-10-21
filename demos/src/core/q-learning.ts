/**
 * One experience tuple for a tabular Q-learning update.
 *
 * Extensions wrap this type instead of changing it: a variant can add fields
 * such as `nextLegalActions` for algorithms that restrict the bootstrap to a
 * per-state legal action set.
 */
export interface Transition<O, A> {
  readonly observation: O;
  readonly action: A;
  readonly reward: number;
  readonly nextObservation: O;
  readonly done: boolean;
}

/**
 * Values of one {@link QLearningAgent.update}. `nextValue` is 0 at a terminal
 * transition, so `bootstrap` is 0 and `target` equals the reward.
 */
export interface QUpdateResult<O, A> {
  /** The transition that produced this update. */
  readonly transition: Transition<O, A>;
  /** Value of the update pair before the update. */
  readonly previousValue: number;
  /** Bootstrapped value of the next observation, or 0 when the transition is done. */
  readonly nextValue: number;
  /** Discount factor that was active during the update. */
  readonly discountFactor: number;
  /** `discountFactor * nextValue`. */
  readonly bootstrap: number;
  /** `transition.reward + bootstrap`. */
  readonly target: number;
  /** `target - previousValue`. */
  readonly tdError: number;
  /** Learning rate that was active during the update. */
  readonly learningRate: number;
  /** Value of the update pair after the update. It is also stored in the table. */
  readonly updatedValue: number;
}

/** Constructor options for {@link QLearningAgent}. */
export interface QLearningOptions {
  /**
   * Step size applied to the temporal-difference error. Default 0.1. The agent
   * does not clamp this value, so any real number is accepted.
   */
  learningRate?: number;
  /**
   * Discount factor applied to the bootstrapped next-state value. Default 0.99.
   * The agent does not clamp this value, so any real number is accepted.
   */
  discountFactor?: number;
  /**
   * Probability of exploring instead of acting greedily. Default 0.1. The agent
   * does not clamp this value, so it must be in [0, 1] for the usual behaviour.
   */
  epsilon?: number;
  /**
   * Value returned for an unseen observation-action pair. Default 0. The agent
   * does not clamp this value, so any real number is accepted.
   */
  initialValue?: number;
  /**
   * Random source used for exploration and greedy tie-breaks. Default
   * `Math.random`. It must return a number in `[0, 1)`. Inject a seeded source
   * for deterministic runs.
   */
  rng?: () => number;
}

const DEFAULT_LEARNING_RATE = 0.1;
const DEFAULT_DISCOUNT_FACTOR = 0.99;
const DEFAULT_EPSILON = 0.1;
const DEFAULT_INITIAL_VALUE = 0;

/**
 * Tabular Q-learning agent. The action space is fixed at construction, so a
 * state can use any subset of it.
 *
 * Per-state legal actions go through the optional `legalActions` parameter of
 * `selectAction`, `greedyAction`, `maxValue` and `policyAction`. When the
 * parameter is omitted, the whole action space is legal.
 *
 * Observation keys use object identity. A rebuilt observation object therefore
 * never matches an earlier one and accumulates a new row every time, so pass
 * stable keys: a number, a string, or one reused object per state.
 *
 * Inject `rng` for deterministic runs. The `update` bootstrap uses `maxValue`
 * over the full action space, not over a per-state legal set.
 */
export class QLearningAgent<O, A> {
  learningRate: number;

  discountFactor: number;

  epsilon: number;

  initialValue: number;

  private readonly actions: readonly A[];

  private rng: () => number;

  private table: Map<O, Map<A, number>> = new Map();

  constructor(actions: readonly A[], options: QLearningOptions = {}) {
    if (actions.length === 0) {
      throw new Error('QLearningAgent needs at least one action.');
    }
    this.actions = actions.slice();
    this.learningRate = options.learningRate ?? DEFAULT_LEARNING_RATE;
    this.discountFactor = options.discountFactor ?? DEFAULT_DISCOUNT_FACTOR;
    this.epsilon = options.epsilon ?? DEFAULT_EPSILON;
    this.initialValue = options.initialValue ?? DEFAULT_INITIAL_VALUE;
    this.rng = options.rng ?? Math.random;
  }

  /**
   * Returns the legal action set for a call: the explicit `legalActions` when
   * given, otherwise the full action space. Throws when the set is empty.
   */
  private legalFrom(legalActions?: readonly A[]): readonly A[] {
    const actions = legalActions ?? this.actions;
    if (actions.length === 0) {
      throw new Error('QLearningAgent needs at least one legal action.');
    }
    return actions;
  }

  /**
   * Picks an action for `observation`. With probability `epsilon` it explores:
   * it returns a uniformly random action from the legal set. Otherwise it
   * returns `greedyAction`, which can consume an rng draw to break a tie.
   */
  selectAction(observation: O, legalActions?: readonly A[]): A {
    const actions = this.legalFrom(legalActions);
    if (this.rng() < this.epsilon) {
      const index = Math.floor(this.rng() * actions.length);
      return actions[index];
    }
    return this.greedyAction(observation, actions);
  }

  /**
   * Applies one Q-learning update and returns its intermediate values.
   *
   * The action must be in the agent's action space. The bootstrap uses
   * `maxValue` over the full action space. Pass `done: true` at a terminal
   * transition so the bootstrap is 0.
   */
  update(transition: Transition<O, A>): QUpdateResult<O, A> {
    const nextValue = transition.done ? 0 : this.maxValue(transition.nextObservation);
    // Read both rates once so a caller that changes them mid-update cannot
    // produce a result whose parts disagree.
    const discountFactor = this.discountFactor;
    const learningRate = this.learningRate;
    const bootstrap = discountFactor * nextValue;
    const target = transition.reward + bootstrap;
    const previousValue = this.valueOf(transition.observation, transition.action);
    const tdError = target - previousValue;
    const updatedValue = previousValue + learningRate * tdError;
    const row = this.table.get(transition.observation);
    if (row) {
      row.set(transition.action, updatedValue);
    } else {
      this.table.set(transition.observation, new Map([[transition.action, updatedValue]]));
    }
    return {
      transition,
      previousValue,
      nextValue,
      discountFactor,
      bootstrap,
      target,
      tdError,
      learningRate,
      updatedValue,
    };
  }

  /** Current value of `observation` and `action`, or `initialValue` if unseen. */
  valueOf(observation: O, action: A): number {
    return this.table.get(observation)?.get(action) ?? this.initialValue;
  }

  /**
   * Highest value over the legal set. The legal set defaults to the full action
   * space and must not be empty.
   */
  maxValue(observation: O, legalActions?: readonly A[]): number {
    const actions = this.legalFrom(legalActions);
    let best = -Infinity;
    for (const action of actions) {
      const value = this.valueOf(observation, action);
      if (value > best) {
        best = value;
      }
    }
    return best;
  }

  /**
   * Returns a greedy action for `observation`.
   * Every legal action whose value equals the maximal value is a candidate.
   * The candidates are tied, so one is picked uniformly at random with
   * `this.rng()`. It consumes one training rng draw when the top value is tied.
   * Use `policyAction` to inspect the policy without changing training.
   */
  greedyAction(observation: O, legalActions?: readonly A[]): A {
    const actions = this.legalFrom(legalActions);
    const bestValue = this.maxValue(observation, actions);
    const tied: A[] = [];
    for (const action of actions) {
      if (this.valueOf(observation, action) === bestValue) {
        tied.push(action);
      }
    }
    // A lone candidate is forced, so it is returned without consuming a draw.
    if (tied.length === 1) {
      return tied[0];
    }
    return tied[Math.floor(this.rng() * tied.length)];
  }

  /**
   * Greedy action for `observation` over the legal set with a deterministic
   * tie-break: the first action in the set wins. Does not consume the rng.
   * For display.
   */
  policyAction(observation: O, legalActions?: readonly A[]): A {
    const actions = this.legalFrom(legalActions);
    let bestAction: A = actions[0];
    let bestValue = -Infinity;
    for (const action of actions) {
      const value = this.valueOf(observation, action);
      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }
    return bestAction;
  }

  /** Read-only view of the Q table, keyed by observation identity. */
  get qTable(): ReadonlyMap<O, ReadonlyMap<A, number>> {
    return this.table;
  }

  /** Clears every learned value. The public options keep their values. */
  reset(): void {
    this.table.clear();
  }
}
