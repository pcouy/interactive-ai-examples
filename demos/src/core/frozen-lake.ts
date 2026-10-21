import { QLearningAgent, type QUpdateResult, type Transition } from './q-learning.js';
import { ReplayBuffer } from './replay-buffer.js';

export type FrozenLakeAction = 'left' | 'down' | 'right' | 'up';

export type FrozenLakeCell = 'start' | 'frozen' | 'hole' | 'goal' | 'wall' | 'candy';

export const FROZEN_LAKE_ACTIONS: readonly FrozenLakeAction[] = ['left', 'down', 'right', 'up'];

/**
 * Phase of the fine-step agent-environment loop. One full step runs `observe`,
 * then `act`, then `transition`.
 */
export type FineStepPhase = 'observe' | 'act' | 'transition';

/**
 * Phases of one full fine step, in order. The names match {@link FineStepPhase}
 * so a panel can list them without re-declaring the order.
 */
export const FINE_STEP_PHASES: readonly FineStepPhase[] = ['observe', 'act', 'transition'];

/** Visible state of one {@link FrozenLakeCore} fine step. */
export interface FineStepState {
  /** Phase the loop is in. The next {@link FrozenLakeCore.advanceFineStep} call runs it. */
  readonly phase: FineStepPhase;
  /** Phase that follows {@link FineStepState.phase}. The next advanceFineStep call enters it. */
  readonly nextPhase: FineStepPhase;
  /** Observation the agent acts on in this step. */
  readonly observation: number;
  /** Selected action, or null before the observe phase and after the transition phase. */
  readonly action: FrozenLakeAction | null;
  /** Environment observation after the action, or null outside the act/transition phases. */
  readonly nextObservation: number | null;
  /** Reward of the step, or null outside the act/transition phases. */
  readonly reward: number | null;
  /** True when the step ended the episode, or null outside the act/transition phases. */
  readonly done: boolean | null;
}

function nextFineStepPhase(phase: FineStepPhase): FineStepPhase {
  switch (phase) {
    case 'observe':
      return 'act';
    case 'act':
      return 'transition';
    case 'transition':
      return 'observe';
  }
}

const DEFAULT_SEED = 42;
export const DEFAULT_CANDY_REWARD = 0.1;
const REWARD_HISTORY_LIMIT = 300;

export type FrozenLakeSize = 4 | 6 | 8;

const MAPS: Readonly<Record<FrozenLakeSize, readonly string[]>> = {
  4: ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
  // User-editable template: change these rows to build your own 6x6 lake.
  // The template showcases a wall (W) and a candy (C). Every row must have
  // exactly 6 characters and the map needs exactly one 'S'.
  6: ['SFFFFF', 'FHFWFF', 'FFFHFF', 'HHFFCF', 'FFFHFF', 'FFFFFG'],
  8: [
    'SFFFFFFF',
    'FFFFFFFF',
    'FFFHFFFF',
    'FFFFFHFF',
    'FFFHFFFF',
    'FHHFFFHF',
    'FHFFHFHF',
    'FFFHFFFG',
  ],
};

// Perpendicular outcomes for a slip: [first, second]. The intended action is
// not listed here, so a slip never repeats the intended move.
const SLIP_OUTCOMES: Readonly<
  Record<FrozenLakeAction, readonly [FrozenLakeAction, FrozenLakeAction]>
> = {
  left: ['up', 'down'],
  down: ['left', 'right'],
  right: ['up', 'down'],
  up: ['left', 'right'],
};

/** Clamps a probability-like value to the closed range [0, 1]. */
function clampUnit(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

/**
 * Returns the grid size for a set of map rows. The environment supports only
 * the sizes 4, 6 and 8, so any other row count throws.
 */
export function sizeFromDesc(rows: readonly string[]): FrozenLakeSize {
  const size = rows.length;
  if (size === 4 || size === 6 || size === 8) {
    return size;
  }
  throw new Error(`Frozen Lake map needs 4, 6 or 8 rows, got ${size}.`);
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Map symbols: S start, H hole, G goal, W wall, C candy, anything else frozen.
function cellFromSymbol(symbol: string): FrozenLakeCell {
  if (symbol === 'S') {
    return 'start';
  }
  if (symbol === 'H') {
    return 'hole';
  }
  if (symbol === 'G') {
    return 'goal';
  }
  if (symbol === 'W') {
    return 'wall';
  }
  if (symbol === 'C') {
    return 'candy';
  }
  return 'frozen';
}

function cellsFromMap(
  size: FrozenLakeSize,
  rows: readonly string[] = MAPS[size],
): FrozenLakeCell[] {
  const cells: FrozenLakeCell[] = [];
  let starts = 0;
  for (const row of rows) {
    if (row.length !== size) {
      throw new Error(
        `Frozen Lake map row "${row}" must have exactly ${size} characters, got ${row.length}.`,
      );
    }
    for (const symbol of row) {
      if (symbol === 'S') {
        starts += 1;
      }
      cells.push(cellFromSymbol(symbol));
    }
  }
  if (starts !== 1) {
    throw new Error(`Frozen Lake map needs exactly one start cell "S", found ${starts}.`);
  }
  if (rows.length !== size) {
    throw new Error(`Frozen Lake map must have exactly ${size} rows, got ${rows.length}.`);
  }
  return cells;
}

/** Result of one {@link FrozenLakeEnv.step}. */
export interface FrozenLakeStep {
  /** Index of the cell the agent occupies after the step. */
  readonly observation: number;
  /** Reward granted by the step. */
  readonly reward: number;
  /** True when the step ended the episode on a hole or the goal. */
  readonly done: boolean;
}

/** Constructor options for {@link FrozenLakeEnv}. */
export interface FrozenLakeEnvOptions {
  /**
   * Grid size. Default 4. Only 4, 6 and 8 are supported. When `desc` is given
   * and this field is omitted, the size is derived from the row count.
   */
  size?: FrozenLakeSize;
  /**
   * Probability that the intended move is replaced by a uniformly random
   * perpendicular move. Default 0. The setter clamps the stored value to [0, 1].
   */
  slipProbability?: number;
  /**
   * Reward collected on every candy visit. Default {@link DEFAULT_CANDY_REWARD}.
   * The setter clamps the stored value to [0, 1].
   */
  candyReward?: number;
  /**
   * Custom map rows. When given without `size`, the row count defines the grid
   * size. Every row must match the width and the map needs exactly one 'S'.
   */
  desc?: readonly string[];
  /**
   * Random source used for slips. Default `Math.random`. It must return a number
   * in [0, 1). Inject a seeded source for deterministic runs.
   */
  rng?: () => number;
}

/**
 * Frozen Lake grid environment.
 *
 * Cells and symbols: 'S' start, 'H' hole, 'G' goal, 'W' wall, 'C' candy, and
 * anything else is a frozen cell.
 *
 * The slip probability is the chance that the intended move is replaced by a
 * uniformly random perpendicular move. A candy cell grants its reward on every
 * visit and the episode continues. A wall blocks a move like a grid edge, so
 * the agent stays in place. Inject `rng` for deterministic runs.
 *
 * The supported sizes are 4, 6 and 8; no other size is supported.
 */
export class FrozenLakeEnv {
  readonly size: FrozenLakeSize;

  private readonly cells: FrozenLakeCell[];

  private readonly rng: () => number;

  private position: number;

  private finished: boolean;

  private slip: number;

  private candy: number;

  constructor(options: FrozenLakeEnvOptions = {}) {
    this.size = options.size ?? (options.desc ? sizeFromDesc(options.desc) : 4);
    this.cells = cellsFromMap(this.size, options.desc ?? MAPS[this.size]);
    this.slip = clampUnit(options.slipProbability ?? 0);
    this.candy = clampUnit(options.candyReward ?? DEFAULT_CANDY_REWARD);
    this.rng = options.rng ?? Math.random;
    this.position = this.startIndex();
    this.finished = false;
  }

  /**
   * Probability that the intended move is replaced by a uniformly random
   * perpendicular move. The setter clamps the stored value to [0, 1].
   */
  get slipProbability(): number {
    return this.slip;
  }

  set slipProbability(value: number) {
    this.slip = clampUnit(value);
  }

  /**
   * Reward collected when the agent steps on a candy cell. The episode
   * continues and the reward is granted again on every visit. The setter
   * clamps the stored value to [0, 1].
   */
  get candyReward(): number {
    return this.candy;
  }

  set candyReward(value: number) {
    this.candy = clampUnit(value);
  }

  get observation(): number {
    return this.position;
  }

  get done(): boolean {
    return this.finished;
  }

  get map(): readonly FrozenLakeCell[] {
    return this.cells;
  }

  /** Returns the agent to the start cell and clears the terminal flag. */
  reset(): number {
    this.position = this.startIndex();
    this.finished = false;
    return this.position;
  }

  /**
   * Applies `action`, applies a possible slip, and returns the new step result.
   * A terminal episode ignores further actions and keeps the final observation.
   */
  step(action: FrozenLakeAction): FrozenLakeStep {
    if (this.finished) {
      return { observation: this.position, reward: 0, done: true };
    }
    // A slip replaces the intended move with one of the two perpendicular
    // moves. With slipProbability 2/3 the split is 1/3 intended and 1/3 for
    // each perpendicular, which matches gym's FrozenLake.
    let chosen = action;
    if (this.rng() < this.slip) {
      const outcomes = SLIP_OUTCOMES[action];
      chosen = outcomes[Math.floor(this.rng() * 2)];
    }
    const target = this.targetIndex(this.position, chosen);
    // A wall blocks the move like a grid edge, so the agent stays put. A wall
    // is never terminal, so the episode continues.
    if (this.cells[target] === 'wall') {
      return { observation: this.position, reward: 0, done: false };
    }
    this.position = target;
    const cell = this.cells[this.position];
    // A candy cell is not terminal: the reward is granted on every visit.
    if (cell === 'candy') {
      return { observation: this.position, reward: this.candy, done: false };
    }
    this.finished = cell === 'hole' || cell === 'goal';
    return { observation: this.position, reward: cell === 'goal' ? 1 : 0, done: this.finished };
  }

  cellAt(index: number): FrozenLakeCell {
    return this.cells[index];
  }

  isTerminal(index: number): boolean {
    const cell = this.cells[index];
    return cell === 'hole' || cell === 'goal';
  }

  isWall(index: number): boolean {
    return this.cells[index] === 'wall';
  }

  private startIndex(): number {
    return this.cells.indexOf('start');
  }

  private targetIndex(index: number, action: FrozenLakeAction): number {
    const row = Math.floor(index / this.size);
    const column = index % this.size;
    switch (action) {
      case 'left':
        return this.clampedIndex(row, column - 1);
      case 'down':
        return this.clampedIndex(row + 1, column);
      case 'right':
        return this.clampedIndex(row, column + 1);
      case 'up':
        return this.clampedIndex(row - 1, column);
    }
  }

  private clampedIndex(row: number, column: number): number {
    const clampedRow = Math.min(Math.max(row, 0), this.size - 1);
    const clampedColumn = Math.min(Math.max(column, 0), this.size - 1);
    return clampedRow * this.size + clampedColumn;
  }
}

/** Aggregate training counters for {@link FrozenLakeCore}. */
export interface FrozenLakeStats {
  /** Number of finished episodes. */
  readonly episodes: number;
  /** Number of episodes that ended on the goal. */
  readonly successes: number;
  /** Number of environment steps taken. */
  readonly steps: number;
  /** Successes divided by episodes, or 0 before the first episode. */
  readonly successRate: number;
}

/** Constructor options for {@link FrozenLakeCore}. */
export interface FrozenLakeCoreOptions {
  /** Grid size. Default 4. Only 4, 6 and 8 are supported. */
  size?: FrozenLakeSize;
  /** Probability that the intended move slips. Default 0. Clamped to [0, 1]. */
  slipProbability?: number;
  /** Reward for a candy visit. Default {@link DEFAULT_CANDY_REWARD}. Clamped to [0, 1]. */
  candyReward?: number;
  /** Agent learning rate. Default 0.1. Not clamped. */
  learningRate?: number;
  /** Agent discount factor. Default 0.99. Not clamped. */
  discountFactor?: number;
  /** Agent exploration rate. Default 0.1. Not clamped. */
  epsilon?: number;
  /** Agent value for an unseen pair. Default 0. Not clamped. */
  initialValue?: number;
  /** Base rng seed. Default {@link DEFAULT_SEED}. */
  seed?: number;
  /**
   * Custom map rows. When given without `size`, the row count defines the grid
   * size. The map is kept across reset() and cleared by setSize().
   */
  desc?: readonly string[];
  /** Enables replay-buffer training. Default false. */
  replayEnabled?: boolean;
  /** Replay batch size. Default 8. */
  replayBatchSize?: number;
  /** Replay buffer capacity. Default 1000. */
  replayCapacity?: number;
}

/**
 * Frozen Lake training core.
 *
 * It wires the environment, the Q-learning agent and an optional replay buffer.
 * The three parts use independent rng streams: the env uses `seed`, the agent
 * uses `seed + 1` and the buffer uses `seed + 2`.
 *
 * `tick` runs one environment step and then one update, or a replay batch of
 * updates when replay is enabled. The reward history keeps the 300 most recent
 * episode returns.
 */
export class FrozenLakeCore {
  private readonly seed: number;

  private env: FrozenLakeEnv;

  private agent: QLearningAgent<number, FrozenLakeAction>;

  private customMap: readonly string[] | undefined;

  private episodes = 0;

  private successes = 0;

  private steps = 0;

  private episodeReturn = 0;

  private history: number[] = [];

  private readonly replayOptions: {
    enabled: boolean;
    batchSize: number;
    capacity: number;
  };

  private readonly replayBuffer: ReplayBuffer<number, FrozenLakeAction>;

  private lastBatch: readonly Transition<number, FrozenLakeAction>[] = [];

  private phase: FineStepPhase = 'observe';

  private stepObservation = 0;

  private stepAction: FrozenLakeAction | null = null;

  private stepNextObservation: number | null = null;

  private stepReward: number | null = null;

  private stepDone: boolean | null = null;

  private pendingReset = false;

  private lastUpdateResult: QUpdateResult<number, FrozenLakeAction> | null = null;

  constructor(options: FrozenLakeCoreOptions = {}) {
    this.seed = options.seed ?? DEFAULT_SEED;
    this.customMap = options.desc;
    this.env = this.buildEnv({
      size: options.size,
      slipProbability: options.slipProbability,
      candyReward: options.candyReward,
      desc: this.customMap,
    });
    this.agent = new QLearningAgent<number, FrozenLakeAction>(FROZEN_LAKE_ACTIONS, {
      learningRate: options.learningRate,
      discountFactor: options.discountFactor,
      epsilon: options.epsilon,
      initialValue: options.initialValue,
      rng: mulberry32((this.seed + 1) >>> 0),
    });
    this.replayOptions = {
      enabled: options.replayEnabled ?? false,
      batchSize: options.replayBatchSize ?? 8,
      capacity: options.replayCapacity ?? 1000,
    };
    // A third independent rng stream: the env uses the seed, the agent the
    // seed plus one and the buffer the seed plus two.
    this.replayBuffer = new ReplayBuffer<number, FrozenLakeAction>({
      capacity: this.replayOptions.capacity,
      rng: mulberry32((this.seed + 2) >>> 0),
    });
    this.resetFineStep();
  }

  get observation(): number {
    return this.env.observation;
  }

  /** Visible state of the fine-step loop. */
  get fineStepState(): FineStepState {
    return {
      phase: this.phase,
      nextPhase: nextFineStepPhase(this.phase),
      observation: this.stepObservation,
      action: this.stepAction,
      nextObservation: this.stepNextObservation,
      reward: this.stepReward,
      done: this.stepDone,
    };
  }

  /** Result of the most recent Q update, or null before the first update. */
  get lastUpdate(): QUpdateResult<number, FrozenLakeAction> | null {
    return this.lastUpdateResult;
  }

  /** Current aggregate training counters. */
  get stats(): FrozenLakeStats {
    return {
      episodes: this.episodes,
      successes: this.successes,
      steps: this.steps,
      successRate: this.episodes === 0 ? 0 : this.successes / this.episodes,
    };
  }

  /** Reward of each recent episode, oldest first, capped at 300 entries. */
  get rewardHistory(): readonly number[] {
    return this.history;
  }

  /**
   * Greedy action of every cell, or null for a terminal or wall cell. Allocates
   * a fresh array on every read.
   */
  get greedyActions(): readonly (FrozenLakeAction | null)[] {
    const total = this.env.size * this.env.size;
    const actions: (FrozenLakeAction | null)[] = [];
    for (let index = 0; index < total; index += 1) {
      actions.push(this.hasPolicy(index) ? this.agent.policyAction(index) : null);
    }
    return actions;
  }

  /**
   * Max value of every cell, or 0 for a terminal or wall cell. Allocates a
   * fresh array on every read.
   */
  get greedyValues(): readonly number[] {
    const total = this.env.size * this.env.size;
    const values: number[] = [];
    for (let index = 0; index < total; index += 1) {
      values.push(this.hasPolicy(index) ? this.agent.maxValue(index) : 0);
    }
    return values;
  }

  /** Q values of every action for `observation`, in FROZEN_LAKE_ACTIONS order. */
  actionValues(observation: number): readonly number[] {
    return FROZEN_LAKE_ACTIONS.map((action) => this.agent.valueOf(observation, action));
  }

  /** Cell type of `index` in the active map. */
  cellTypeAt(index: number): FrozenLakeCell {
    return this.env.cellAt(index);
  }

  get learningRate(): number {
    return this.agent.learningRate;
  }

  set learningRate(value: number) {
    this.agent.learningRate = value;
  }

  get epsilon(): number {
    return this.agent.epsilon;
  }

  set epsilon(value: number) {
    this.agent.epsilon = value;
  }

  get discountFactor(): number {
    return this.agent.discountFactor;
  }

  set discountFactor(value: number) {
    this.agent.discountFactor = value;
  }

  get initialValue(): number {
    return this.agent.initialValue;
  }

  set initialValue(value: number) {
    this.agent.initialValue = value;
  }

  /**
   * Runs one phase of the fine-step loop and returns the phase it entered.
   *
   * The loop starts in `observe` with the environment observation already
   * available, so the first call selects an action and enters `act`. The `act`
   * call steps the environment, updates the agent and enters `transition`. The
   * `transition` call applies a pending episode reset, clears the step result
   * and enters `observe`.
   */
  advanceFineStep(): FineStepPhase {
    switch (this.phase) {
      case 'observe': {
        this.stepObservation = this.env.observation;
        this.stepAction = this.agent.selectAction(this.stepObservation);
        this.phase = 'act';
        return this.phase;
      }
      case 'act': {
        const action = this.stepAction;
        if (action === null) {
          throw new Error('FrozenLakeCore act phase has no selected action.');
        }
        const result = this.env.step(action);
        const transition: Transition<number, FrozenLakeAction> = {
          observation: this.stepObservation,
          action,
          reward: result.reward,
          nextObservation: result.observation,
          done: result.done,
        };
        if (this.replayOptions.enabled) {
          this.replayBuffer.store(transition);
          const batch = this.replayBuffer.sample(this.replayOptions.batchSize);
          this.lastBatch = batch;
          // The newest transition is last, so the final result belongs to it.
          for (const item of batch) {
            this.lastUpdateResult = this.agent.update(item);
          }
        } else {
          this.lastUpdateResult = this.agent.update(transition);
        }
        this.stepNextObservation = result.observation;
        this.stepReward = result.reward;
        this.stepDone = result.done;
        this.episodeReturn += result.reward;
        this.steps += 1;
        if (result.done) {
          this.episodes += 1;
          // Only the goal ends the episode as a success: an episode that
          // collected candy and then fell in a hole has a positive return.
          if (this.env.cellAt(result.observation) === 'goal') {
            this.successes += 1;
          }
          this.history.push(this.episodeReturn);
          if (this.history.length > REWARD_HISTORY_LIMIT) {
            this.history.shift();
          }
          this.episodeReturn = 0;
          // The transition phase reports the terminal observation, so the
          // environment reset waits until then.
          this.pendingReset = true;
        }
        this.phase = 'transition';
        return this.phase;
      }
      case 'transition': {
        if (this.pendingReset) {
          this.env.reset();
          this.pendingReset = false;
        }
        this.stepObservation = this.env.observation;
        this.stepAction = null;
        this.stepNextObservation = null;
        this.stepReward = null;
        this.stepDone = null;
        this.phase = 'observe';
        return this.phase;
      }
    }
  }

  /**
   * Runs one environment step and trains the agent on it from any phase, then
   * leaves the loop in the observe phase.
   */
  tick(): void {
    while (this.advanceFineStep() !== 'transition') {
      // act or observe steps only.
    }
    this.advanceFineStep();
  }

  /** Rebuilds the environment from the current options and resets training. */
  reset(): void {
    this.env = this.buildEnv({
      size: this.env.size,
      slipProbability: this.env.slipProbability,
      candyReward: this.env.candyReward,
      desc: this.customMap,
    });
    this.resetTraining();
  }

  /** Switches to the preset map of `size` and resets training. */
  setSize(size: FrozenLakeSize): void {
    this.customMap = undefined;
    this.env = this.buildEnv({
      size,
      slipProbability: this.env.slipProbability,
      candyReward: this.env.candyReward,
    });
    this.resetTraining();
  }

  /** Probability that the intended move slips. Clamped to [0, 1] by the env. */
  get slipProbability(): number {
    return this.env.slipProbability;
  }

  set slipProbability(value: number) {
    this.env.slipProbability = value;
  }

  get candyReward(): number {
    return this.env.candyReward;
  }

  set candyReward(value: number) {
    this.env.candyReward = value;
  }

  get replayEnabled(): boolean {
    return this.replayOptions.enabled;
  }

  set replayEnabled(value: boolean) {
    this.replayOptions.enabled = value;
  }

  get replayBatchSize(): number {
    return this.replayOptions.batchSize;
  }

  set replayBatchSize(value: number) {
    this.replayOptions.batchSize = Math.max(1, Math.round(value));
  }

  get replayBufferSize(): number {
    return this.replayBuffer.size;
  }

  get replayCapacity(): number {
    return this.replayOptions.capacity;
  }

  /** Last sampled replay batch, or an empty batch when replay is disabled. */
  get lastReplayBatch(): readonly Transition<number, FrozenLakeAction>[] {
    return this.replayOptions.enabled ? this.lastBatch : [];
  }

  /**
   * Replaces the environment map with custom rows and resets training. The
   * rows define the grid size, so they must contain 4, 6 or 8 rows of matching
   * width and one start cell. Invalid rows throw before the current map is
   * replaced. The custom map is kept for every later rebuild, including
   * reset().
   */
  setCustomMap(rows: readonly string[]): void {
    const customMap = rows.slice();
    // Build first so an invalid map leaves the current environment untouched.
    this.env = this.buildEnv({
      size: this.env.size,
      slipProbability: this.env.slipProbability,
      candyReward: this.env.candyReward,
      desc: customMap,
    });
    this.customMap = customMap;
    this.resetTraining();
  }

  private resetTraining(): void {
    this.agent = this.buildAgent();
    this.episodes = 0;
    this.successes = 0;
    this.steps = 0;
    this.episodeReturn = 0;
    this.history = [];
    this.replayBuffer.clear();
    this.lastBatch = [];
    this.resetFineStep();
  }

  /**
   * Clears the fine-step loop for the current environment. The environment
   * observation is already available, so the loop restarts in `observe`.
   */
  private resetFineStep(): void {
    this.phase = 'observe';
    this.stepObservation = this.env.observation;
    this.stepAction = null;
    this.stepNextObservation = null;
    this.stepReward = null;
    this.stepDone = null;
    this.pendingReset = false;
    this.lastUpdateResult = null;
  }

  private buildAgent(): QLearningAgent<number, FrozenLakeAction> {
    return new QLearningAgent<number, FrozenLakeAction>(FROZEN_LAKE_ACTIONS, {
      learningRate: this.agent.learningRate,
      discountFactor: this.agent.discountFactor,
      epsilon: this.agent.epsilon,
      initialValue: this.agent.initialValue,
      rng: mulberry32((this.seed + 1) >>> 0),
    });
  }

  private hasPolicy(index: number): boolean {
    return !this.env.isTerminal(index) && !this.env.isWall(index);
  }

  private buildEnv(options: FrozenLakeEnvOptions): FrozenLakeEnv {
    // A custom map defines its own grid size, so it overrides the size option
    // while it is active. setSize() clears it. Omitting the size lets the
    // environment derive it from the rows.
    return new FrozenLakeEnv({
      ...options,
      size: options.desc ? undefined : options.size,
      rng: mulberry32(this.seed),
    });
  }
}
