import { QLearningAgent } from './q-learning.js';

export type FrozenLakeAction = 'left' | 'down' | 'right' | 'up';

export type FrozenLakeCell = 'start' | 'frozen' | 'hole' | 'goal';

export const FROZEN_LAKE_ACTIONS: readonly FrozenLakeAction[] = ['left', 'down', 'right', 'up'];

const DEFAULT_SEED = 42;
const REWARD_HISTORY_LIMIT = 300;

export type FrozenLakeSize = 4 | 6 | 8;

const MAPS: Readonly<Record<FrozenLakeSize, readonly string[]>> = {
  4: ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
  // User-editable template: change these rows to build your own 6x6 lake.
  // Every row must have exactly 6 characters and the map needs exactly one 'S'.
  6: ['SFFFFF', 'FHFHFF', 'FFFHFF', 'HHFFFF', 'FFFHFF', 'FFFFFG'],
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

function clampProbability(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

function sizeFromRows(rows: readonly string[]): FrozenLakeSize {
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

export interface FrozenLakeStep {
  readonly observation: number;
  readonly reward: number;
  readonly done: boolean;
}

export class FrozenLakeEnv {
  readonly size: FrozenLakeSize;

  private readonly cells: FrozenLakeCell[];

  private readonly rng: () => number;

  private position: number;

  private finished: boolean;

  private slip: number;

  constructor(
    options: {
      size?: FrozenLakeSize;
      slipProbability?: number;
      desc?: readonly string[];
      rng?: () => number;
    } = {},
  ) {
    this.size = options.size ?? 4;
    this.cells = cellsFromMap(this.size, options.desc ?? MAPS[this.size]);
    this.slip = clampProbability(options.slipProbability ?? 0);
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
    this.slip = clampProbability(value);
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

  reset(): number {
    this.position = this.startIndex();
    this.finished = false;
    return this.position;
  }

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
    this.position = this.targetIndex(this.position, chosen);
    const cell = this.cells[this.position];
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

export interface FrozenLakeStats {
  readonly episodes: number;
  readonly successes: number;
  readonly steps: number;
  readonly successRate: number;
}

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

  constructor(
    options: {
      size?: FrozenLakeSize;
      slipProbability?: number;
      learningRate?: number;
      discountFactor?: number;
      epsilon?: number;
      initialValue?: number;
      seed?: number;
      desc?: readonly string[];
    } = {},
  ) {
    this.seed = options.seed ?? DEFAULT_SEED;
    this.customMap = options.desc;
    this.env = this.buildEnv(options.size, options.slipProbability, this.customMap);
    this.agent = new QLearningAgent<number, FrozenLakeAction>(FROZEN_LAKE_ACTIONS, {
      learningRate: options.learningRate,
      discountFactor: options.discountFactor,
      epsilon: options.epsilon,
      initialValue: options.initialValue,
      rng: mulberry32((this.seed + 1) >>> 0),
    });
  }

  get observation(): number {
    return this.env.observation;
  }

  get stats(): FrozenLakeStats {
    return {
      episodes: this.episodes,
      successes: this.successes,
      steps: this.steps,
      successRate: this.episodes === 0 ? 0 : this.successes / this.episodes,
    };
  }

  get rewardHistory(): readonly number[] {
    return this.history;
  }

  get greedyActions(): readonly (FrozenLakeAction | null)[] {
    const total = this.env.size * this.env.size;
    const actions: (FrozenLakeAction | null)[] = [];
    for (let index = 0; index < total; index += 1) {
      actions.push(this.env.isTerminal(index) ? null : this.agent.policyAction(index));
    }
    return actions;
  }

  get greedyValues(): readonly number[] {
    const total = this.env.size * this.env.size;
    const values: number[] = [];
    for (let index = 0; index < total; index += 1) {
      values.push(this.env.isTerminal(index) ? 0 : this.agent.maxValue(index));
    }
    return values;
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

  tick(): void {
    const observation = this.env.observation;
    const action = this.agent.selectAction(observation);
    const result = this.env.step(action);
    this.agent.update({
      observation,
      action,
      reward: result.reward,
      nextObservation: result.observation,
      done: result.done,
    });
    this.episodeReturn += result.reward;
    this.steps += 1;
    if (result.done) {
      this.episodes += 1;
      if (this.episodeReturn > 0) {
        this.successes += 1;
      }
      this.history.push(this.episodeReturn);
      if (this.history.length > REWARD_HISTORY_LIMIT) {
        this.history.shift();
      }
      this.episodeReturn = 0;
      this.env.reset();
    }
  }

  reset(): void {
    this.env = this.buildEnv(this.env.size, this.env.slipProbability, this.customMap);
    this.resetTraining();
  }

  setSize(size: FrozenLakeSize): void {
    this.customMap = undefined;
    this.env = this.buildEnv(size, this.env.slipProbability, undefined);
    this.resetTraining();
  }

  setSlipProbability(value: number): void {
    this.env.slipProbability = value;
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
    this.env = this.buildEnv(this.env.size, this.env.slipProbability, customMap);
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

  private buildEnv(
    size: FrozenLakeSize | undefined,
    slipProbability: number | undefined,
    desc: readonly string[] | undefined,
  ): FrozenLakeEnv {
    // A custom map defines its own grid size, so it overrides the size option
    // while it is active. setSize() clears it.
    return new FrozenLakeEnv({
      size: desc ? sizeFromRows(desc) : size,
      slipProbability,
      desc,
      rng: mulberry32(this.seed),
    });
  }
}
