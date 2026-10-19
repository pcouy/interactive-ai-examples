import { QLearningAgent } from './q-learning.js';

export type FrozenLakeAction = 'left' | 'down' | 'right' | 'up';

export type FrozenLakeCell = 'start' | 'frozen' | 'hole' | 'goal';

export const FROZEN_LAKE_ACTIONS: readonly FrozenLakeAction[] = ['left', 'down', 'right', 'up'];

const DEFAULT_SEED = 42;
const REWARD_HISTORY_LIMIT = 300;

const MAPS: Readonly<Record<4 | 8, readonly string[]>> = {
  4: ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
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

// Slip outcomes when the lake is slippery: [intended, first perpendicular, second perpendicular].
const SLIP_OUTCOMES: Readonly<
  Record<FrozenLakeAction, readonly [FrozenLakeAction, FrozenLakeAction]>
> = {
  left: ['up', 'down'],
  down: ['left', 'right'],
  right: ['up', 'down'],
  up: ['left', 'right'],
};

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

function cellsFromMap(size: 4 | 8): FrozenLakeCell[] {
  const cells: FrozenLakeCell[] = [];
  for (const row of MAPS[size]) {
    for (const symbol of row) {
      cells.push(cellFromSymbol(symbol));
    }
  }
  return cells;
}

export interface FrozenLakeStep {
  readonly observation: number;
  readonly reward: number;
  readonly done: boolean;
}

export class FrozenLakeEnv {
  readonly size: 4 | 8;

  slippery: boolean;

  private readonly cells: FrozenLakeCell[];

  private readonly rng: () => number;

  private position: number;

  private finished: boolean;

  constructor(options: { size?: 4 | 8; slippery?: boolean; rng?: () => number } = {}) {
    this.size = options.size ?? 4;
    this.cells = cellsFromMap(this.size);
    this.slippery = options.slippery ?? false;
    this.rng = options.rng ?? Math.random;
    this.position = this.startIndex();
    this.finished = false;
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
    let chosen = action;
    if (this.slippery) {
      const outcomes: readonly FrozenLakeAction[] = [action, ...SLIP_OUTCOMES[action]];
      chosen = outcomes[Math.floor(this.rng() * 3)];
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

  private episodes = 0;

  private successes = 0;

  private steps = 0;

  private episodeReturn = 0;

  private history: number[] = [];

  constructor(
    options: {
      size?: 4 | 8;
      slippery?: boolean;
      learningRate?: number;
      discountFactor?: number;
      epsilon?: number;
      seed?: number;
    } = {},
  ) {
    this.seed = options.seed ?? DEFAULT_SEED;
    this.env = this.buildEnv(options.size, options.slippery);
    this.agent = new QLearningAgent<number, FrozenLakeAction>(FROZEN_LAKE_ACTIONS, {
      learningRate: options.learningRate,
      discountFactor: options.discountFactor,
      epsilon: options.epsilon,
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
    this.env = this.buildEnv(this.env.size, this.env.slippery);
    this.resetTraining();
  }

  setSize(size: 4 | 8): void {
    this.env = this.buildEnv(size, this.env.slippery);
    this.resetTraining();
  }

  setSlippery(value: boolean): void {
    this.env.slippery = value;
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
      rng: mulberry32((this.seed + 1) >>> 0),
    });
  }

  private buildEnv(size: 4 | 8 | undefined, slippery: boolean | undefined): FrozenLakeEnv {
    return new FrozenLakeEnv({ size, slippery, rng: mulberry32(this.seed) });
  }
}
