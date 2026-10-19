// biome-ignore lint/style/useImportType: BallMove is a type-only import from a value module
import { BallMove, mulberry32 } from './gradient-descent.js';

export interface NeuronExample {
  x1: number;
  x2: number;
  t: number;
}

export type LossKind = 'l1' | 'l2';

export type NeuronPhase = 'idle' | 'present' | 'gradient' | 'fade' | 'mean' | 'move';

export interface NeuronGradient {
  dw1: number;
  dw2: number;
  db: number;
}

const DEFAULT_SEED = 42;
const DEFAULT_COUNT = 40;
const MAX_COUNT = 500;
const DEFAULT_BATCH_SIZE = 4;
const MIN_BATCH_SIZE = 1;
const MAX_BATCH_SIZE = 12;
const DOMAIN_MIN = 5;
const DOMAIN_MAX = 20;
const DOMAIN_FALLBACK = 5;
const DOMAIN_FIT_SCALE = 1.5;
const DIVERGENCE_FACTOR = 10;
const PROBE_X1 = -0.25;
const PROBE_X2 = 0.25;
const EPOCH_STRIDE = 0x9e3779b1;
const RANK_EPS = 1e-12;

const defaultTarget = (x1: number, x2: number): number => x1 + x2;

export function compileTarget(
  expression: string,
): { fn: (x1: number, x2: number) => number } | { error: string } {
  let fn: (x1: number, x2: number) => number;
  try {
    fn = new Function('x1', 'x2', `return (${expression});`) as (x1: number, x2: number) => number;
  } catch {
    return { error: 'SyntaxError in expression' };
  }
  try {
    const probe = fn(PROBE_X1, PROBE_X2);
    if (typeof probe !== 'number' || !Number.isFinite(probe)) {
      return { error: 'Expression must return a finite number' };
    }
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
  return { fn };
}

export class NeuronCore {
  static readonly DEFAULT_EXPRESSION = 'x1 + x2';

  private currentPhase: NeuronPhase = 'idle';

  private epochCount = 0;

  private resetCount = 0;

  private seed = DEFAULT_SEED;

  private count = DEFAULT_COUNT;

  private currentExpression = NeuronCore.DEFAULT_EXPRESSION;

  private compileError: string | null = null;

  private targetFn: (x1: number, x2: number) => number = defaultTarget;

  private currentLossKind: LossKind = 'l2';

  private biasOn = false;

  private currentBatchSize = DEFAULT_BATCH_SIZE;

  private examples: NeuronExample[] = [];

  private batchItems: NeuronExample[] = [];

  private index = 0;

  private gradients: NeuronGradient[] = [];

  private meanGradient: NeuronGradient | null = null;

  private lastBallMove: BallMove | null = null;

  private isDiverged = false;

  private startWeight1 = 0;

  private startWeight2 = 0;

  private startBias = 0;

  private weight1 = 0;

  private weight2 = 0;

  private biasValue = 0;

  private currentDomain = DOMAIN_FALLBACK;

  constructor() {
    this.setDataset();
  }

  get dataset(): readonly NeuronExample[] {
    return this.examples;
  }

  get expression(): string {
    return this.currentExpression;
  }

  get targetError(): string | null {
    return this.compileError;
  }

  get lossKind(): LossKind {
    return this.currentLossKind;
  }

  get biasEnabled(): boolean {
    return this.biasOn;
  }

  get phase(): NeuronPhase {
    return this.currentPhase;
  }

  get batchSize(): number {
    return this.currentBatchSize;
  }

  get batch(): readonly NeuronExample[] {
    return this.batchItems;
  }

  get currentExample(): NeuronExample | null {
    if (
      this.currentPhase === 'present' ||
      this.currentPhase === 'gradient' ||
      this.currentPhase === 'fade'
    ) {
      return this.batchItems[this.index] ?? null;
    }
    return null;
  }

  get exampleIndex(): number {
    return this.index;
  }

  get epoch(): number {
    return this.epochCount;
  }

  get w1(): number {
    return this.weight1;
  }

  get w2(): number {
    return this.weight2;
  }

  get b(): number {
    return this.biasValue;
  }

  get loss(): number {
    return this.landscapeLoss(this.weight1, this.weight2);
  }

  get gradient(): NeuronGradient {
    if (this.examples.length === 0) {
      return { dw1: 0, dw2: 0, db: 0 };
    }
    let dw1 = 0;
    let dw2 = 0;
    let db = 0;
    for (const example of this.examples) {
      const gradient = this.exampleGradientValue(example);
      dw1 += gradient.dw1;
      dw2 += gradient.dw2;
      db += gradient.db;
    }
    const n = this.examples.length;
    return { dw1: dw1 / n, dw2: dw2 / n, db: db / n };
  }

  get gradientMagnitude(): number {
    const gradient = this.gradient;
    return Math.hypot(gradient.dw1, gradient.dw2);
  }

  get domain(): number {
    return this.currentDomain;
  }

  get lastMove(): BallMove | null {
    return this.lastBallMove;
  }

  get diverged(): boolean {
    return this.isDiverged;
  }

  setDataset(options: { expression?: string; count?: number; seed?: number } = {}): void {
    if (options.seed !== undefined) {
      this.seed = options.seed;
    }
    if (options.expression !== undefined) {
      const result = compileTarget(options.expression);
      if ('fn' in result) {
        this.currentExpression = options.expression;
        this.targetFn = result.fn;
        this.compileError = null;
      } else {
        this.currentExpression = NeuronCore.DEFAULT_EXPRESSION;
        this.targetFn = defaultTarget;
        this.compileError = result.error;
      }
    }
    if (options.count !== undefined) {
      this.count = Math.min(MAX_COUNT, Math.max(1, Math.round(options.count)));
    }
    this.buildDataset();
    this.computeDomain();
    this.reset();
  }

  setLoss(kind: LossKind): void {
    this.currentLossKind = kind;
  }

  setBiasEnabled(enabled: boolean): void {
    this.biasOn = enabled;
    if (!enabled) {
      this.biasValue = 0;
    }
    this.computeDomain();
  }

  setBatchSize(size: number): void {
    this.currentBatchSize = Math.min(MAX_BATCH_SIZE, Math.max(MIN_BATCH_SIZE, Math.round(size)));
  }

  reset(options: { startW1?: number; startW2?: number; startB?: number } = {}): void {
    this.resetCount += 1;
    const rng = mulberry32(this.seed + this.resetCount);
    if (options.startW1 !== undefined && options.startW2 !== undefined) {
      this.startWeight1 = options.startW1;
      this.startWeight2 = options.startW2;
    } else {
      let w1 = 0;
      let w2 = 0;
      do {
        w1 = rng() - 0.5;
        w2 = rng() - 0.5;
      } while (Math.hypot(w1, w2) < 0.1);
      this.startWeight1 = w1;
      this.startWeight2 = w2;
    }
    this.startBias = this.biasOn ? (options.startB ?? rng() - 0.5) : 0;
    this.weight1 = this.startWeight1;
    this.weight2 = this.startWeight2;
    this.biasValue = this.startBias;
    this.currentPhase = 'idle';
    this.epochCount = 0;
    this.batchItems = [];
    this.index = 0;
    this.gradients = [];
    this.meanGradient = null;
    this.lastBallMove = null;
    this.isDiverged = false;
  }

  advance(learningRate: number, options: { detailed?: boolean } = {}): void {
    if (this.isDiverged || this.examples.length === 0) {
      return;
    }
    if (options.detailed === false) {
      this.advanceCoarse(learningRate);
      return;
    }
    switch (this.currentPhase) {
      case 'idle': {
        this.drawBatch();
        this.index = 0;
        this.gradients = [];
        this.meanGradient = null;
        this.currentPhase = 'present';
        break;
      }
      case 'present': {
        this.currentPhase = 'gradient';
        break;
      }
      case 'gradient': {
        const example = this.batchItems[this.index];
        if (example !== undefined) {
          this.gradients.push(this.exampleGradientValue(example));
        }
        this.currentPhase = 'fade';
        break;
      }
      case 'fade': {
        if (this.index + 1 >= this.batchItems.length) {
          this.meanGradient = this.averageCollected();
          this.currentPhase = 'mean';
        } else {
          this.index += 1;
          this.currentPhase = 'present';
        }
        break;
      }
      case 'mean': {
        const mean = this.meanGradient ?? this.averageCollected();
        this.meanGradient = mean;
        this.applyMove(learningRate, mean);
        this.currentPhase = 'move';
        break;
      }
      case 'move': {
        this.epochCount += 1;
        this.drawBatch();
        this.index = 0;
        this.gradients = [];
        this.meanGradient = null;
        this.currentPhase = 'present';
        break;
      }
    }
  }

  advanceBatch(learningRate: number): BallMove | null {
    if (this.isDiverged || this.examples.length === 0) {
      return null;
    }
    const targetEpoch = this.epochCount + 1;
    const guard = 64 * (this.currentBatchSize + 2);
    for (let step = 0; step < guard; step += 1) {
      this.advance(learningRate, { detailed: false });
      if (this.isDiverged) {
        return this.lastBallMove;
      }
      if (this.currentPhase === 'present' && this.epochCount === targetEpoch) {
        return this.lastBallMove;
      }
    }
    return this.lastBallMove;
  }

  landscapeLoss(w1: number, w2: number): number {
    if (this.examples.length === 0) {
      return 0;
    }
    let sum = 0;
    for (const example of this.examples) {
      const err =
        w1 * example.x1 + w2 * example.x2 + (this.biasOn ? this.biasValue : 0) - example.t;
      sum += this.lossOf(err);
    }
    return sum / this.examples.length;
  }

  private advanceCoarse(learningRate: number): void {
    switch (this.currentPhase) {
      case 'idle':
      case 'move': {
        if (this.currentPhase === 'move') {
          this.epochCount += 1;
        }
        this.drawBatch();
        this.index = 0;
        this.gradients = [];
        this.meanGradient = this.batchMeanGradient();
        // The coarse machine ends an epoch on 'present' so that advanceBatch
        // can detect the completed epoch via its phase check.
        this.currentPhase = this.currentPhase === 'move' ? 'present' : 'mean';
        break;
      }
      case 'mean': {
        const mean = this.meanGradient ?? this.batchMeanGradient();
        this.meanGradient = mean;
        this.applyMove(learningRate, mean);
        this.currentPhase = 'move';
        break;
      }
      default: {
        this.gradients = [];
        this.meanGradient = this.batchMeanGradient();
        this.currentPhase = 'mean';
        break;
      }
    }
  }

  private applyMove(learningRate: number, mean: NeuronGradient): void {
    const fromW1 = this.weight1;
    const fromW2 = this.weight2;
    this.weight1 -= learningRate * mean.dw1;
    this.weight2 -= learningRate * mean.dw2;
    if (this.biasOn) {
      this.biasValue -= learningRate * mean.db;
    }
    this.lastBallMove = { fromA: fromW1, fromB: fromW2, toA: this.weight1, toB: this.weight2 };
    this.checkDivergence();
  }

  private checkDivergence(): void {
    const badW1 = this.weight1;
    const badW2 = this.weight2;
    const limit = DIVERGENCE_FACTOR * this.currentDomain;
    if (
      !Number.isFinite(this.weight1) ||
      !Number.isFinite(this.weight2) ||
      !Number.isFinite(this.biasValue) ||
      Math.abs(this.weight1) > limit ||
      Math.abs(this.weight2) > limit ||
      Math.abs(this.biasValue) > limit
    ) {
      this.weight1 = this.startWeight1;
      this.weight2 = this.startWeight2;
      this.biasValue = this.startBias;
      this.isDiverged = true;
      this.lastBallMove = {
        fromA: badW1,
        fromB: badW2,
        toA: this.startWeight1,
        toB: this.startWeight2,
      };
    }
  }

  private drawBatch(): void {
    const rng = mulberry32(this.seed + this.epochCount * EPOCH_STRIDE);
    const n = this.examples.length;
    const batch: NeuronExample[] = [];
    for (let i = 0; i < this.currentBatchSize; i += 1) {
      const source = this.examples[Math.min(n - 1, Math.floor(rng() * n))];
      if (source !== undefined) {
        batch.push({ ...source });
      }
    }
    this.batchItems = batch;
  }

  private buildDataset(): void {
    const rng = mulberry32(this.seed);
    const dataset: NeuronExample[] = [];
    for (let i = 0; i < this.count; i += 1) {
      const x1 = rng() * 2 - 1;
      const x2 = rng() * 2 - 1;
      try {
        const t = this.targetFn(x1, x2);
        if (typeof t === 'number' && Number.isFinite(t)) {
          dataset.push({ x1, x2, t });
        }
      } catch {
        // Skip samples where the target function throws.
      }
    }
    this.examples = dataset;
  }

  private computeDomain(): void {
    const fit = this.leastSquaresFit();
    if (fit === null) {
      this.currentDomain = DOMAIN_FALLBACK;
      return;
    }
    const scale = DOMAIN_FIT_SCALE * Math.max(Math.abs(fit.w1), Math.abs(fit.w2), 1);
    this.currentDomain = Math.min(DOMAIN_MAX, Math.max(DOMAIN_MIN, scale));
  }

  private leastSquaresFit(): { w1: number; w2: number } | null {
    const n = this.examples.length;
    const dim = this.biasOn ? 3 : 2;
    if (n === 0 || n < dim) {
      return null;
    }
    const xtx: number[][] = [];
    const xt: number[] = [];
    for (let i = 0; i < dim; i += 1) {
      const row: number[] = [];
      for (let j = 0; j < dim; j += 1) {
        row.push(0);
      }
      xtx.push(row);
      xt.push(0);
    }
    for (const example of this.examples) {
      const row = this.biasOn ? [example.x1, example.x2, 1] : [example.x1, example.x2];
      for (let i = 0; i < dim; i += 1) {
        xt[i] += row[i] * example.t;
        for (let j = 0; j < dim; j += 1) {
          xtx[i][j] += row[i] * row[j];
        }
      }
    }
    const solution = this.solveLinearSystem(xtx, xt);
    if (solution === null) {
      return null;
    }
    return { w1: solution[0], w2: solution[1] };
  }

  private solveLinearSystem(matrix: number[][], rhs: number[]): number[] | null {
    const size = rhs.length;
    const augmented = matrix.map((row, i) => [...row, rhs[i]]);
    let maxDiag = 0;
    for (let i = 0; i < size; i += 1) {
      maxDiag = Math.max(maxDiag, Math.abs(augmented[i][i]));
    }
    for (let col = 0; col < size; col += 1) {
      let pivotRow = col;
      for (let row = col + 1; row < size; row += 1) {
        if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivotRow][col])) {
          pivotRow = row;
        }
      }
      const pivot = augmented[pivotRow][col];
      if (Math.abs(pivot) <= RANK_EPS * maxDiag) {
        return null;
      }
      if (pivotRow !== col) {
        const swap = augmented[col];
        augmented[col] = augmented[pivotRow];
        augmented[pivotRow] = swap;
      }
      for (let row = col + 1; row < size; row += 1) {
        const factor = augmented[row][col] / pivot;
        for (let k = col; k <= size; k += 1) {
          augmented[row][k] -= factor * augmented[col][k];
        }
      }
    }
    const solution = new Array<number>(size).fill(0);
    for (let row = size - 1; row >= 0; row -= 1) {
      let sum = augmented[row][size];
      for (let k = row + 1; k < size; k += 1) {
        sum -= augmented[row][k] * solution[k];
      }
      solution[row] = sum / augmented[row][row];
    }
    for (const value of solution) {
      if (!Number.isFinite(value)) {
        return null;
      }
    }
    return solution;
  }

  private averageCollected(): NeuronGradient {
    if (this.gradients.length === 0) {
      return { dw1: 0, dw2: 0, db: 0 };
    }
    let dw1 = 0;
    let dw2 = 0;
    let db = 0;
    for (const gradient of this.gradients) {
      dw1 += gradient.dw1;
      dw2 += gradient.dw2;
      db += gradient.db;
    }
    const n = this.gradients.length;
    return { dw1: dw1 / n, dw2: dw2 / n, db: db / n };
  }

  private batchMeanGradient(): NeuronGradient {
    if (this.batchItems.length === 0) {
      return { dw1: 0, dw2: 0, db: 0 };
    }
    let dw1 = 0;
    let dw2 = 0;
    let db = 0;
    for (const example of this.batchItems) {
      const gradient = this.exampleGradientValue(example);
      dw1 += gradient.dw1;
      dw2 += gradient.dw2;
      db += gradient.db;
    }
    const n = this.batchItems.length;
    return { dw1: dw1 / n, dw2: dw2 / n, db: db / n };
  }

  private exampleError(example: NeuronExample): number {
    return (
      this.weight1 * example.x1 +
      this.weight2 * example.x2 +
      (this.biasOn ? this.biasValue : 0) -
      example.t
    );
  }

  private lossOf(err: number): number {
    return this.currentLossKind === 'l1' ? Math.abs(err) : 0.5 * err * err;
  }

  private exampleGradientValue(example: NeuronExample): NeuronGradient {
    const err = this.exampleError(example);
    const factor = this.currentLossKind === 'l1' ? Math.sign(err) : err;
    return {
      dw1: factor * example.x1,
      dw2: factor * example.x2,
      db: this.biasOn ? factor : 0,
    };
  }
}
