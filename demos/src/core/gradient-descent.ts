export interface Example {
  x1: number;
  x2: number;
  t: number;
}

export type Phase = 'idle' | 'present' | 'gradient' | 'fade' | 'mean' | 'move';

export interface Gradient2 {
  da: number;
  db: number;
}

export interface BallMove {
  fromA: number;
  fromB: number;
  toA: number;
  toB: number;
}

const TARGET_C = 1.2;
const TARGET_D = -0.8;
const DOMAIN = 5;

const DEFAULT_START_A = -2.5;
const DEFAULT_START_B = 2.2;
const DEFAULT_BATCH_SIZE = 4;
const DEFAULT_SEED = 42;
const DEFAULT_LEARNING_RATE = 0.15;

const EPOCH_STRIDE = 0x9e3779b1;

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function landscapeLoss(a: number, b: number): number {
  const dc = a - TARGET_C;
  const dd = b - TARGET_D;
  return (dc * dc + dd * dd) / 6;
}

export function landscapeGradient(a: number, b: number): Gradient2 {
  return { da: (a - TARGET_C) / 3, db: (b - TARGET_D) / 3 };
}

export function exampleLoss(a: number, b: number, example: Example): number {
  const err = a * example.x1 + b * example.x2 - example.t;
  return 0.5 * err * err;
}

export function exampleGradient(a: number, b: number, example: Example): Gradient2 {
  const err = a * example.x1 + b * example.x2 - example.t;
  return { da: err * example.x1, db: err * example.x2 };
}

export interface ResetOptions {
  startA?: number;
  startB?: number;
  batchSize?: number;
  seed?: number;
}

export interface AdvanceOptions {
  /**
   * When true (default), walk the full per-example cycle:
   * present → gradient → fade → … → mean → move, where fade archives the
   * example arrow and clears the loss patch in a single micro-step.
   * When false, skip per-example detail and cycle mean ↔ move, drawing a
   * fresh batch (via the PRNG) at each epoch.
   */
  detailed?: boolean;
}

export class GradientDescentCore {
  static readonly TARGET_C = TARGET_C;
  static readonly TARGET_D = TARGET_D;
  static readonly DOMAIN = DOMAIN;

  private currentPhase: Phase = 'idle';

  private epochCount = 0;

  private startA = DEFAULT_START_A;

  private startB = DEFAULT_START_B;

  private batchSize = DEFAULT_BATCH_SIZE;

  private seed = DEFAULT_SEED;

  private learningRate = DEFAULT_LEARNING_RATE;

  private a = DEFAULT_START_A;

  private b = DEFAULT_START_B;

  private examples: Example[] = [];

  private index = 0;

  private gradients: Gradient2[] = [];

  private meanGradient: Gradient2 | null = null;

  private lastMove: BallMove | null = null;

  constructor() {
    this.reset();
  }

  get phase(): Phase {
    return this.currentPhase;
  }

  get epoch(): number {
    return this.epochCount;
  }

  get ballA(): number {
    return this.a;
  }

  get ballB(): number {
    return this.b;
  }

  get currentExample(): Example | null {
    if (
      this.currentPhase === 'present' ||
      this.currentPhase === 'gradient' ||
      this.currentPhase === 'fade'
    ) {
      return this.examples[this.index] ?? null;
    }
    return null;
  }

  get exampleIndex(): number {
    return this.index;
  }

  get batch(): Example[] {
    return this.examples;
  }

  get collected(): Gradient2[] {
    return this.gradients;
  }

  get batchGradient(): Gradient2 | null {
    return this.meanGradient;
  }

  get loss(): number {
    return landscapeLoss(this.a, this.b);
  }

  get gradientMagnitude(): number {
    const gradient = landscapeGradient(this.a, this.b);
    return Math.hypot(gradient.da, gradient.db);
  }

  reset(options: ResetOptions = {}): void {
    this.startA = options.startA ?? DEFAULT_START_A;
    this.startB = options.startB ?? DEFAULT_START_B;
    this.batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
    this.seed = options.seed ?? DEFAULT_SEED;
    this.a = this.startA;
    this.b = this.startB;
    this.currentPhase = 'idle';
    this.epochCount = 0;
    this.examples = [];
    this.index = 0;
    this.gradients = [];
    this.meanGradient = null;
    this.lastMove = null;
  }

  advance(learningRate: number = this.learningRate, options: AdvanceOptions = {}): void {
    this.learningRate = learningRate;
    if (options.detailed === false) {
      this.advanceCoarse();
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
        const example = this.examples[this.index];
        if (example !== undefined) {
          this.gradients.push(exampleGradient(this.a, this.b, example));
        }
        this.currentPhase = 'fade';
        break;
      }
      case 'fade': {
        if (this.index + 1 >= this.examples.length) {
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
        this.applyMove(mean);
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

  private advanceCoarse(): void {
    switch (this.currentPhase) {
      case 'idle': {
        this.drawBatch();
        this.index = 0;
        this.gradients = [];
        this.meanGradient = this.batchMeanGradient();
        this.currentPhase = 'mean';
        break;
      }
      case 'mean': {
        const mean = this.meanGradient ?? this.batchMeanGradient();
        this.meanGradient = mean;
        this.applyMove(mean);
        this.currentPhase = 'move';
        break;
      }
      case 'move': {
        this.epochCount += 1;
        this.drawBatch();
        this.index = 0;
        this.gradients = [];
        this.meanGradient = this.batchMeanGradient();
        this.currentPhase = 'mean';
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

  private applyMove(mean: Gradient2): void {
    const fromA = this.a;
    const fromB = this.b;
    this.a -= this.learningRate * mean.da;
    this.b -= this.learningRate * mean.db;
    this.lastMove = { fromA, fromB, toA: this.a, toB: this.b };
  }

  advanceBatch(learningRate: number): BallMove | null {
    const targetEpoch = this.epochCount + 1;
    this.lastMove = null;
    const guard = 64 * (this.batchSize + 2);
    for (let step = 0; step < guard; step += 1) {
      this.advance(learningRate);
      if (this.currentPhase === 'present' && this.epochCount === targetEpoch) {
        return this.lastMove;
      }
    }
    return this.lastMove;
  }

  private averageCollected(): Gradient2 {
    if (this.gradients.length === 0) {
      return { da: 0, db: 0 };
    }
    let da = 0;
    let db = 0;
    for (const gradient of this.gradients) {
      da += gradient.da;
      db += gradient.db;
    }
    return { da: da / this.gradients.length, db: db / this.gradients.length };
  }

  private batchMeanGradient(): Gradient2 {
    if (this.examples.length === 0) {
      return { da: 0, db: 0 };
    }
    let da = 0;
    let db = 0;
    for (const example of this.examples) {
      const gradient = exampleGradient(this.a, this.b, example);
      da += gradient.da;
      db += gradient.db;
    }
    return { da: da / this.examples.length, db: db / this.examples.length };
  }

  private drawBatch(): void {
    const rng = mulberry32(this.seed + this.epochCount * EPOCH_STRIDE);
    const examples: Example[] = [];
    for (let i = 0; i < this.batchSize; i += 1) {
      const x1 = rng() * 2 - 1;
      const x2 = rng() * 2 - 1;
      examples.push({ x1, x2, t: TARGET_C * x1 + TARGET_D * x2 });
    }
    this.examples = examples;
  }
}
