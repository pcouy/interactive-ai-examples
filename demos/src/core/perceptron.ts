export interface LabeledPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

export type DatasetName = 'separable' | 'noisy' | 'non-separable';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianFrom(rng: () => number): () => number {
  return (): number => {
    let u = 0;
    let v = 0;
    while (u === 0) {
      u = rng();
    }
    while (v === 0) {
      v = rng();
    }
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

function clamp01(value: number): number {
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

function distanceToDiagonal(x: number, y: number): number {
  return Math.abs(x + y - 1) / Math.SQRT2;
}

function sampleCluster(
  gaussian: () => number,
  count: number,
  centerX: number,
  centerY: number,
  std: number,
  label: 0 | 1,
  margin: number,
): LabeledPoint[] {
  const result: LabeledPoint[] = [];
  while (result.length < count) {
    const x = clamp01(centerX + gaussian() * std);
    const y = clamp01(centerY + gaussian() * std);
    if (distanceToDiagonal(x, y) < margin) {
      continue;
    }
    const signed = x + y - 1;
    if (label === 0 ? signed >= 0 : signed <= 0) {
      continue;
    }
    result.push({ x, y, label });
  }
  return result;
}

export function separableDataset(): LabeledPoint[] {
  const rng = mulberry32(1001);
  const gaussian = gaussianFrom(rng);
  return [
    ...sampleCluster(gaussian, 10, 0.3, 0.28, 0.09, 0, 0.15),
    ...sampleCluster(gaussian, 10, 0.7, 0.72, 0.09, 1, 0.15),
  ];
}

export function noisyDataset(): LabeledPoint[] {
  const rng = mulberry32(2002);
  const gaussian = gaussianFrom(rng);
  return [
    ...sampleCluster(gaussian, 12, 0.42, 0.4, 0.12, 0, 0.03),
    ...sampleCluster(gaussian, 12, 0.58, 0.6, 0.12, 1, 0.03),
  ];
}

export function nonSeparableDataset(): LabeledPoint[] {
  const rng = mulberry32(3003);
  const margin = 0.08;
  const points: LabeledPoint[] = [];
  for (const xLow of [true, false]) {
    for (const yLow of [true, false]) {
      const label: 0 | 1 = xLow === yLow ? 1 : 0;
      let count = 0;
      while (count < 6) {
        const x = rng();
        const y = rng();
        if (xLow ? x >= 0.5 : x < 0.5) {
          continue;
        }
        if (yLow ? y >= 0.5 : y < 0.5) {
          continue;
        }
        if (Math.abs(x - 0.5) < margin || Math.abs(y - 0.5) < margin) {
          continue;
        }
        points.push({ x, y, label });
        count += 1;
      }
    }
  }
  for (let i = points.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    const temporary = points[i];
    points[i] = points[j];
    points[j] = temporary;
  }
  return points;
}

export const PRESET_DATASETS: Record<DatasetName, () => LabeledPoint[]> = {
  separable: separableDataset,
  noisy: noisyDataset,
  'non-separable': nonSeparableDataset,
};

export class PerceptronCore {
  private readonly seed: number;

  private resetCount = 0;

  private w1 = 0;

  private w2 = 0;

  private biasValue = 0;

  private data: LabeledPoint[] = [];

  private cursor = 0;

  private ticks = 0;

  private epochCount = 0;

  private errorsInPass = 0;

  private converged = false;

  private convergenceSteps: number | null = null;

  private lastIndex: number | null = null;

  constructor(seed = 42) {
    this.seed = seed;
    this.reset();
  }

  tick(learningRate: number): void {
    if (this.converged || this.data.length === 0) {
      return;
    }
    const point = this.data[this.cursor];
    const prediction = this.predictionAt(point.x, point.y);
    if (prediction !== point.label) {
      const error = point.label - prediction;
      this.w1 += learningRate * error * point.x;
      this.w2 += learningRate * error * point.y;
      this.biasValue += learningRate * error;
      this.errorsInPass += 1;
    }
    this.ticks += 1;
    this.lastIndex = this.cursor;
    this.cursor = (this.cursor + 1) % this.data.length;
    if (this.cursor === 0) {
      this.epochCount += 1;
      if (this.errorsInPass === 0) {
        this.converged = true;
        this.convergenceSteps = this.ticks;
      }
      this.errorsInPass = 0;
    }
  }

  reset(): void {
    this.initWeights();
    this.cursor = 0;
    this.ticks = 0;
    this.epochCount = 0;
    this.errorsInPass = 0;
    this.converged = false;
    this.convergenceSteps = null;
    this.lastIndex = null;
  }

  setData(points: LabeledPoint[]): void {
    this.data = points.map((point) => ({ ...point }));
    this.reset();
  }

  addPoint(point: LabeledPoint): void {
    this.data.push({ ...point });
    this.converged = false;
    this.convergenceSteps = null;
  }

  clearPoints(): void {
    this.data = [];
    this.reset();
  }

  predictionAt(x: number, y: number): 0 | 1 {
    return this.w1 * x + this.w2 * y + this.biasValue >= 0 ? 1 : 0;
  }

  get weights(): { w1: number; w2: number } {
    return { w1: this.w1, w2: this.w2 };
  }

  get bias(): number {
    return this.biasValue;
  }

  get points(): readonly LabeledPoint[] {
    return this.data;
  }

  get accuracy(): number {
    if (this.data.length === 0) {
      return 1;
    }
    let correct = 0;
    for (const point of this.data) {
      if (this.predictionAt(point.x, point.y) === point.label) {
        correct += 1;
      }
    }
    return correct / this.data.length;
  }

  get misclassifiedIndices(): number[] {
    const indices: number[] = [];
    this.data.forEach((point, index) => {
      if (this.predictionAt(point.x, point.y) !== point.label) {
        indices.push(index);
      }
    });
    return indices;
  }

  get isConverged(): boolean {
    return this.converged;
  }

  get stepsToConvergence(): number | null {
    return this.convergenceSteps;
  }

  get epochs(): number {
    return this.epochCount;
  }

  get ticksElapsed(): number {
    return this.ticks;
  }

  get lastPresentedIndex(): number | null {
    return this.lastIndex;
  }

  /** Index du prochain point qui sera présenté, ou null si le jeu est vide. */
  get nextIndex(): number | null {
    return this.data.length === 0 ? null : this.cursor;
  }

  /** Prochain point qui sera présenté, ou null si le jeu est vide. */
  get nextPoint(): LabeledPoint | null {
    return this.data.length === 0 ? null : this.data[this.cursor];
  }

  private initWeights(): void {
    this.resetCount += 1;
    const rng = mulberry32(this.seed + this.resetCount);
    let w1 = 0;
    let w2 = 0;
    do {
      w1 = rng() - 0.5;
      w2 = rng() - 0.5;
    } while (Math.hypot(w1, w2) < 0.1);
    this.w1 = w1;
    this.w2 = w2;
    this.biasValue = rng() - 0.5;
  }
}
