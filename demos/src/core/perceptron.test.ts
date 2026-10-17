import { describe, expect, it } from 'vitest';
import {
  type LabeledPoint,
  noisyDataset,
  nonSeparableDataset,
  PerceptronCore,
  PRESET_DATASETS,
  separableDataset,
} from './perceptron.js';

function trainUntil(core: PerceptronCore, learningRate: number, maxTicks: number): void {
  for (let i = 0; i < maxTicks && !core.isConverged; i += 1) {
    core.tick(learningRate);
  }
}

function sidesOfDiagonal(points: LabeledPoint[]): boolean {
  return points.every((point) =>
    point.label === 0 ? point.x + point.y < 1 : point.x + point.y > 1,
  );
}

describe('datasets', () => {
  it('provides balanced presets with fresh arrays', () => {
    const first = separableDataset();
    const second = separableDataset();
    expect(first).not.toBe(second);
    expect(first).toHaveLength(20);
    expect(first.filter((point) => point.label === 0)).toHaveLength(10);
    expect(first.filter((point) => point.label === 1)).toHaveLength(10);
    expect(noisyDataset()).toHaveLength(24);
    expect(nonSeparableDataset()).toHaveLength(24);
    expect(Object.keys(PRESET_DATASETS).sort()).toEqual(['noisy', 'non-separable', 'separable']);
  });

  it('keeps separable and noisy clusters strictly on opposite sides of x + y = 1', () => {
    expect(sidesOfDiagonal(separableDataset())).toBe(true);
    expect(sidesOfDiagonal(noisyDataset())).toBe(true);
  });

  it('keeps every generated coordinate inside the unit square', () => {
    for (const build of [separableDataset, noisyDataset, nonSeparableDataset]) {
      for (const point of build()) {
        expect(point.x).toBeGreaterThanOrEqual(0);
        expect(point.x).toBeLessThanOrEqual(1);
        expect(point.y).toBeGreaterThanOrEqual(0);
        expect(point.y).toBeLessThanOrEqual(1);
      }
    }
  });

  it('builds a non-separable XOR pattern', () => {
    for (const point of nonSeparableDataset()) {
      const expected = point.x < 0.5 === point.y < 0.5 ? 1 : 0;
      expect(point.label).toBe(expected);
    }
  });
});

describe('PerceptronCore convergence', () => {
  it('converges on the separable dataset', () => {
    const core = new PerceptronCore(1);
    core.setData(separableDataset());
    trainUntil(core, 0.1, 5000);
    expect(core.isConverged).toBe(true);
    expect(core.accuracy).toBe(1);
    expect(core.stepsToConvergence).not.toBeNull();
  });

  it('converges on the noisy dataset', () => {
    const core = new PerceptronCore(1);
    core.setData(noisyDataset());
    trainUntil(core, 0.1, 20000);
    expect(core.isConverged).toBe(true);
    expect(core.accuracy).toBe(1);
    expect(core.stepsToConvergence).not.toBeNull();
  });

  it('does not converge on the non-separable dataset', () => {
    const core = new PerceptronCore(5);
    core.setData(nonSeparableDataset());
    trainUntil(core, 0.1, 10000);
    expect(core.isConverged).toBe(false);
    expect(core.accuracy).toBeLessThan(1);
    expect(Number.isFinite(core.weights.w1)).toBe(true);
    expect(Number.isFinite(core.weights.w2)).toBe(true);
    expect(Number.isFinite(core.bias)).toBe(true);
  });
});

describe('PerceptronCore determinism', () => {
  it('produces identical weights for identical seeds and data', () => {
    const data = separableDataset();
    const a = new PerceptronCore(7);
    const b = new PerceptronCore(7);
    a.setData(data);
    b.setData(data);
    for (let i = 0; i < 50; i += 1) {
      a.tick(0.1);
      b.tick(0.1);
    }
    expect(a.weights.w1).toBeCloseTo(b.weights.w1);
    expect(a.weights.w2).toBeCloseTo(b.weights.w2);
    expect(a.bias).toBeCloseTo(b.bias);
    expect(a.ticksElapsed).toBe(b.ticksElapsed);
  });

  it('varies weights from one reset to the next', () => {
    const core = new PerceptronCore(13);
    core.setData(separableDataset());
    const first = { ...core.weights };
    core.reset();
    const second = { ...core.weights };
    const differs = first.w1 !== second.w1 || first.w2 !== second.w2;
    expect(differs).toBe(true);
  });
});

describe('PerceptronCore training step', () => {
  it('freezes entirely once converged', () => {
    const core = new PerceptronCore(3);
    core.setData(separableDataset());
    trainUntil(core, 0.1, 5000);
    expect(core.isConverged).toBe(true);
    const snapshot = {
      w1: core.weights.w1,
      w2: core.weights.w2,
      bias: core.bias,
      ticks: core.ticksElapsed,
      epochs: core.epochs,
      last: core.lastPresentedIndex,
    };
    core.tick(0.5);
    expect(core.weights.w1).toBeCloseTo(snapshot.w1);
    expect(core.weights.w2).toBeCloseTo(snapshot.w2);
    expect(core.bias).toBeCloseTo(snapshot.bias);
    expect(core.ticksElapsed).toBe(snapshot.ticks);
    expect(core.epochs).toBe(snapshot.epochs);
    expect(core.lastPresentedIndex).toBe(snapshot.last);
  });

  it('applies the exact Rosenblatt update on a misclassified point', () => {
    const core = new PerceptronCore(11);
    core.setData([{ x: 0.7, y: 0.6, label: 1 }]);
    const w1 = core.weights.w1;
    const w2 = core.weights.w2;
    const bias = core.bias;
    const learningRate = 0.2;
    const before = core.predictionAt(0.7, 0.6);
    core.tick(learningRate);
    if (before === 0) {
      expect(core.weights.w1).toBeCloseTo(w1 + learningRate * 0.7);
      expect(core.weights.w2).toBeCloseTo(w2 + learningRate * 0.6);
      expect(core.bias).toBeCloseTo(bias + learningRate);
    } else {
      expect(core.weights.w1).toBeCloseTo(w1);
      expect(core.weights.w2).toBeCloseTo(w2);
      expect(core.bias).toBeCloseTo(bias);
    }
  });

  it('cycles the cursor and counts epochs', () => {
    const core = new PerceptronCore(9);
    core.setData([
      { x: 0.2, y: 0.5, label: 1 },
      { x: 0.5, y: 0.5, label: 0 },
      { x: 0.8, y: 0.5, label: 1 },
    ]);
    core.tick(0.1);
    expect(core.lastPresentedIndex).toBe(0);
    core.tick(0.1);
    expect(core.lastPresentedIndex).toBe(1);
    core.tick(0.1);
    expect(core.lastPresentedIndex).toBe(2);
    expect(core.epochs).toBe(1);
    core.tick(0.1);
    expect(core.lastPresentedIndex).toBe(0);
    core.tick(0.1);
    core.tick(0.1);
    expect(core.epochs).toBe(2);
    expect(core.isConverged).toBe(false);
  });
});

describe('PerceptronCore point management', () => {
  it('resumes training after adding a point', () => {
    const core = new PerceptronCore(2);
    core.setData(separableDataset());
    trainUntil(core, 0.1, 5000);
    const epochsBefore = core.epochs;
    const ticksBefore = core.ticksElapsed;
    const lengthBefore = core.points.length;
    core.addPoint({ x: 0.5, y: 0.5, label: 1 });
    expect(core.isConverged).toBe(false);
    expect(core.stepsToConvergence).toBeNull();
    expect(core.points.length).toBe(lengthBefore + 1);
    expect(core.epochs).toBe(epochsBefore);
    expect(core.ticksElapsed).toBe(ticksBefore);
    core.tick(0.1);
    expect(core.ticksElapsed).toBe(ticksBefore + 1);
  });

  it('resets counters and weights while keeping the points', () => {
    const core = new PerceptronCore(4);
    const data = separableDataset();
    core.setData(data);
    for (let i = 0; i < 10; i += 1) {
      core.tick(0.1);
    }
    core.reset();
    expect(core.points).toEqual(data);
    expect(core.epochs).toBe(0);
    expect(core.ticksElapsed).toBe(0);
    expect(core.lastPresentedIndex).toBeNull();
    expect(core.isConverged).toBe(false);
    expect(core.stepsToConvergence).toBeNull();
    expect(Math.abs(core.weights.w1)).toBeLessThanOrEqual(0.5);
    expect(Math.abs(core.weights.w2)).toBeLessThanOrEqual(0.5);
    expect(Math.hypot(core.weights.w1, core.weights.w2)).toBeGreaterThanOrEqual(0.1);
  });

  it('clears points and makes tick a no-op', () => {
    const core = new PerceptronCore(6);
    core.setData(separableDataset());
    core.clearPoints();
    expect(core.points).toHaveLength(0);
    expect(core.accuracy).toBe(1);
    expect(core.misclassifiedIndices).toEqual([]);
    core.tick(0.1);
    expect(core.ticksElapsed).toBe(0);
    expect(core.lastPresentedIndex).toBeNull();
  });
});

describe('PerceptronCore metrics', () => {
  it('keeps accuracy, misclassifiedIndices and predictionAt consistent', () => {
    const points: LabeledPoint[] = [
      { x: 0.1, y: 0.1, label: 0 },
      { x: 0.9, y: 0.9, label: 1 },
      { x: 0.9, y: 0.1, label: 0 },
      { x: 0.1, y: 0.9, label: 1 },
    ];
    const core = new PerceptronCore(21);
    core.setData(points);
    const expected = points
      .map((point, index) => (core.predictionAt(point.x, point.y) !== point.label ? index : -1))
      .filter((index) => index >= 0);
    expect(core.misclassifiedIndices).toEqual(expected);
    expect(core.accuracy).toBeCloseTo((points.length - expected.length) / points.length);
    for (const index of core.misclassifiedIndices) {
      expect(core.predictionAt(points[index].x, points[index].y)).not.toBe(points[index].label);
    }
  });
});
