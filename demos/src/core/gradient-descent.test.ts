import { describe, expect, it } from 'vitest';
import {
  type Example,
  exampleGradient,
  exampleLoss,
  type Gradient2,
  GradientDescentCore,
  landscapeGradient,
  landscapeLoss,
} from './gradient-descent.js';

const C = GradientDescentCore.TARGET_C;
const D = GradientDescentCore.TARGET_D;

function makeExample(x1: number, x2: number): Example {
  return { x1, x2, t: C * x1 + D * x2 };
}

function numericLandscapeLoss(a: number, b: number, n = 400): number {
  const step = 2 / n;
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const x1 = -1 + (i + 0.5) * step;
    for (let j = 0; j < n; j += 1) {
      const x2 = -1 + (j + 0.5) * step;
      sum += exampleLoss(a, b, makeExample(x1, x2));
    }
  }
  return sum / (n * n);
}

function meanGradient(a: number, b: number, examples: Example[]): Gradient2 {
  let da = 0;
  let db = 0;
  for (const example of examples) {
    const gradient = exampleGradient(a, b, example);
    da += gradient.da;
    db += gradient.db;
  }
  return { da: da / examples.length, db: db / examples.length };
}

function numericPartial(
  fn: (a: number, b: number) => number,
  a: number,
  b: number,
  axis: 'da' | 'db',
  h = 1e-4,
): number {
  if (axis === 'da') {
    return (fn(a + h, b) - fn(a - h, b)) / (2 * h);
  }
  return (fn(a, b + h) - fn(a, b - h)) / (2 * h);
}

describe('landscape closed forms', () => {
  it('landscapeLoss matches numeric integration over the uniform square', () => {
    for (const [a, b] of [
      [0, 0],
      [1.2, -0.8],
      [-2.5, 2.2],
      [2.7, 1.4],
    ]) {
      expect(landscapeLoss(a, b)).toBeCloseTo(numericLandscapeLoss(a, b), 4);
    }
  });

  it('landscapeGradient matches numeric partial derivatives', () => {
    for (const [a, b] of [
      [0, 0],
      [-1.3, 0.7],
      [2.2, -1.9],
    ]) {
      const gradient = landscapeGradient(a, b);
      expect(gradient.da).toBeCloseTo(numericPartial(landscapeLoss, a, b, 'da'), 6);
      expect(gradient.db).toBeCloseTo(numericPartial(landscapeLoss, a, b, 'db'), 6);
    }
  });

  it('is minimized at the target (c, d) with a vanishing gradient', () => {
    expect(landscapeLoss(C, D)).toBeCloseTo(0, 12);
    const gradient = landscapeGradient(C, D);
    expect(gradient.da).toBeCloseTo(0, 12);
    expect(gradient.db).toBeCloseTo(0, 12);
  });

  it('exampleGradient matches numeric partial derivatives of exampleLoss', () => {
    const example = makeExample(0.4, -0.7);
    const gradient = exampleGradient(-0.3, 1.1, example);
    const loss = (a: number, b: number): number => exampleLoss(a, b, example);
    expect(gradient.da).toBeCloseTo(numericPartial(loss, -0.3, 1.1, 'da'), 6);
    expect(gradient.db).toBeCloseTo(numericPartial(loss, -0.3, 1.1, 'db'), 6);
  });
});

describe('deterministic examples', () => {
  it('produces the same batch for the same seed', () => {
    const first = new GradientDescentCore();
    first.reset({ seed: 7, batchSize: 5 });
    first.advance();

    const second = new GradientDescentCore();
    second.reset({ seed: 7, batchSize: 5 });
    second.advance();

    expect(first.batch).toEqual(second.batch);
    expect(first.epoch).toBe(0);
  });

  it('reproduces the epoch-1 batch deterministically for a given seed', () => {
    const first = new GradientDescentCore();
    first.reset({ seed: 99, batchSize: 3 });
    first.advanceBatch(0.15);

    const second = new GradientDescentCore();
    second.reset({ seed: 99, batchSize: 3 });
    second.advanceBatch(0.15);

    expect(first.epoch).toBe(1);
    expect(first.batch).toEqual(second.batch);
  });

  it('draws the batch with t = c*x1 + d*x2', () => {
    const core = new GradientDescentCore();
    core.reset({ seed: 3, batchSize: 6 });
    core.advance();
    for (const example of core.batch) {
      expect(example.t).toBeCloseTo(C * example.x1 + D * example.x2, 12);
      expect(example.x1).toBeGreaterThanOrEqual(-1);
      expect(example.x1).toBeLessThanOrEqual(1);
      expect(example.x2).toBeGreaterThanOrEqual(-1);
      expect(example.x2).toBeLessThanOrEqual(1);
    }
  });
});

describe('state machine', () => {
  it('walks one full batch of size 4 through every phase', () => {
    const core = new GradientDescentCore();
    core.reset({ batchSize: 4 });
    expect(core.phase).toBe('idle');

    core.advance(0.15);
    expect(core.phase).toBe('present');

    for (let i = 0; i < 4; i += 1) {
      core.advance(0.15);
      expect(core.phase).toBe('gradient');
      core.advance(0.15);
      expect(core.phase).toBe('fade');
      core.advance(0.15);
      expect(core.phase).toBe(i < 3 ? 'present' : 'mean');
    }

    core.advance(0.15);
    expect(core.phase).toBe('move');
    core.advance(0.15);
    expect(core.phase).toBe('present');
    expect(core.epoch).toBe(1);
  });

  it('accumulates one arrow per fade and exposes batchGradient only from mean on', () => {
    const core = new GradientDescentCore();
    core.reset({ batchSize: 4 });
    core.advance(0.15);
    expect(core.collected).toHaveLength(0);
    expect(core.batchGradient).toBeNull();

    for (let i = 0; i < 4; i += 1) {
      core.advance(0.15);
      expect(core.phase).toBe('gradient');
      expect(core.batchGradient).toBeNull();
      core.advance(0.15);
      expect(core.phase).toBe('fade');
      expect(core.collected).toHaveLength(i + 1);
      expect(core.batchGradient).toBeNull();
      core.advance(0.15);
    }

    expect(core.phase).toBe('mean');
    expect(core.collected).toHaveLength(4);
    expect(core.batchGradient).not.toBeNull();
  });

  it('keeps the ball fixed until the move step', () => {
    const core = new GradientDescentCore();
    core.reset({ startA: -2.5, startB: 2.2, batchSize: 4 });
    core.advance(0.15);
    const a = core.ballA;
    const b = core.ballB;

    while (core.phase !== 'mean') {
      core.advance(0.15);
      expect(core.ballA).toBe(a);
      expect(core.ballB).toBe(b);
    }

    core.advance(0.15);
    expect(core.phase).toBe('move');
  });
});

describe('batch gradient invariant', () => {
  it('sets batchGradient to the mean of the collected per-example gradients', () => {
    const core = new GradientDescentCore();
    core.reset({ startA: -2.5, startB: 2.2, batchSize: 4 });
    core.advance(0.15);
    const examples = core.batch.map((example) => ({ ...example }));
    const expected = meanGradient(core.ballA, core.ballB, examples);

    while (core.phase !== 'mean') {
      core.advance(0.15);
    }

    const batchGradient = core.batchGradient;
    expect(batchGradient).not.toBeNull();
    if (batchGradient === null) {
      throw new Error('batchGradient should be defined at the mean phase');
    }
    expect(batchGradient.da).toBeCloseTo(expected.da, 12);
    expect(batchGradient.db).toBeCloseTo(expected.db, 12);

    for (let i = 0; i < examples.length; i += 1) {
      const arrow = core.collected[i];
      const direct = exampleGradient(core.ballA, core.ballB, examples[i]);
      expect(arrow.da).toBeCloseTo(direct.da, 12);
      expect(arrow.db).toBeCloseTo(direct.db, 12);
    }
  });

  it('moves the ball by exactly -learningRate * batchGradient over advanceBatch', () => {
    const core = new GradientDescentCore();
    core.reset({ startA: 1.1, startB: -1.4, batchSize: 4, seed: 5 });
    core.advance(0.15);
    const examples = core.batch.map((example) => ({ ...example }));
    const fromA = core.ballA;
    const fromB = core.ballB;
    const expected = meanGradient(fromA, fromB, examples);

    const move = core.advanceBatch(0.15);
    expect(move).not.toBeNull();
    if (move === null) {
      throw new Error('advanceBatch should report a move');
    }

    expect(move.fromA).toBeCloseTo(fromA, 12);
    expect(move.fromB).toBeCloseTo(fromB, 12);
    expect(move.toA).toBeCloseTo(fromA - 0.15 * expected.da, 12);
    expect(move.toB).toBeCloseTo(fromB - 0.15 * expected.db, 12);
    expect(core.ballA).toBeCloseTo(move.toA, 12);
    expect(core.ballB).toBeCloseTo(move.toB, 12);
    expect(core.epoch).toBe(1);
    expect(core.phase).toBe('present');
  });
});

describe('coarse (play) mode', () => {
  it('cycles mean → move → mean, skipping per-example detail', () => {
    const core = new GradientDescentCore();
    core.reset({ startA: -2.5, startB: 2.2, batchSize: 4, seed: 5 });
    const startA = core.ballA;
    const startB = core.ballB;

    core.advance(0.15, { detailed: false });
    expect(core.phase).toBe('mean');
    expect(core.currentExample).toBeNull();
    expect(core.collected).toHaveLength(0);
    expect(core.batchGradient).not.toBeNull();
    expect(core.ballA).toBe(startA);
    expect(core.ballB).toBe(startB);

    core.advance(0.15, { detailed: false });
    expect(core.phase).toBe('move');
    expect(core.ballA).not.toBe(startA);
    expect(core.epoch).toBe(0);

    core.advance(0.15, { detailed: false });
    expect(core.phase).toBe('mean');
    expect(core.epoch).toBe(1);
    expect(core.collected).toHaveLength(0);
  });

  it('moves the ball by exactly -learningRate * batch mean each cycle', () => {
    const core = new GradientDescentCore();
    core.reset({ startA: 1.1, startB: -1.4, batchSize: 4, seed: 5 });
    core.advance(0.15, { detailed: false });
    const examples = core.batch.map((example) => ({ ...example }));
    const fromA = core.ballA;
    const fromB = core.ballB;
    const expected = meanGradient(fromA, fromB, examples);

    core.advance(0.15, { detailed: false });

    expect(core.phase).toBe('move');
    expect(core.ballA).toBeCloseTo(fromA - 0.15 * expected.da, 12);
    expect(core.ballB).toBeCloseTo(fromB - 0.15 * expected.db, 12);
  });

  it('draws a fresh deterministic batch each epoch', () => {
    const first = new GradientDescentCore();
    first.reset({ seed: 77, batchSize: 3 });
    const second = new GradientDescentCore();
    second.reset({ seed: 77, batchSize: 3 });

    for (let i = 0; i < 6; i += 1) {
      first.advance(0.15, { detailed: false });
      second.advance(0.15, { detailed: false });
    }

    expect(first.epoch).toBe(second.epoch);
    expect(first.batch).toEqual(second.batch);
    expect(first.ballA).toBeCloseTo(second.ballA, 12);
    expect(first.ballB).toBeCloseTo(second.ballB, 12);
  });
});

describe('loss and gradient readouts', () => {
  it('reports landscape loss and gradient magnitude at the ball position', () => {
    const a = 0.5;
    const b = -0.3;
    const core = new GradientDescentCore();
    core.reset({ startA: a, startB: b });

    expect(core.loss).toBeCloseTo(((a - C) ** 2 + (b - D) ** 2) / 6, 12);
    expect(core.gradientMagnitude).toBeCloseTo(Math.hypot((a - C) / 3, (b - D) / 3), 12);
  });
});
