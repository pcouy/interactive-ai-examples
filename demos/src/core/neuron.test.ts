import { describe, expect, it } from 'vitest';
import { mulberry32 } from './gradient-descent.js';
import { compileTarget, type LossKind, NeuronCore, type NeuronExample } from './neuron.js';

const EPOCH_STRIDE = 0x9e3779b1;

// The helpers mirror the implementation: pass b = 0 when bias is off, because
// the core then ignores the bias term in both the loss and the gradient.
function meanLoss(
  w1: number,
  w2: number,
  b: number,
  examples: readonly NeuronExample[],
  kind: LossKind,
): number {
  let sum = 0;
  for (const example of examples) {
    const err = w1 * example.x1 + w2 * example.x2 + b - example.t;
    sum += kind === 'l1' ? Math.abs(err) : 0.5 * err * err;
  }
  return sum / examples.length;
}

function meanGradient(
  w1: number,
  w2: number,
  b: number,
  examples: readonly NeuronExample[],
  kind: LossKind,
): { dw1: number; dw2: number; db: number } {
  let dw1 = 0;
  let dw2 = 0;
  let db = 0;
  for (const example of examples) {
    const err = w1 * example.x1 + w2 * example.x2 + b - example.t;
    const factor = kind === 'l1' ? Math.sign(err) : err;
    dw1 += factor * example.x1;
    dw2 += factor * example.x2;
    db += factor;
  }
  const n = examples.length;
  return { dw1: dw1 / n, dw2: dw2 / n, db: db / n };
}

function numericPartial(
  fn: (w1: number, w2: number) => number,
  w1: number,
  w2: number,
  axis: 'w1' | 'w2',
  h = 1e-4,
): number {
  if (axis === 'w1') {
    return (fn(w1 + h, w2) - fn(w1 - h, w2)) / (2 * h);
  }
  return (fn(w1, w2 + h) - fn(w1, w2 - h)) / (2 * h);
}

function makeCore(
  opts: {
    expression?: string;
    count?: number;
    seed?: number;
    loss?: LossKind;
    bias?: boolean;
  } = {},
): NeuronCore {
  const core = new NeuronCore();
  core.setDataset({ expression: opts.expression, count: opts.count, seed: opts.seed });
  if (opts.loss !== undefined) {
    core.setLoss(opts.loss);
  }
  if (opts.bias !== undefined) {
    core.setBiasEnabled(opts.bias);
  }
  return core;
}

function fitLinear(examples: readonly NeuronExample[]): { w1: number; w2: number } | null {
  let s11 = 0;
  let s12 = 0;
  let s22 = 0;
  let t1 = 0;
  let t2 = 0;
  for (const example of examples) {
    s11 += example.x1 * example.x1;
    s12 += example.x1 * example.x2;
    s22 += example.x2 * example.x2;
    t1 += example.x1 * example.t;
    t2 += example.x2 * example.t;
  }
  const det = s11 * s22 - s12 * s12;
  if (Math.abs(det) < 1e-12) {
    return null;
  }
  return { w1: (t1 * s22 - t2 * s12) / det, w2: (s11 * t2 - s12 * t1) / det };
}

describe('compileTarget', () => {
  it('evaluates a valid expression', () => {
    const first = compileTarget('x1 * 2 + x2');
    expect('fn' in first).toBe(true);
    if ('fn' in first) {
      expect(first.fn(0.3, -0.7)).toBeCloseTo(-0.1, 12);
    }

    const second = compileTarget('x1 + x2');
    expect('fn' in second).toBe(true);
    if ('fn' in second) {
      expect(second.fn(0.5, 0.25)).toBeCloseTo(0.75, 12);
    }
  });

  it('reports a syntax error without throwing', () => {
    for (const expression of ['x1 +', '']) {
      const result = compileTarget(expression);
      expect('error' in result).toBe(true);
      if ('error' in result) {
        expect(result.error).toBe('SyntaxError in expression');
      }
    }
  });

  it('reports a runtime error without throwing', () => {
    const result = compileTarget('foo()');
    expect('error' in result).toBe(true);
  });

  it('rejects a non-finite probe result', () => {
    for (const expression of ['1/0', 'NaN', '"abc"']) {
      const result = compileTarget(expression);
      expect('error' in result).toBe(true);
      if ('error' in result) {
        expect(result.error).toBe('Expression must return a finite number');
      }
    }
  });

  it('uses x1 + x2 as the default expression', () => {
    expect(NeuronCore.DEFAULT_EXPRESSION).toBe('x1 + x2');
  });
});

describe('dataset', () => {
  it('is deterministic per seed', () => {
    const first = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    const second = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    const other = makeCore({ expression: 'x1 + x2', count: 24, seed: 8 });

    expect(first.dataset).toEqual(second.dataset);
    expect(first.dataset).not.toEqual(other.dataset);
  });

  it('keeps x1 and x2 in [-1, 1] and targets t = x1 - x2', () => {
    const core = makeCore({ expression: 'x1 - x2', count: 24, seed: 7 });
    expect(core.dataset).toHaveLength(24);
    for (const example of core.dataset) {
      expect(example.x1).toBeGreaterThanOrEqual(-1);
      expect(example.x1).toBeLessThanOrEqual(1);
      expect(example.x2).toBeGreaterThanOrEqual(-1);
      expect(example.x2).toBeLessThanOrEqual(1);
      expect(example.t).toBeCloseTo(example.x1 - example.x2, 12);
    }
  });

  it('skips non-finite targets', () => {
    const core = makeCore({ expression: 'x1 < 0 ? x1 : Infinity', count: 40, seed: 42 });
    expect(core.dataset.length).toBeGreaterThan(0);
    expect(core.dataset.length).toBeLessThanOrEqual(40);
    for (const example of core.dataset) {
      expect(example.x1).toBeLessThan(0);
      expect(example.t).toBeCloseTo(example.x1, 12);
    }
  });

  it('falls back to the default expression on a bad expression', () => {
    const core = makeCore();
    core.setDataset({ expression: 'x1 +' });
    expect(core.expression).toBe('x1 + x2');
    expect(typeof core.targetError).toBe('string');
    expect(core.targetError).not.toBe('');
    for (const example of core.dataset) {
      expect(example.t).toBeCloseTo(example.x1 + example.x2, 12);
    }
  });

  it('keeps the expression when it is omitted', () => {
    const core = makeCore();
    core.setDataset({ expression: 'x1 * x2' });
    core.setDataset({ count: 10 });
    expect(core.expression).toBe('x1 * x2');
    expect(core.targetError).toBeNull();
  });

  it('handles an empty dataset', () => {
    const core = makeCore({ expression: 'x1 < 0 ? x1 : Infinity', count: 1, seed: 1 });
    expect(core.dataset).toHaveLength(0);
    expect(core.domain).toBe(5);
    expect(core.gradient).toEqual({ dw1: 0, dw2: 0, db: 0 });
    expect(core.loss).toBe(0);
    expect(core.advanceBatch(0.1)).toBeNull();
    expect(() => core.advance(0.1)).not.toThrow();
  });

  it('keeps the current count when count is not finite', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    expect(core.dataset).toHaveLength(24);

    core.setDataset({ count: NaN });
    expect(core.dataset).toHaveLength(24);

    core.setDataset({ count: Infinity });
    expect(core.dataset).toHaveLength(24);

    core.setDataset({ count: -Infinity });
    expect(core.dataset).toHaveLength(24);

    core.setDataset({ count: 10 });
    expect(core.dataset).toHaveLength(10);
  });
});

describe('losses', () => {
  it('matches the independent L2 mean loss', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    for (const [w1, w2] of [
      [0, 0],
      [0.5, -0.3],
      [1.2, -0.8],
      [-2.5, 2.2],
    ]) {
      expect(core.landscapeLoss(w1, w2)).toBeCloseTo(meanLoss(w1, w2, 0, core.dataset, 'l2'), 12);
    }
  });

  it('matches the independent L1 mean loss after setLoss', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    core.setLoss('l1');
    for (const [w1, w2] of [
      [0, 0],
      [0.5, -0.3],
      [1.2, -0.8],
    ]) {
      expect(core.landscapeLoss(w1, w2)).toBeCloseTo(meanLoss(w1, w2, 0, core.dataset, 'l1'), 12);
    }
  });

  it('uses the current bias when bias is on', () => {
    const core = makeCore({ bias: true });
    core.reset({ startB: 0.3 });
    expect(core.landscapeLoss(core.w1, core.w2)).toBeCloseTo(
      meanLoss(core.w1, core.w2, 0.3, core.dataset, 'l2'),
      12,
    );
  });
});

describe('gradient', () => {
  it('matches central differences of the L2 landscape', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    for (const [w1, w2] of [
      [0.5, -0.3],
      [1.2, -0.8],
      [-2.5, 2.2],
    ]) {
      core.reset({ startW1: w1, startW2: w2 });
      const landscape = (a: number, b: number): number => core.landscapeLoss(a, b);
      const gradient = core.gradient;
      expect(gradient.dw1).toBeCloseTo(numericPartial(landscape, w1, w2, 'w1'), 5);
      expect(gradient.dw2).toBeCloseTo(numericPartial(landscape, w1, w2, 'w2'), 5);
    }

    const biased = makeCore({ expression: 'x1 + x2', count: 24, seed: 7, bias: true });
    biased.reset({ startW1: 0.5, startW2: -0.3, startB: 0.2 });
    const h = 1e-5;
    const db =
      (meanLoss(0.5, -0.3, 0.2 + h, biased.dataset, 'l2') -
        meanLoss(0.5, -0.3, 0.2 - h, biased.dataset, 'l2')) /
      (2 * h);
    expect(biased.gradient.db).toBeCloseTo(db, 5);
  });

  it('matches central differences of the L1 landscape away from err = 0', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    core.setLoss('l1');
    const h = 1e-5;
    for (const [w1, w2] of [
      [0.5, -0.3],
      [2, 3],
    ]) {
      core.reset({ startW1: w1, startW2: w2 });
      for (const example of core.dataset) {
        expect(Math.abs(w1 * example.x1 + w2 * example.x2 - example.t)).toBeGreaterThan(1e-3);
      }
      const landscape = (a: number, b: number): number => core.landscapeLoss(a, b);
      expect(core.gradient.dw1).toBeCloseTo(numericPartial(landscape, w1, w2, 'w1', h), 5);
      expect(core.gradient.dw2).toBeCloseTo(numericPartial(landscape, w1, w2, 'w2', h), 5);
    }
  });

  it('reports a zero bias gradient when bias is off', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    core.reset({ startW1: 0.5, startW2: -0.3 });
    expect(core.biasEnabled).toBe(false);
    expect(core.gradient.db).toBe(0);
  });
});

describe('bias', () => {
  it('keeps the bias at zero when bias is off', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    for (let i = 0; i < 5; i += 1) {
      core.advanceBatch(0.1);
    }
    expect(core.b).toBe(0);
  });

  it('moves the bias when bias is on', () => {
    const core = makeCore({ expression: 'x1 + x2 + 0.5', count: 24, seed: 7, bias: true });
    core.reset({ startB: 0 });
    for (let i = 0; i < 5; i += 1) {
      core.advanceBatch(0.1);
    }
    expect(core.b).not.toBe(0);
  });
});

describe('phase machine', () => {
  it('walks one batch through every phase', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    core.setBatchSize(4);
    core.reset();
    expect(core.phase).toBe('idle');

    core.advance(0.15);
    expect(core.phase).toBe('present');
    expect(core.exampleIndex).toBe(0);
    expect(core.currentExample).toEqual(core.batch[0]);

    for (let i = 0; i < 4; i += 1) {
      core.advance(0.15);
      expect(core.phase).toBe('gradient');
      core.advance(0.15);
      expect(core.phase).toBe('fade');
      core.advance(0.15);
      expect(core.phase).toBe(i < 3 ? 'present' : 'mean');
      if (i < 3) {
        expect(core.exampleIndex).toBe(i + 1);
      }
    }

    core.advance(0.15);
    expect(core.phase).toBe('move');
    core.advance(0.15);
    expect(core.phase).toBe('present');
    expect(core.epoch).toBe(1);
  });

  it('applies one batch in coarse mode', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 5 });
    core.reset({ startW1: 1.1, startW2: -1.4 });
    core.advance(0.15, { detailed: false });
    expect(core.phase).toBe('mean');

    const examples = core.batch.map((example) => ({ ...example }));
    const fromW1 = core.w1;
    const fromW2 = core.w2;
    const expected = meanGradient(fromW1, fromW2, 0, examples, 'l2');

    core.advance(0.15, { detailed: false });
    expect(core.phase).toBe('move');
    expect(core.w1).toBeCloseTo(fromW1 - 0.15 * expected.dw1, 12);
    expect(core.w2).toBeCloseTo(fromW2 - 0.15 * expected.dw2, 12);

    core.advance(0.15, { detailed: false });
    expect(core.phase).toBe('present');
    expect(core.epoch).toBe(1);
  });

  it('moves by -learningRate times the epoch batch mean', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.reset({ startW1: 1.1, startW2: -1.4, startB: 0.2 });
    expect(core.b).toBe(0);

    const rng = mulberry32(5 + 0 * EPOCH_STRIDE);
    const n = core.dataset.length;
    const batch: NeuronExample[] = [];
    for (let i = 0; i < 4; i += 1) {
      const index = Math.min(n - 1, Math.floor(rng() * n));
      batch.push({ ...core.dataset[index] });
    }
    const expected = meanGradient(1.1, -1.4, 0, batch, 'l2');

    const move = core.advanceBatch(0.15);
    expect(move).not.toBeNull();
    if (move === null) {
      throw new Error('advanceBatch should report a move');
    }
    expect(move.fromA).toBeCloseTo(1.1, 12);
    expect(move.fromB).toBeCloseTo(-1.4, 12);
    expect(move.toA).toBeCloseTo(1.1 - 0.15 * expected.dw1, 12);
    expect(move.toB).toBeCloseTo(-1.4 - 0.15 * expected.dw2, 12);
    expect(core.epoch).toBe(1);
    expect(core.phase).toBe('present');
  });

  it('returns null when diverged or empty', () => {
    const empty = makeCore({ expression: 'x1 < 0 ? x1 : Infinity', count: 1, seed: 1 });
    expect(empty.advanceBatch(0.1)).toBeNull();

    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.advanceBatch(1e6);
    expect(core.diverged).toBe(true);
    expect(core.advanceBatch(0.1)).toBeNull();
  });

  it('applies the same batch-mean update in detailed and coarse mode', () => {
    const configs: Array<{ bias: boolean; loss: LossKind }> = [
      { bias: false, loss: 'l2' },
      { bias: true, loss: 'l2' },
      { bias: false, loss: 'l1' },
      { bias: true, loss: 'l1' },
    ];
    for (const { bias, loss } of configs) {
      const detailed = makeCore({ expression: 'x1 + x2', count: 24, seed: 7, bias, loss });
      const coarse = makeCore({ expression: 'x1 + x2', count: 24, seed: 7, bias, loss });
      detailed.reset({ startW1: 1.1, startW2: -1.4, startB: 0.2 });
      coarse.reset({ startW1: 1.1, startW2: -1.4, startB: 0.2 });

      // The first advance draws the epoch batch; snapshot it before walking.
      detailed.advance(0.15);
      const batch = detailed.batch.map((example) => ({ ...example }));
      const startW1 = detailed.w1;
      const startW2 = detailed.w2;
      const startB = detailed.b;

      let steps = 0;
      while (detailed.epoch < 1 && steps < 1000) {
        detailed.advance(0.15);
        steps += 1;
      }
      expect(detailed.epoch).toBe(1);

      coarse.advanceBatch(0.15);
      expect(coarse.epoch).toBe(1);

      expect(detailed.w1).toBeCloseTo(coarse.w1, 12);
      expect(detailed.w2).toBeCloseTo(coarse.w2, 12);
      expect(detailed.b).toBeCloseTo(coarse.b, 12);

      const expected = meanGradient(startW1, startW2, startB, batch, loss);
      expect(detailed.w1).toBeCloseTo(startW1 - 0.15 * expected.dw1, 12);
      expect(detailed.w2).toBeCloseTo(startW2 - 0.15 * expected.dw2, 12);
      if (bias) {
        expect(detailed.b).toBeCloseTo(startB - 0.15 * expected.db, 12);
      } else {
        expect(detailed.b).toBe(0);
      }
    }
  });

  it('treats a non-finite learning rate as a no-op', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7, bias: true });
    core.reset({ startW1: 0.5, startW2: -0.3, startB: 0.2 });

    for (const learningRate of [NaN, Infinity, -Infinity]) {
      core.advance(learningRate);
      core.advance(learningRate, { detailed: false });
      expect(core.advanceBatch(learningRate)).toBeNull();
      expect(core.phase).toBe('idle');
      expect(core.w1).toBe(0.5);
      expect(core.w2).toBe(-0.3);
      expect(core.b).toBe(0.2);
      expect(core.epoch).toBe(0);
      expect(core.diverged).toBe(false);
    }

    core.advanceBatch(0.1);
    expect(core.epoch).toBe(1);
    expect(core.advanceBatch(NaN)).toBeNull();
    expect(core.epoch).toBe(1);
    expect(core.phase).toBe('present');
  });
});

describe('determinism and reset', () => {
  it('produces the same trajectory for the same seed', () => {
    const first = makeCore({ expression: 'x1 + x2', count: 24, seed: 11 });
    const second = makeCore({ expression: 'x1 + x2', count: 24, seed: 11 });
    for (let i = 0; i < 10; i += 1) {
      first.advanceBatch(0.1);
      second.advanceBatch(0.1);
    }
    expect(first.w1).toEqual(second.w1);
    expect(first.w2).toEqual(second.w2);
    expect(first.b).toEqual(second.b);
    expect(first.epoch).toEqual(second.epoch);
  });

  it('resets to the given start weights', () => {
    const core = makeCore({ bias: true });
    core.reset({ startW1: 0.3, startW2: -0.2, startB: 0.1 });
    expect(core.w1).toBe(0.3);
    expect(core.w2).toBe(-0.2);
    expect(core.b).toBe(0.1);
    expect(core.epoch).toBe(0);
    expect(core.phase).toBe('idle');
    expect(core.diverged).toBe(false);
  });

  it('initializes random weights deterministically per seed', () => {
    const first = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    const second = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    expect(first.w1).toEqual(second.w1);
    expect(first.w2).toEqual(second.w2);
    expect(Math.hypot(first.w1, first.w2)).toBeGreaterThanOrEqual(0.1);
  });
});

describe('domain', () => {
  it('derives the domain from the least-squares fit', () => {
    const two = makeCore({ expression: '2*x1', count: 40, seed: 7 });
    const fitTwo = fitLinear(two.dataset);
    expect(fitTwo).not.toBeNull();
    if (fitTwo !== null) {
      expect(fitTwo.w1).toBeCloseTo(2, 10);
    }
    expect(two.domain).toBe(5);

    const twelve = makeCore({ expression: '12*x1', count: 40, seed: 7 });
    expect(twelve.domain).toBeCloseTo(18, 10);

    const twentyFive = makeCore({ expression: '25*x1', count: 40, seed: 7 });
    expect(twentyFive.domain).toBe(20);
  });

  it('uses the fallback domain when the fit is rank-deficient or empty', () => {
    const one = makeCore({ expression: 'x1 + x2', count: 1, seed: 7 });
    expect(one.domain).toBe(5);

    const twoBias = makeCore({ expression: 'x1 + x2', count: 2, seed: 7, bias: true });
    expect(twoBias.domain).toBe(5);

    const empty = makeCore({ expression: 'x1 < 0 ? x1 : Infinity', count: 1, seed: 1 });
    expect(empty.domain).toBe(5);
  });
});

describe('divergence', () => {
  it('snaps back and reports divergence on a huge learning rate', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    const startW1 = core.w1;
    const startW2 = core.w2;

    const move = core.advanceBatch(1e6);
    expect(core.diverged).toBe(true);
    expect(core.w1).toBe(startW1);
    expect(core.w2).toBe(startW2);
    expect(move).not.toBeNull();
    if (move === null) {
      throw new Error('advanceBatch should report the snap-back move');
    }
    expect(move.toA).toBeCloseTo(startW1, 12);
    expect(move.toB).toBeCloseTo(startW2, 12);
    expect(Math.abs(move.fromA)).toBeGreaterThan(10 * core.domain);

    expect(core.advanceBatch(0.1)).toBeNull();
    core.advance(0.1);
    expect(core.w1).toBe(startW1);
    expect(core.w2).toBe(startW2);

    core.reset({ startW1: 0.1, startW2: 0.2 });
    expect(core.diverged).toBe(false);
  });

  it('does not diverge on a large-offset target with bias', () => {
    const core = makeCore({ expression: 'x1 + x2 + 100', count: 40, bias: true });
    expect(core.domain).toBe(20);
    core.reset({ startB: 0 });

    for (let i = 0; i < 10; i += 1) {
      core.advanceBatch(0.05);
    }
    const bAfter10 = core.b;
    for (let i = 0; i < 10; i += 1) {
      core.advanceBatch(0.05);
    }

    expect(core.diverged).toBe(false);
    expect(core.epoch).toBe(20);
    expect(core.b).toBeGreaterThan(bAfter10);
    expect(core.b).toBeGreaterThan(50);
    expect(core.b).toBeLessThan(200);
  });
});

describe('readouts', () => {
  it('reports loss and gradient magnitude at the current weights', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    core.reset({ startW1: 0.5, startW2: -0.3 });
    expect(core.loss).toBeCloseTo(meanLoss(0.5, -0.3, 0, core.dataset, 'l2'), 12);
    const gradient = meanGradient(0.5, -0.3, 0, core.dataset, 'l2');
    expect(core.gradientMagnitude).toBeCloseTo(Math.hypot(gradient.dw1, gradient.dw2), 12);
  });
});
