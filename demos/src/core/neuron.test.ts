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
    inputDomain?: number;
  } = {},
): NeuronCore {
  const core = new NeuronCore();
  core.setDataset({
    expression: opts.expression,
    count: opts.count,
    seed: opts.seed,
    inputDomain: opts.inputDomain,
  });
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
      expect(core.lastMove).toBeNull();
      expect(core.phase).toBe('idle');
      expect(core.w1).toBe(0.5);
      expect(core.w2).toBe(-0.3);
      expect(core.b).toBe(0.2);
      expect(core.epoch).toBe(0);
      expect(core.diverged).toBe(false);
    }

    core.advanceBatch(0.1);
    expect(core.epoch).toBe(1);
    expect(core.lastMove).not.toBeNull();
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

  it('gives the minimum domain for a constant target', () => {
    const core = makeCore({ expression: '1', count: 40, bias: true });
    expect(core.domain).toBe(5);
    const noBias = makeCore({ expression: '1', count: 40, bias: false });
    expect(noBias.domain).toBe(5);
  });
});

describe('input domain', () => {
  it('defaults to 1', () => {
    const core = makeCore();
    expect(core.inputDomain).toBe(1);
  });

  it('draws both inputs in [-d, d] with matching targets', () => {
    for (const d of [0.1, 2, 5, 20]) {
      const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 7, inputDomain: d });
      expect(core.inputDomain).toBe(d);
      expect(core.dataset).toHaveLength(40);
      for (const example of core.dataset) {
        expect(Math.abs(example.x1)).toBeLessThanOrEqual(d);
        expect(Math.abs(example.x2)).toBeLessThanOrEqual(d);
        expect(example.t).toBeCloseTo(example.x1 + example.x2, 12);
      }
    }
  });

  it('clamps the input domain and keeps the current value when not finite', () => {
    const core = makeCore();
    core.setDataset({ inputDomain: 0 });
    expect(core.inputDomain).toBe(0.1);
    core.setDataset({ inputDomain: -5 });
    expect(core.inputDomain).toBe(0.1);
    core.setDataset({ inputDomain: 100 });
    expect(core.inputDomain).toBe(20);
    core.setDataset({ inputDomain: NaN });
    expect(core.inputDomain).toBe(20);
    core.setDataset({ inputDomain: 2.5 });
    expect(core.inputDomain).toBe(2.5);
  });

  it('rescales the dataset deterministically with the domain', () => {
    const unit = makeCore({ expression: 'x1 + x2', count: 40, seed: 7, inputDomain: 1 });
    const scaled = makeCore({ expression: 'x1 + x2', count: 40, seed: 7, inputDomain: 5 });
    expect(scaled.dataset.length).toBe(unit.dataset.length);
    for (let i = 0; i < unit.dataset.length; i += 1) {
      const base = unit.dataset[i];
      const big = scaled.dataset[i];
      expect(big.x1).toBeCloseTo(base.x1 * 5, 12);
      expect(big.x2).toBeCloseTo(base.x2 * 5, 12);
      expect(big.t).toBeCloseTo(base.t * 5, 12);
    }
  });

  it('resets training when the input domain changes', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    for (let i = 0; i < 5; i += 1) {
      core.advanceBatch(0.1);
    }
    expect(core.epoch).toBe(5);

    core.setDataset({ inputDomain: 5 });
    expect(core.dataset.some((example) => Math.abs(example.x1) > 1)).toBe(true);
    expect(core.epoch).toBe(0);
    expect(core.phase).toBe('idle');
    expect(core.diverged).toBe(false);
    for (const example of core.dataset) {
      expect(Math.abs(example.x1)).toBeLessThanOrEqual(5);
      expect(Math.abs(example.x2)).toBeLessThanOrEqual(5);
    }
  });

  it('derives the weight domain from the scaled dataset', () => {
    const core = makeCore({ expression: '12*x1', count: 40, seed: 7, inputDomain: 5 });
    expect(core.domain).toBeCloseTo(18, 10);
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

  it('continues training after a divergence when stop-on-diverge is off', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.setStopOnDiverge(false);
    const startW1 = core.w1;
    const startW2 = core.w2;

    const move = core.advanceBatch(1e6);
    expect(move).not.toBeNull();
    expect(core.diverged).toBe(false);
    expect(core.w1).toBe(startW1);
    expect(core.w2).toBe(startW2);
    expect(core.divergenceCount).toBeGreaterThan(0);

    const epoch = core.epoch;
    expect(core.advanceBatch(0.1)).not.toBeNull();
    expect(core.epoch).toBe(epoch + 1);
  });

  it('counts every divergence in halting and continue mode', () => {
    const halting = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    expect(halting.divergenceCount).toBe(0);
    halting.advanceBatch(1e6);
    expect(halting.divergenceCount).toBe(1);
    expect(halting.diverged).toBe(true);

    const continuing = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    continuing.setStopOnDiverge(false);
    continuing.advanceBatch(1e6);
    expect(continuing.divergenceCount).toBe(1);
    continuing.advanceBatch(1e6);
    expect(continuing.divergenceCount).toBe(2);
    expect(continuing.diverged).toBe(false);
  });

  it('disabling stop-on-diverge clears a latched divergence', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.advanceBatch(1e6);
    expect(core.diverged).toBe(true);

    core.setStopOnDiverge(false);
    expect(core.diverged).toBe(false);
    expect(core.advanceBatch(0.1)).not.toBeNull();
    expect(core.epoch).toBe(1);
  });

  it('enabling stop-on-diverge does not retro-halt', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.setStopOnDiverge(false);
    core.advanceBatch(1e6);
    expect(core.diverged).toBe(false);

    core.setStopOnDiverge(true);
    expect(core.diverged).toBe(false);
    expect(core.advanceBatch(0.1)).not.toBeNull();
  });

  it('reset and setWeights clear the divergence count', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.advanceBatch(1e6);
    expect(core.divergenceCount).toBe(1);

    core.reset({ startW1: 0.1, startW2: 0.2 });
    expect(core.divergenceCount).toBe(0);

    core.advanceBatch(1e6);
    expect(core.divergenceCount).toBe(1);

    core.setWeights({ w1: 0.1, w2: 0.2 });
    expect(core.divergenceCount).toBe(0);
  });
});

describe('setWeights', () => {
  it('applies finite weights and uses them as the divergence reference', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5, bias: true });
    core.setWeights({ w1: 0.7, w2: -0.9, b: 0.3 });
    expect(core.w1).toBe(0.7);
    expect(core.w2).toBe(-0.9);
    expect(core.b).toBe(0.3);
    expect(core.phase).toBe('idle');
    expect(core.epoch).toBe(0);

    const move = core.advanceBatch(1e6);
    expect(core.diverged).toBe(true);
    expect(core.w1).toBe(0.7);
    expect(core.w2).toBe(-0.9);
    expect(core.b).toBe(0.3);
    expect(move).not.toBeNull();
    if (move === null) {
      throw new Error('advanceBatch should report the snap-back move');
    }
    expect(move.toA).toBeCloseTo(0.7, 12);
    expect(move.toB).toBeCloseTo(-0.9, 12);
  });

  it('ignores non-finite values', () => {
    const core = makeCore({ bias: true });
    core.setWeights({ w1: 0.4, w2: -0.6, b: 0.2 });
    core.setWeights({ w1: NaN, w2: Infinity, b: -Infinity });
    expect(core.w1).toBe(0.4);
    expect(core.w2).toBe(-0.6);
    expect(core.b).toBe(0.2);
  });

  it('ignores b and keeps it 0 when bias is off', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 7 });
    expect(core.biasEnabled).toBe(false);
    core.setWeights({ w1: 0.5, w2: -0.4, b: 0.9 });
    expect(core.w1).toBe(0.5);
    expect(core.w2).toBe(-0.4);
    expect(core.b).toBe(0);
  });

  it('clears a prior divergence', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.advanceBatch(1e6);
    expect(core.diverged).toBe(true);

    core.setWeights({ w1: 0.2, w2: 0.3 });
    expect(core.diverged).toBe(false);
    expect(core.w1).toBe(0.2);
    expect(core.w2).toBe(0.3);
    expect(core.phase).toBe('idle');
    expect(core.epoch).toBe(0);
    expect(core.lastMove).toBeNull();
  });

  it('restarts the batch state and epoch', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 7, bias: true });
    core.advanceBatch(0.1);
    expect(core.epoch).toBe(1);
    expect(core.phase).not.toBe('idle');

    core.setWeights({ w1: 0.5 });
    expect(core.w1).toBe(0.5);
    expect(core.phase).toBe('idle');
    expect(core.batch).toHaveLength(0);
    expect(core.epoch).toBe(0);
    expect(core.lastMove).toBeNull();
    expect(core.diverged).toBe(false);
  });

  it('clamps manual weights against the current weight domain', () => {
    const core = makeCore({ expression: 'x1*x2', count: 40, seed: 5, bias: true });
    const defaultLimit = 10 * core.domain;

    core.setDataset({ inputDomain: 20 });
    const newLimit = 10 * core.domain;
    expect(newLimit).not.toBe(defaultLimit);

    core.setWeights({ w1: 1e6 });
    expect(core.w1).toBe(newLimit);
  });

  it('clamps huge values to the divergence limit', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.setWeights({ w1: 1e6 });
    expect(Math.abs(core.w1)).toBeLessThanOrEqual(10 * core.domain);
    expect(core.w1).toBe(10 * core.domain);

    core.setWeights({ w2: -1e6 });
    expect(Math.abs(core.w2)).toBeLessThanOrEqual(10 * core.domain);
    expect(core.w2).toBe(-10 * core.domain);
  });

  it('keeps the weights when no key is given', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 7, bias: true });
    core.setWeights({ w1: 0.6, w2: -0.2, b: 0.15 });
    core.advanceBatch(0.1);
    const w1 = core.w1;
    const w2 = core.w2;
    const b = core.b;

    core.setWeights({});
    expect(core.w1).toBe(w1);
    expect(core.w2).toBe(w2);
    expect(core.b).toBe(b);
    expect(core.phase).toBe('idle');
    expect(core.epoch).toBe(0);
    expect(core.batch).toHaveLength(0);
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

describe('evaluate', () => {
  it('matches the independent L2 formula', () => {
    const core = makeCore({ expression: 'x1 * x2 + 1', count: 24, seed: 7 });
    core.reset({ startW1: 0.5, startW2: -0.3 });

    const result = core.evaluate(0.4, -0.6);

    // Independent arithmetic, not read back from the core.
    const expectedTarget = 0.4 * -0.6 + 1;
    const expectedPredicted = 0.5 * 0.4 + -0.3 * -0.6;
    const expectedLoss =
      0.5 * (expectedPredicted - expectedTarget) * (expectedPredicted - expectedTarget);

    expect(result.x1).toBe(0.4);
    expect(result.x2).toBe(-0.6);
    expect(result.target).toBeCloseTo(expectedTarget, 12);
    expect(result.predicted).toBeCloseTo(expectedPredicted, 12);
    expect(result.loss).toBeCloseTo(expectedLoss, 12);

    expect(result.target).toBeCloseTo(0.76, 12);
    expect(result.predicted).toBeCloseTo(0.38, 12);
    expect(result.loss).toBeCloseTo(0.0722, 12);
  });

  it('uses the L1 loss after setLoss', () => {
    const core = makeCore({ expression: 'x1 * x2 + 1', count: 24, seed: 7 });
    core.reset({ startW1: 0.5, startW2: -0.3 });
    core.setLoss('l1');

    const result = core.evaluate(0.4, -0.6);

    expect(result.loss).toBeCloseTo(Math.abs(result.predicted - result.target), 12);
    expect(result.loss).toBeCloseTo(0.38, 12);
  });

  it('includes the bias only when bias is enabled', () => {
    const core = makeCore({ expression: 'x1 * x2 + 1', count: 24, seed: 7 });
    core.setWeights({ w1: 0.5, w2: -0.3, b: 0.9 });
    expect(core.b).toBe(0);

    const withoutBias = core.evaluate(0.4, -0.6);
    expect(withoutBias.predicted).toBeCloseTo(0.5 * 0.4 + -0.3 * -0.6, 12);
    expect(withoutBias.predicted).toBeCloseTo(0.38, 12);

    core.setBiasEnabled(true);
    core.reset({ startW1: 0.5, startW2: -0.3, startB: 0.9 });
    expect(core.b).toBe(0.9);

    const withBias = core.evaluate(0.4, -0.6);
    expect(withBias.predicted).toBeCloseTo(0.5 * 0.4 + -0.3 * -0.6 + 0.9, 12);
    expect(withBias.predicted).toBeCloseTo(1.28, 12);
  });

  it('reads the live weights after each batch', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 7 });
    core.reset({ startW1: 0.5, startW2: -0.3 });

    const before = core.evaluate(0.3, 0.2);
    expect(before.predicted).toBeCloseTo(0.5 * 0.3 + -0.3 * 0.2, 12);

    for (let i = 0; i < 5; i += 1) {
      core.advanceBatch(0.1);
    }

    const after = core.evaluate(0.3, 0.2);
    // Pins the deterministic trajectory (seed 7, count 40, lr 0.1, 5 batches
    // from w1=0.5, w2=-0.3) instead of reading the live weights, so a
    // regression in the update rule cannot move the expectation with it.
    expect(after.predicted).toBeCloseTo(0.1805728909330556, 12);
    expect(after.predicted).not.toBe(before.predicted);
  });

  it('falls back to x1 + x2 after a bad expression', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    core.setDataset({ expression: 'x1 +' });

    const result = core.evaluate(0.4, -0.6);

    expect(core.expression).toBe('x1 + x2');
    expect(result.target).toBeCloseTo(0.4 + -0.6, 12);
    expect(result.target).toBeCloseTo(-0.2, 12);
  });

  it('returns an infinite target and loss at a singular point', () => {
    const core = makeCore({ expression: '1/x1', count: 24, seed: 7 });

    const result = core.evaluate(0, 0.5);

    expect(result.target).toBe(Number.POSITIVE_INFINITY);
    expect(result.loss).toBe(Number.POSITIVE_INFINITY);
  });

  it('returns NaN when the target is not a number', () => {
    const core = makeCore({ expression: 'x1 > 1 ? "abc" : 1', count: 24, seed: 7 });

    const result = core.evaluate(2, 0.5);

    expect(result.target).toBeNaN();
    expect(result.loss).toBeNaN();
  });

  it('propagates a non-finite input as NaN', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 24, seed: 7 });
    core.reset({ startW1: 0.5, startW2: -0.3 });

    const result = core.evaluate(Number.NaN, 0.5);

    expect(result.predicted).toBeNaN();
    expect(result.loss).toBeNaN();
  });

  it('evaluates a point outside the input domain without clamping', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5, inputDomain: 1 });
    core.reset({ startW1: 0.5, startW2: -0.3 });

    const result = core.evaluate(50, -50);

    expect(result.target).toBeCloseTo(0, 12);
    expect(result.predicted).toBeCloseTo(0.5 * 50 + -0.3 * -50, 12);
    expect(result.predicted).toBeCloseTo(40, 12);
  });
});

describe('point landscape', () => {
  // Independent arithmetic: pass b = 0 when bias is off, because the core
  // then ignores the bias term.
  function pointLoss(
    w1: number,
    w2: number,
    b: number,
    x1: number,
    x2: number,
    target: number,
    kind: LossKind,
  ): number {
    const err = w1 * x1 + w2 * x2 + b - target;
    return kind === 'l1' ? Math.abs(err) : 0.5 * err * err;
  }

  it('matches the independent L2 formula at several weights', () => {
    const core = makeCore({ expression: 'x1 * x2 + 1', count: 24, seed: 7 });
    const x1 = 0.4;
    const x2 = -0.6;
    const target = 0.4 * -0.6 + 1;
    for (const [w1, w2] of [
      [0, 0],
      [0.5, -0.3],
      [1.2, -0.8],
      [-2.5, 2.2],
    ]) {
      expect(core.landscapeLossForPoint(w1, w2, x1, x2)).toBeCloseTo(
        pointLoss(w1, w2, 0, x1, x2, target, 'l2'),
        12,
      );
    }
  });

  it('matches the independent L1 formula after setLoss', () => {
    const core = makeCore({ expression: 'x1 * x2 + 1', count: 24, seed: 7 });
    core.setLoss('l1');
    const x1 = 0.4;
    const x2 = -0.6;
    const target = 0.4 * -0.6 + 1;
    for (const [w1, w2] of [
      [0, 0],
      [0.5, -0.3],
      [1.2, -0.8],
    ]) {
      expect(core.landscapeLossForPoint(w1, w2, x1, x2)).toBeCloseTo(
        pointLoss(w1, w2, 0, x1, x2, target, 'l1'),
        12,
      );
    }
  });

  it('uses the current bias when bias is on and ignores it when off', () => {
    const core = makeCore({ expression: 'x1 * x2 + 1', count: 24, seed: 7 });
    const x1 = 0.4;
    const x2 = -0.6;
    const target = 0.4 * -0.6 + 1;

    // setWeights with b and bias off keeps b at 0.
    core.setWeights({ w1: 0.5, w2: -0.3, b: 0.9 });
    expect(core.b).toBe(0);
    expect(core.landscapeLossForPoint(0.5, -0.3, x1, x2)).toBeCloseTo(
      pointLoss(0.5, -0.3, 0, x1, x2, target, 'l2'),
      12,
    );

    // Bias on uses the stored b.
    core.setBiasEnabled(true);
    core.reset({ startW1: 0.5, startW2: -0.3, startB: 0.9 });
    expect(core.b).toBe(0.9);
    expect(core.landscapeLossForPoint(0.5, -0.3, x1, x2)).toBeCloseTo(
      pointLoss(0.5, -0.3, 0.9, x1, x2, target, 'l2'),
      12,
    );
  });

  it('matches evaluate loss at the current weights', () => {
    const core = makeCore({ expression: 'x1 * x2 + 1', count: 24, seed: 7 });
    core.reset({ startW1: 0.5, startW2: -0.3 });
    const result = core.evaluate(0.4, -0.6);
    expect(core.landscapeLossForPoint(core.w1, core.w2, 0.4, -0.6)).toBeCloseTo(result.loss, 12);

    const biased = makeCore({ expression: 'x1 * x2 + 1', count: 24, seed: 7, bias: true });
    biased.reset({ startW1: 0.5, startW2: -0.3, startB: 0.9 });
    const biasedResult = biased.evaluate(0.4, -0.6);
    expect(biased.landscapeLossForPoint(biased.w1, biased.w2, 0.4, -0.6)).toBeCloseTo(
      biasedResult.loss,
      12,
    );
  });

  it('stays valid for a point outside the input domain without clamping', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5, inputDomain: 1 });
    core.reset({ startW1: 0.5, startW2: -0.3 });
    const x1 = 50;
    const x2 = -50;
    const target = 0;
    expect(core.landscapeLossForPoint(0.5, -0.3, x1, x2)).toBeCloseTo(
      pointLoss(0.5, -0.3, 0, x1, x2, target, 'l2'),
      12,
    );
    expect(core.landscapeLossForPoint(0.5, -0.3, x1, x2)).toBeCloseTo(800, 12);
  });

  it('yields a non-finite loss for a non-finite target', () => {
    const core = makeCore({ expression: '1/x1', count: 24, seed: 7 });
    core.reset({ startW1: 0.5, startW2: -0.3 });

    const loss = core.landscapeLossForPoint(0.5, -0.3, 0, 0.5);

    expect(Number.isFinite(loss)).toBe(false);
    expect(loss).toBe(Number.POSITIVE_INFINITY);
  });

  it('yields a NaN loss when the target is not a number', () => {
    const core = makeCore({ expression: 'x1 > 1 ? "abc" : 1', count: 24, seed: 7 });

    expect(core.landscapeLossForPoint(0.5, -0.3, 2, 0.5)).toBeNaN();
  });
});

describe('landscape moments', () => {
  it('matches the independent L2 mean loss across random datasets', () => {
    const rng = mulberry32(20240517);
    const expressions = ['x1 + x2', 'x1 * x2', '2*x1 - 0.5', '1.2*x1 - 0.8*x2 + 0.3'];
    for (let i = 0; i < 25; i += 1) {
      const expression = expressions[Math.floor(rng() * expressions.length)] ?? 'x1 + x2';
      const count = 1 + Math.floor(rng() * 60);
      const bias = rng() < 0.5;
      const core = makeCore({ expression, count, seed: 100 + i, bias });
      if (bias) {
        core.reset({ startB: rng() * 2 - 1 });
      }
      const b = core.b;
      for (let j = 0; j < 8; j += 1) {
        const w1 = (rng() * 2 - 1) * 5;
        const w2 = (rng() * 2 - 1) * 5;
        const expected = meanLoss(w1, w2, b, core.dataset, 'l2');
        const tolerance = 1e-9 * Math.max(1, Math.abs(expected));
        expect(Math.abs(core.landscapeLoss(w1, w2) - expected)).toBeLessThan(tolerance);
      }
    }
  });

  it('stays non-negative at an exact fit', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 7 });
    const loss = core.landscapeLoss(1, 1);
    expect(loss).toBeGreaterThanOrEqual(0);
    expect(loss).toBeLessThan(1e-9);

    const biased = makeCore({ expression: 'x1 + x2 + 0.5', count: 40, seed: 7, bias: true });
    biased.reset({ startB: 0.5 });
    const biasedLoss = biased.landscapeLoss(1, 1);
    expect(biasedLoss).toBeGreaterThanOrEqual(0);
    expect(biasedLoss).toBeLessThan(1e-9);
  });
});

describe('weight domain', () => {
  it('overrides the fitted domain and restores it with null', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    const fitted = core.domain;

    core.setWeightDomain(7);
    expect(core.domain).toBe(7);

    core.setWeightDomain(null);
    expect(core.domain).toBe(fitted);
  });

  it('clamps the manual domain to [1, 50]', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });

    core.setWeightDomain(0.5);
    expect(core.domain).toBe(1);

    core.setWeightDomain(100);
    expect(core.domain).toBe(50);

    core.setWeightDomain(25);
    expect(core.domain).toBe(25);
  });

  it('ignores a non-finite manual domain', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.setWeightDomain(7);

    core.setWeightDomain(Number.NaN);
    expect(core.domain).toBe(7);

    core.setWeightDomain(Number.POSITIVE_INFINITY);
    expect(core.domain).toBe(7);
  });

  it('survives setDataset', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.setWeightDomain(7);

    core.setDataset({ count: 10 });
    expect(core.domain).toBe(7);

    core.setWeightDomain(null);
    expect(core.domain).toBe(5);
  });

  it('clamps setWeights to ten times the manual domain', () => {
    const core = makeCore({ expression: 'x1 + x2', count: 40, seed: 5 });
    core.setWeightDomain(5);

    core.setWeights({ w1: 1e6 });
    expect(core.w1).toBe(50);

    core.setWeights({ w2: -1e6 });
    expect(core.w2).toBe(-50);
  });

  it('uses the manual domain as the divergence limit', () => {
    const core = makeCore({ expression: '25*x1', count: 40, seed: 5 });
    expect(core.domain).toBe(20);

    core.setWeights({ w1: 60, w2: 0 });
    expect(core.w1).toBe(60);
    core.advanceBatch(0.1);
    expect(core.diverged).toBe(false);

    core.setWeights({ w1: 60, w2: 0 });
    core.setWeightDomain(5);
    expect(core.domain).toBe(5);
    core.advanceBatch(0.1);
    expect(core.diverged).toBe(true);
  });
});
