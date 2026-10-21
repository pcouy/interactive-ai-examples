import { describe, expect, it } from 'vitest';
import type { PerceptronCore } from '../../core/perceptron.js';
import { Perceptron } from './drl-perceptron.js';

interface PendingPredictionView {
  index: number;
  point: { label: 0 | 1 };
  prediction: 0 | 1;
}

/**
 * The node test bench has no DOM: neutralise the Lit render pass and inspect
 * the core and the manual-step fields directly.
 */
function createPerceptron(): Perceptron {
  const element = new Perceptron();
  (element as unknown as { performUpdate(): void }).performUpdate = () => {};
  return element;
}

function readCore(element: Perceptron): PerceptronCore {
  return (element as unknown as { core: PerceptronCore }).core;
}

function readPending(element: Perceptron): PendingPredictionView | null {
  return (element as unknown as { pendingPrediction: PendingPredictionView | null })
    .pendingPrediction;
}

function readApplied(element: Perceptron): {
  before: { w1: number; w2: number; bias: number };
  after: { w1: number; w2: number; bias: number };
} | null {
  return (
    element as unknown as {
      appliedUpdate: {
        before: { w1: number; w2: number; bias: number };
        after: { w1: number; w2: number; bias: number };
      } | null;
    }
  ).appliedUpdate;
}

/** Installs one point whose label is the opposite of the current prediction. */
function installMisclassifiedPoint(element: Perceptron): { prediction: 0 | 1; label: 0 | 1 } {
  const core = readCore(element);
  core.setData([]);
  const prediction = core.predictionAt(1, 1);
  const label = prediction === 1 ? 0 : 1;
  core.addPoint({ x: 1, y: 1, label });
  return { prediction, label };
}

describe('Perceptron manual step', () => {
  it('shows the wrong prediction before touching the weights', () => {
    const element = createPerceptron();
    const { prediction, label } = installMisclassifiedPoint(element);
    const core = readCore(element);
    const before = { ...core.weights, bias: core.bias };

    element.step();

    expect(readPending(element)).not.toBeNull();
    expect(readPending(element)?.prediction).toBe(prediction);
    expect(readPending(element)?.point.label).toBe(label);
    expect(core.weights.w1).toBe(before.w1);
    expect(core.weights.w2).toBe(before.w2);
    expect(core.bias).toBe(before.bias);
    expect(core.ticksElapsed).toBe(0);
    expect(element.state.decision).toBe(prediction);
    expect(element.state.expected).toBe(label);
  });

  it('applies the update on the next step and keeps the rule values', () => {
    const element = createPerceptron();
    element.learningRate = 1;
    const { prediction, label } = installMisclassifiedPoint(element);
    const core = readCore(element);
    const before = { ...core.weights, bias: core.bias };

    element.step();
    element.step();

    expect(readPending(element)).toBeNull();
    expect(readApplied(element)?.before).toEqual(before);
    expect(readApplied(element)?.after).toEqual({ ...core.weights, bias: core.bias });
    expect(core.weights.w1).toBeCloseTo(before.w1 + (label - prediction) * 1);
    expect(core.weights.w2).toBeCloseTo(before.w2 + (label - prediction) * 1);
    expect(core.bias).toBeCloseTo(before.bias + (label - prediction));
    expect(core.ticksElapsed).toBe(1);
    expect(element.state.decision).toBe(label);
  });

  it('advances immediately when the prediction is correct', () => {
    const element = createPerceptron();
    const core = readCore(element);
    core.setData([]);
    const prediction = core.predictionAt(1, 1);
    core.addPoint({ x: 1, y: 1, label: prediction });

    element.step();

    expect(readPending(element)).toBeNull();
    expect(readApplied(element)).toBeNull();
    expect(core.ticksElapsed).toBe(1);
    expect(core.lastPresentedIndex).toBe(0);
  });
});
