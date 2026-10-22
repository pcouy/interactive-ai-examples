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

/** Flattens a Lit `TemplateResult` into plain text for assertions. */
function renderTemplateTextFlat(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }
  if (Array.isArray(value)) {
    return value.map(renderTemplateTextFlat).join('');
  }
  if (typeof value === 'object' && 'strings' in value && 'values' in value) {
    const template = value as { strings: readonly string[]; values: readonly unknown[] };
    let out = '';
    for (let index = 0; index < template.strings.length; index += 1) {
      out += template.strings[index];
      if (index < template.values.length) {
        out += renderTemplateTextFlat(template.values[index]);
      }
    }
    return out.replace(/\s+/g, ' ').trim();
  }
  return String(value).replace(/\s+/g, ' ').trim();
}

/** A D3-like selection whose chainable methods all return itself. */
function chainSelection(): Record<string, unknown> {
  const selection: Record<string, unknown> = {};
  const chain = (): Record<string, unknown> => selection;
  selection.attr = chain;
  selection.text = chain;
  selection.selectAll = chain;
  selection.data = chain;
  selection.join = chain;
  return selection;
}

/** A text node that records the text written by the component. */
function textNode(): { textContent: string; text: (value: string) => void } {
  const node = {
    textContent: '',
    text: (value: string): void => {
      node.textContent = value;
    },
  };
  return node;
}

interface DrawNodes {
  inputValueOne: ReturnType<typeof textNode>;
  inputValueTwo: ReturnType<typeof textNode>;
  weightOne: ReturnType<typeof textNode>;
  weightTwo: ReturnType<typeof textNode>;
  biasLabel: ReturnType<typeof textNode>;
  statusText: { textContent: string };
  statusBadge: { textContent: string; style: { display: string } };
}

/** Installs the D3 selections and text nodes that draw() writes to. */
function installDrawNodes(element: Perceptron): DrawNodes {
  const nodes: DrawNodes = {
    inputValueOne: textNode(),
    inputValueTwo: textNode(),
    weightOne: textNode(),
    weightTwo: textNode(),
    biasLabel: textNode(),
    statusText: { textContent: '' },
    statusBadge: { textContent: '', style: { display: '' } },
  };
  const internals = element as unknown as Record<string, unknown>;
  internals.boundary = chainSelection();
  internals.pointsGroup = chainSelection();
  internals.halo = chainSelection();
  internals.inputValueOne = nodes.inputValueOne;
  internals.inputValueTwo = nodes.inputValueTwo;
  internals.weightOne = nodes.weightOne;
  internals.weightTwo = nodes.weightTwo;
  internals.biasLabel = nodes.biasLabel;
  internals.statusText = nodes.statusText;
  internals.statusBadge = nodes.statusBadge;
  return nodes;
}

describe('Perceptron locale', () => {
  it('renders the French controls and SVG labels by default', () => {
    const element = createPerceptron();
    const text = renderTemplateTextFlat(
      (element as unknown as { renderContent(): unknown }).renderContent(),
    );

    expect(text).toContain('Séparables');
    expect(text).toContain('Non séparables');
    expect(text).toContain('Classe bleue');
    expect(text).toContain('Classe rouge');
    expect(text).toContain('Effacer');
    expect(text).toContain('Règle de mise à jour');
    expect(text).toContain('Contrôles du perceptron');
    expect(text).toContain('Plan de points et frontière de décision du perceptron');
    expect(text).toContain('seuil');
  });

  it('renders the English controls and SVG labels', () => {
    const element = createPerceptron();
    element.locale = 'en';
    const text = renderTemplateTextFlat(
      (element as unknown as { renderContent(): unknown }).renderContent(),
    );

    expect(text).toContain('Separable');
    expect(text).toContain('Non-separable');
    expect(text).toContain('Blue class');
    expect(text).toContain('Red class');
    expect(text).toContain('Clear');
    expect(text).toContain('Update rule');
    expect(text).toContain('Perceptron controls');
    expect(text).toContain('Point plane and perceptron decision boundary');
    expect(text).toContain('threshold');
  });

  it('formats the SVG weights with a comma in French and a point in English', () => {
    const element = createPerceptron();
    const nodes = installDrawNodes(element);

    (element as unknown as { draw(): void }).draw();
    expect(nodes.weightOne.textContent).toMatch(/^w₁ = -?\d+,\d{2}$/);
    expect(nodes.weightTwo.textContent).toMatch(/^w₂ = -?\d+,\d{2}$/);
    expect(nodes.biasLabel.textContent).toMatch(/^b = -?\d+,\d{2}$/);
    expect(nodes.statusText.textContent).toBe('Époque 0 · Précision 100 % · 0 présentations');
    expect(nodes.inputValueOne.textContent).toBe('–');

    element.locale = 'en';
    (element as unknown as { onLocaleChange(): void }).onLocaleChange();
    expect(nodes.weightOne.textContent).toMatch(/^w₁ = -?\d+\.\d{2}$/);
    expect(nodes.weightTwo.textContent).toMatch(/^w₂ = -?\d+\.\d{2}$/);
    expect(nodes.biasLabel.textContent).toMatch(/^b = -?\d+\.\d{2}$/);
    expect(nodes.statusText.textContent).toBe('Epoch 0 · Accuracy 100 % · 0 presentations');
  });

  it('renders the English explanation and update rule', () => {
    const element = createPerceptron();
    element.locale = 'en';
    const headline = { textContent: '' };
    const detail = { textContent: '' };
    const hint = { textContent: '' };
    const rule = { hidden: false };
    const ruleW1 = { textContent: '' };
    const ruleW2 = { textContent: '' };
    const ruleBias = { textContent: '' };
    const internals = element as unknown as Record<string, unknown>;
    internals.explanation = { hidden: false, dataset: {} };
    internals.explanationHeadline = headline;
    internals.explanationDetail = detail;
    internals.explanationHint = hint;
    internals.explanationRule = rule;
    internals.ruleW1 = ruleW1;
    internals.ruleW2 = ruleW2;
    internals.ruleBias = ruleBias;

    internals.pendingPrediction = {
      index: 0,
      point: { x: 1.5, y: -0.25, label: 0 },
      prediction: 1,
    };
    (element as unknown as { renderExplanation(): void }).renderExplanation();

    expect(headline.textContent).toBe('Wrong prediction');
    expect(detail.textContent).toBe('ŷ = 1 (blue class) · expected t = 0 (red class).');
    expect(hint.textContent).toBe('Step: show the update rule.');

    internals.pendingPrediction = null;
    internals.appliedUpdate = {
      point: { x: 1.5, y: -0.25, label: 0 },
      prediction: 1,
      expected: 0,
      learningRate: 0.1,
      before: { w1: 1.5, w2: -0.25, bias: 0.75 },
      after: { w1: 1.4, w2: -0.2, bias: 0.65 },
    };
    (element as unknown as { renderExplanation(): void }).renderExplanation();

    expect(headline.textContent).toBe('Update rule applied');
    expect(hint.textContent).toBe('Step: present the next point.');
    expect(ruleW1.textContent).toBe('w₁ ← w₁ + η·(t − ŷ)·x₁ = 1.50 + 0.10·(0 − 1)·1.50 = 1.40');
    expect(ruleW2.textContent).toBe('w₂ ← w₂ + η·(t − ŷ)·x₂ = -0.25 + 0.10·(0 − 1)·-0.25 = -0.20');
    expect(ruleBias.textContent).toBe('b ← b + η·(t − ŷ) = 0.75 + 0.10·(0 − 1) = 0.65');
  });
});
