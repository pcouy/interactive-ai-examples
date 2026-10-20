import { describe, expect, it } from 'vitest';
import {
  DEFAULT_FILTER_COUNTS,
  forwardMnistCnn,
  initializeMnistCnnWeights,
  type MnistCnnConfig,
} from '../../core/mnist-cnn.js';
import {
  describeWeightSelection,
  parseWeightInput,
  selectedWeightValue,
  type WeightSelection,
} from './drl-mnist-cnn.js';
import {
  applyWeightDraft,
  formatWeightValue,
  weightEditorHint,
  weightEditorView,
} from './weight-editor.js';

/** Architecture de référence, quatre convolutions de trois filtres. */
const CONFIG: MnistCnnConfig = { filters: DEFAULT_FILTER_COUNTS };

/** Autre architecture, pour éprouver l'incohérence poids / filtres. */
const OTHER_CONFIG: MnistCnnConfig = { filters: [2, 2, 2, 2] };

/** Trois couches de convolution seulement. */
const THREE_LAYER_CONFIG: MnistCnnConfig = { filters: DEFAULT_FILTER_COUNTS, convLayers: 3 };

/** Poids déterministes de l'architecture par défaut, pour les tests. */
const WEIGHTS = initializeMnistCnnWeights(CONFIG, 42);

/** Poids de `conv2` (entrée 1, sortie 3), au centre du noyau. */
const WEIGHT_SELECTION: WeightSelection = {
  kind: 'weight',
  boundary: 1,
  inputChannel: 0,
  outputChannel: 2,
  ky: 1,
  kx: 1,
};

const LINK_SELECTION: WeightSelection = {
  kind: 'link',
  boundary: 1,
  inputChannel: 0,
  outputChannel: 2,
};

/** Lit la valeur du poids sélectionné en exigeant qu'elle existe. */
function requiredValue(weights: typeof WEIGHTS, selection: WeightSelection): number {
  const value = selectedWeightValue(selection, weights, CONFIG);
  if (value === null) {
    throw new Error('expected the selection to name an existing weight');
  }
  return value;
}

describe('weightEditorView', () => {
  it('returns null without a selection or for a link selection', () => {
    expect(weightEditorView(null, WEIGHTS, CONFIG)).toBeNull();
    expect(weightEditorView(LINK_SELECTION, WEIGHTS, CONFIG)).toBeNull();
  });

  it('fills the view from the shared helpers for a weight selection', () => {
    const view = weightEditorView(WEIGHT_SELECTION, WEIGHTS, CONFIG);
    const value = requiredValue(WEIGHTS, WEIGHT_SELECTION);
    expect(view).toEqual({
      label: describeWeightSelection(WEIGHT_SELECTION, CONFIG),
      value,
      text: formatWeightValue(value),
    });
  });

  it('returns null for a weight selection outside the architecture', () => {
    const outside: WeightSelection = {
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 5,
      ky: 0,
      kx: 0,
    };
    expect(weightEditorView(outside, WEIGHTS, CONFIG)).toBeNull();
  });

  it('returns null without throwing for a non-integer or missing index', () => {
    const nonIntegerKy: WeightSelection = {
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1.5,
      kx: 1,
    };
    const nanKx: WeightSelection = {
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: Number.NaN,
    };
    const missingKx = {
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
    } as unknown as WeightSelection;
    expect(weightEditorView(nonIntegerKy, WEIGHTS, CONFIG)).toBeNull();
    expect(weightEditorView(nanKx, WEIGHTS, CONFIG)).toBeNull();
    expect(weightEditorView(missingKx, WEIGHTS, CONFIG)).toBeNull();
  });

  it('reads a weight on the last active layer of a three-layer architecture', () => {
    const weights = initializeMnistCnnWeights(THREE_LAYER_CONFIG, 42);
    const selection: WeightSelection = {
      kind: 'weight',
      boundary: 2,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 1,
    };
    const value = selectedWeightValue(selection, weights, THREE_LAYER_CONFIG);
    if (value === null) {
      throw new Error('expected the selection to name an existing weight');
    }
    expect(weightEditorView(selection, weights, THREE_LAYER_CONFIG)).toEqual({
      label: describeWeightSelection(selection, THREE_LAYER_CONFIG),
      value,
      text: formatWeightValue(value),
    });
  });

  it('returns null for a weight past the active layer count', () => {
    const weights = initializeMnistCnnWeights(THREE_LAYER_CONFIG, 42);
    const selection: WeightSelection = {
      kind: 'weight',
      boundary: 3,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 1,
    };
    expect(weightEditorView(selection, weights, THREE_LAYER_CONFIG)).toBeNull();
  });
});

describe('formatWeightValue', () => {
  it('round-trips through parseWeightInput', () => {
    for (const value of [0.5, -1.25, 3, 0.30000000000000004]) {
      expect(parseWeightInput(formatWeightValue(value))).toBe(value);
    }
  });

  it('uses the shortest plain decimal form', () => {
    expect(formatWeightValue(0.5)).toBe('0.5');
    expect(formatWeightValue(-1.25)).toBe('-1.25');
    expect(formatWeightValue(3)).toBe('3');
  });

  it('expands exponential notation into plain decimals', () => {
    expect(formatWeightValue(1e-7)).toBe('0.0000001');
    expect(parseWeightInput(formatWeightValue(1e-7))).toBe(1e-7);
    expect(parseWeightInput(formatWeightValue(1e21))).toBe(1e21);
  });

  it('rejects a non-finite value', () => {
    expect(() => formatWeightValue(Number.NaN)).toThrow(RangeError);
    expect(() => formatWeightValue(Number.POSITIVE_INFINITY)).toThrow(RangeError);
  });
});

describe('applyWeightDraft', () => {
  it('applies a valid draft and leaves the input weights unchanged', () => {
    const before = requiredValue(WEIGHTS, WEIGHT_SELECTION);
    const result = applyWeightDraft('0.75', WEIGHT_SELECTION, WEIGHTS, CONFIG);
    expect(result.kind).toBe('applied');
    if (result.kind !== 'applied') {
      return;
    }
    expect(result.value).toBe(0.75);
    expect(result.weights).not.toBe(WEIGHTS);
    expect(requiredValue(result.weights, WEIGHT_SELECTION)).toBe(0.75);
    expect(requiredValue(WEIGHTS, WEIGHT_SELECTION)).toBe(before);
  });

  it('accepts a comma decimal separator', () => {
    const result = applyWeightDraft('0,75', WEIGHT_SELECTION, WEIGHTS, CONFIG);
    expect(result.kind).toBe('applied');
    if (result.kind !== 'applied') {
      return;
    }
    expect(result.value).toBe(0.75);
    expect(requiredValue(result.weights, WEIGHT_SELECTION)).toBe(0.75);
  });

  it('applies a draft on the last active layer of a three-layer architecture', () => {
    const weights = initializeMnistCnnWeights(THREE_LAYER_CONFIG, 42);
    const selection: WeightSelection = {
      kind: 'weight',
      boundary: 2,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 1,
    };
    const result = applyWeightDraft('1.5', selection, weights, THREE_LAYER_CONFIG);
    expect(result.kind).toBe('applied');
    if (result.kind !== 'applied') {
      return;
    }
    expect(result.value).toBe(1.5);
    expect(selectedWeightValue(selection, result.weights, THREE_LAYER_CONFIG)).toBe(1.5);
  });

  it('returns an error for an empty draft', () => {
    const result = applyWeightDraft('', WEIGHT_SELECTION, WEIGHTS, CONFIG);
    expect(result).toEqual({ kind: 'error', message: expect.stringContaining('Saisissez') });
  });

  it('returns an error for an unreadable draft', () => {
    for (const draft of ['abc', '1.2.3', 'NaN']) {
      const result = applyWeightDraft(draft, WEIGHT_SELECTION, WEIGHTS, CONFIG);
      expect(result).toEqual({ kind: 'error', message: expect.stringContaining('illisible') });
    }
  });

  it('returns an error for a non-finite value', () => {
    const result = applyWeightDraft('9'.repeat(400), WEIGHT_SELECTION, WEIGHTS, CONFIG);
    expect(result).toEqual({ kind: 'error', message: expect.stringContaining('non finie') });
  });

  it('returns an error for a selection that is not a weight', () => {
    for (const selection of [null, LINK_SELECTION] as const) {
      const result = applyWeightDraft('0.5', selection, WEIGHTS, CONFIG);
      expect(result).toEqual({
        kind: 'error',
        message: expect.stringContaining("n'est pas un poids"),
      });
    }
  });

  it('returns an error for a weight outside the architecture', () => {
    const outside: WeightSelection = {
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 5,
      ky: 0,
      kx: 0,
    };
    const result = applyWeightDraft('0.5', outside, WEIGHTS, CONFIG);
    expect(result).toEqual({
      kind: 'error',
      message: expect.stringContaining("n'existe pas"),
    });
  });

  it('reports an out-of-architecture selection before an unreadable draft', () => {
    const outside: WeightSelection = {
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 5,
      ky: 0,
      kx: 0,
    };
    const result = applyWeightDraft('abc', outside, WEIGHTS, CONFIG);
    expect(result).toEqual({
      kind: 'error',
      message: expect.stringContaining("n'existe pas"),
    });
  });

  it('trims surrounding whitespace before parsing', () => {
    const blank = applyWeightDraft('   ', WEIGHT_SELECTION, WEIGHTS, CONFIG);
    expect(blank).toEqual({ kind: 'error', message: expect.stringContaining('Saisissez') });
    const padded = applyWeightDraft(' 0.5 ', WEIGHT_SELECTION, WEIGHTS, CONFIG);
    expect(padded.kind).toBe('applied');
    if (padded.kind !== 'applied') {
      return;
    }
    expect(padded.value).toBe(0.5);
    expect(requiredValue(padded.weights, WEIGHT_SELECTION)).toBe(0.5);
  });

  it('returns an error when the weights do not match the filters', () => {
    const mismatched = initializeMnistCnnWeights(OTHER_CONFIG, 42);
    const result = applyWeightDraft('0.5', WEIGHT_SELECTION, mismatched, CONFIG);
    expect(result).toEqual({
      kind: 'error',
      message: expect.stringContaining('correspondent'),
    });
  });

  it('produces weights that still run forwardMnistCnn and change the probabilities', () => {
    const input = new Float32Array(784).fill(0.5);
    const before = forwardMnistCnn(input, WEIGHTS, CONFIG);
    const result = applyWeightDraft('1.5', WEIGHT_SELECTION, WEIGHTS, CONFIG);
    expect(result.kind).toBe('applied');
    if (result.kind !== 'applied') {
      return;
    }
    const after = forwardMnistCnn(input, result.weights, CONFIG);
    expect(Array.from(after.probabilities)).not.toEqual(Array.from(before.probabilities));
  });
});

describe('weightEditorHint', () => {
  it('describes the gesture in one French sentence', () => {
    const hint = weightEditorHint();
    expect(hint).toContain('liaison');
    expect(hint).toContain('toile de filtre');
    expect(hint).toContain('valeur');
  });
});
