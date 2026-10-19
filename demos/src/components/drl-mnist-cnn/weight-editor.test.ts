import { describe, expect, it } from 'vitest';
import {
  DEFAULT_FILTER_COUNTS,
  forwardMnistCnn,
  initializeMnistCnnWeights,
  type MnistCnnFilters,
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

const FILTERS: MnistCnnFilters = DEFAULT_FILTER_COUNTS;

/** Autre architecture, pour éprouver l'incohérence poids / filtres. */
const OTHER_FILTERS: MnistCnnFilters = [2, 2, 2, 2];

/** Poids déterministes de l'architecture par défaut, pour les tests. */
const WEIGHTS = initializeMnistCnnWeights({ filters: FILTERS }, 42);

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
  const value = selectedWeightValue(weights, selection);
  if (value === null) {
    throw new Error('expected the selection to name an existing weight');
  }
  return value;
}

describe('weightEditorView', () => {
  it('returns null without a selection or for a link selection', () => {
    expect(weightEditorView(null, WEIGHTS, FILTERS)).toBeNull();
    expect(weightEditorView(LINK_SELECTION, WEIGHTS, FILTERS)).toBeNull();
  });

  it('fills the view from the shared helpers for a weight selection', () => {
    const view = weightEditorView(WEIGHT_SELECTION, WEIGHTS, FILTERS);
    const value = requiredValue(WEIGHTS, WEIGHT_SELECTION);
    expect(view).toEqual({
      label: describeWeightSelection(WEIGHT_SELECTION, FILTERS),
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
    expect(weightEditorView(outside, WEIGHTS, FILTERS)).toBeNull();
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
    expect(weightEditorView(nonIntegerKy, WEIGHTS, FILTERS)).toBeNull();
    expect(weightEditorView(nanKx, WEIGHTS, FILTERS)).toBeNull();
    expect(weightEditorView(missingKx, WEIGHTS, FILTERS)).toBeNull();
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
    const result = applyWeightDraft('0.75', WEIGHT_SELECTION, WEIGHTS, FILTERS);
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
    const result = applyWeightDraft('0,75', WEIGHT_SELECTION, WEIGHTS, FILTERS);
    expect(result.kind).toBe('applied');
    if (result.kind !== 'applied') {
      return;
    }
    expect(result.value).toBe(0.75);
    expect(requiredValue(result.weights, WEIGHT_SELECTION)).toBe(0.75);
  });

  it('returns an error for an empty draft', () => {
    const result = applyWeightDraft('', WEIGHT_SELECTION, WEIGHTS, FILTERS);
    expect(result).toEqual({ kind: 'error', message: expect.stringContaining('Saisissez') });
  });

  it('returns an error for an unreadable draft', () => {
    for (const draft of ['abc', '1.2.3', 'NaN']) {
      const result = applyWeightDraft(draft, WEIGHT_SELECTION, WEIGHTS, FILTERS);
      expect(result).toEqual({ kind: 'error', message: expect.stringContaining('illisible') });
    }
  });

  it('returns an error for a non-finite value', () => {
    const result = applyWeightDraft('9'.repeat(400), WEIGHT_SELECTION, WEIGHTS, FILTERS);
    expect(result).toEqual({ kind: 'error', message: expect.stringContaining('non finie') });
  });

  it('returns an error for a selection that is not a weight', () => {
    for (const selection of [null, LINK_SELECTION] as const) {
      const result = applyWeightDraft('0.5', selection, WEIGHTS, FILTERS);
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
    const result = applyWeightDraft('0.5', outside, WEIGHTS, FILTERS);
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
    const result = applyWeightDraft('abc', outside, WEIGHTS, FILTERS);
    expect(result).toEqual({
      kind: 'error',
      message: expect.stringContaining("n'existe pas"),
    });
  });

  it('trims surrounding whitespace before parsing', () => {
    const blank = applyWeightDraft('   ', WEIGHT_SELECTION, WEIGHTS, FILTERS);
    expect(blank).toEqual({ kind: 'error', message: expect.stringContaining('Saisissez') });
    const padded = applyWeightDraft(' 0.5 ', WEIGHT_SELECTION, WEIGHTS, FILTERS);
    expect(padded.kind).toBe('applied');
    if (padded.kind !== 'applied') {
      return;
    }
    expect(padded.value).toBe(0.5);
    expect(requiredValue(padded.weights, WEIGHT_SELECTION)).toBe(0.5);
  });

  it('returns an error when the weights do not match the filters', () => {
    const mismatched = initializeMnistCnnWeights({ filters: OTHER_FILTERS }, 42);
    const result = applyWeightDraft('0.5', WEIGHT_SELECTION, mismatched, FILTERS);
    expect(result).toEqual({
      kind: 'error',
      message: expect.stringContaining('correspondent'),
    });
  });

  it('produces weights that still run forwardMnistCnn and change the probabilities', () => {
    const input = new Float32Array(784).fill(0.5);
    const before = forwardMnistCnn(input, WEIGHTS, { filters: FILTERS });
    const result = applyWeightDraft('1.5', WEIGHT_SELECTION, WEIGHTS, FILTERS);
    expect(result.kind).toBe('applied');
    if (result.kind !== 'applied') {
      return;
    }
    const after = forwardMnistCnn(input, result.weights, { filters: FILTERS });
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
