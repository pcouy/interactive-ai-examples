import { describe, expect, it } from 'vitest';
import type { CutInfo, VisitKind } from '../../core/contracts.js';
import { TreeModel } from '../../core/search/tree-model.js';
import {
  boundsSummary,
  cutDescription,
  cutLabel,
  formatBound,
  formatEntryDepth,
  formatMove,
  formatNumber,
  formatScore,
  keepFirstPruned,
  NO_CUT_LABEL,
} from './display.js';

interface FakeState {
  readonly id: string;
}

function cut(overrides: Partial<CutInfo> = {}): CutInfo {
  return {
    bound: 'beta',
    strict: true,
    bestScore: 0.5,
    limit: 0.2,
    triggerKey: null,
    ...overrides,
  };
}

function buildModel(): TreeModel<FakeState> {
  const model = new TreeModel<FakeState>();
  model.insert({
    nodeKey: 'root',
    parentKey: null,
    move: null,
    state: { id: 'root' },
    depth: 0,
    kind: 'expand',
    window: { alpha: Number.NEGATIVE_INFINITY, beta: Number.POSITIVE_INFINITY },
  });
  model.insert({
    nodeKey: 'a',
    parentKey: 'root',
    move: { index: 4 },
    state: { id: 'a' },
    depth: 1,
    kind: 'expand',
    window: { alpha: 0.1, beta: 0.4 },
  });
  model.insert({
    nodeKey: 'b',
    parentKey: 'root',
    move: { index: 7 },
    state: { id: 'b' },
    depth: 1,
    kind: 'expand',
    window: { alpha: 0.1, beta: 0.4 },
  });
  return model;
}

describe('formatNumber', () => {
  it('uses a French decimal comma', () => {
    expect(formatNumber(0.5)).toBe('0,50');
    expect(formatNumber(-0.25)).toBe('-0,25');
  });
});

describe('formatScore', () => {
  it('prefixes positive scores with +', () => {
    expect(formatScore(0.5)).toBe('+0,50');
    expect(formatScore(-0.25)).toBe('-0,25');
    expect(formatScore(0)).toBe('0,00');
  });

  it('collapses negative zero to zero', () => {
    expect(formatScore(-0.001)).toBe('0,00');
  });

  it('renders null as an empty string', () => {
    expect(formatScore(null)).toBe('');
  });
});

describe('formatMove', () => {
  it('renders primitives', () => {
    expect(formatMove(4)).toBe('4');
    expect(formatMove('e4')).toBe('e4');
  });

  it('probes known object fields', () => {
    expect(formatMove({ index: 4 })).toBe('4');
    expect(formatMove({ col: 2 })).toBe('2');
  });

  it('falls back to truncated JSON', () => {
    expect(formatMove({ foo: 1 })).toBe('{"foo":1}');
    expect(formatMove({ a: 'abcdefghijklmnop' })).toBe('{"a":"abcde…');
  });

  it('renders null and undefined as empty', () => {
    expect(formatMove(null)).toBe('');
    expect(formatMove(undefined)).toBe('');
  });
});

describe('formatBound', () => {
  it('renders infinities with their own glyphs', () => {
    expect(formatBound(Number.NEGATIVE_INFINITY)).toBe('−∞');
    expect(formatBound(Number.POSITIVE_INFINITY)).toBe('+∞');
  });

  it('renders finite bounds as signed scores', () => {
    expect(formatBound(0)).toBe('0,00');
    expect(formatBound(0.5)).toBe('+0,50');
    expect(formatBound(-0.25)).toBe('-0,25');
    expect(formatBound(-0.001)).toBe('0,00');
  });
});

describe('NO_CUT_LABEL', () => {
  it('uses the exact French label of the bounds panel for an uncut node', () => {
    expect(NO_CUT_LABEL).toBe('Développement du nœud');
  });
});

describe('formatEntryDepth', () => {
  it('renders a finite certified depth as digits', () => {
    expect(formatEntryDepth(3)).toBe('3');
    expect(formatEntryDepth(0)).toBe('0');
  });

  it('renders the solved sentinel with the infinity glyph', () => {
    expect(formatEntryDepth(Number.POSITIVE_INFINITY)).toBe('∞');
  });
});

describe('keepFirstPruned', () => {
  const child = (kind: VisitKind): { kind: VisitKind } => ({ kind });

  it('keeps only the first pruned sibling', () => {
    const children = [child('pruned'), child('pruned'), child('pruned')];
    expect(keepFirstPruned(children)).toEqual([child('pruned')]);
  });

  it('keeps every non-pruned sibling in order', () => {
    const children = [
      child('expand'),
      child('pruned'),
      child('leaf'),
      child('pruned'),
      child('expand'),
    ];
    expect(keepFirstPruned(children).map((c) => c.kind)).toEqual([
      'expand',
      'pruned',
      'leaf',
      'expand',
    ]);
  });

  it('leaves zero or one pruned sibling unchanged', () => {
    const none = [child('expand'), child('leaf')];
    expect(keepFirstPruned(none)).toEqual(none);
    const one = [child('expand'), child('pruned'), child('leaf')];
    expect(keepFirstPruned(one)).toEqual(one);
  });

  it('keeps the first pruned sibling in its original position', () => {
    const children = [child('expand'), child('pruned'), child('pruned'), child('leaf')];
    expect(keepFirstPruned(children).map((c) => c.kind)).toEqual(['expand', 'pruned', 'leaf']);
  });
});

describe('cutLabel', () => {
  it('uses the strict symbols on the right bound', () => {
    expect(cutLabel(cut({ bound: 'beta', strict: true }))).toBe('> β');
    expect(cutLabel(cut({ bound: 'beta', strict: false }))).toBe('≥ β');
    expect(cutLabel(cut({ bound: 'alpha', strict: true }))).toBe('< α');
    expect(cutLabel(cut({ bound: 'alpha', strict: false }))).toBe('≤ α');
  });
});

describe('cutDescription', () => {
  it('names the triggering move', () => {
    expect(cutDescription(cut({ triggerKey: 'a' }), 4)).toBe(
      'meilleur +0,50 > β +0,20 · déclenché par le coup 4',
    );
  });

  it('handles a non-strict cut without a trigger', () => {
    expect(cutDescription(cut({ strict: false }), null)).toBe(
      'meilleur +0,50 ≥ β +0,20 · sans coup déclencheur',
    );
  });

  it('falls back to an unknown move label', () => {
    // Une chaîne vide n'est pas un coup absent : c'est un coup non représentable.
    expect(cutDescription(cut(), '')).toBe(
      'meilleur +0,50 > β +0,20 · déclenché par le coup coup inconnu',
    );
  });
});

describe('boundsSummary', () => {
  it('returns null for a null or unknown key', () => {
    const model = buildModel();
    expect(boundsSummary(model, null)).toBeNull();
    expect(boundsSummary(model, 'missing')).toBeNull();
  });

  it('summarises an uncut node', () => {
    const model = buildModel();
    expect(boundsSummary(model, 'a')).toEqual({
      nodeKey: 'a',
      move: { index: 4 },
      alpha: 0.1,
      beta: 0.4,
      score: null,
      exact: false,
      cut: null,
      triggerMove: null,
    });
  });

  it('resolves the triggering move through the model', () => {
    const model = buildModel();
    const info = cut({ triggerKey: 'b' });
    model.applyValue({ nodeKey: 'a', score: 0.5, depth: 1, exact: false, cut: info });
    const summary = boundsSummary(model, 'a');
    expect(summary?.cut).toEqual(info);
    expect(summary?.triggerMove).toEqual({ index: 7 });
  });
});
