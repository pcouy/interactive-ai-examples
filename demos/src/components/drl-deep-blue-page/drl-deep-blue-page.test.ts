import { describe, expect, it } from 'vitest';
import type { VisitEvent } from '../../core/contracts.js';
import { TreeModel } from '../../core/search/tree-model.js';
import { firstMoveOnPath } from './drl-deep-blue-page.js';

interface PageState {
  readonly id: number;
}

const STATES: readonly PageState[] = [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }];

function visit(
  nodeKey: string,
  parentKey: string | null,
  move: unknown,
  depth: number,
): VisitEvent<PageState> {
  return { nodeKey, parentKey, move, depth, state: STATES[depth] ?? STATES[0], kind: 'expand' };
}

function buildModel(): TreeModel<PageState> {
  const model = new TreeModel<PageState>();
  model.insert(visit('root', null, null, 0));
  model.insert(visit('left', 'root', { index: 4 }, 1));
  model.insert(visit('left-left', 'left', { index: 0 }, 2));
  model.insert(visit('right', 'root', { col: 2 }, 1));
  return model;
}

describe('firstMoveOnPath', () => {
  it('returns null for the search root node itself', () => {
    const model = buildModel();
    expect(firstMoveOnPath(model, 'root')).toBeNull();
  });

  it('returns null for a null node key', () => {
    const model = buildModel();
    expect(firstMoveOnPath(model, null)).toBeNull();
  });

  it('returns the move of the direct child of the root for a first-level node', () => {
    const model = buildModel();
    expect(firstMoveOnPath(model, 'left')).toEqual({ index: 4 });
    expect(firstMoveOnPath(model, 'right')).toEqual({ col: 2 });
  });

  it('returns the first move of the path for a deep node', () => {
    const model = buildModel();
    expect(firstMoveOnPath(model, 'left-left')).toEqual({ index: 4 });
  });

  it('returns null for an unknown node key', () => {
    const model = buildModel();
    expect(firstMoveOnPath(model, 'missing')).toBeNull();
  });

  it('returns null for a node detached from the search root', () => {
    const model = buildModel();
    model.applyValue({ nodeKey: 'orphan', score: 0.5, depth: 1, exact: false });
    expect(firstMoveOnPath(model, 'orphan')).toBeNull();
  });

  it('returns null when the first move of the path is missing', () => {
    const model = new TreeModel<PageState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('child', 'root', null, 1));
    model.insert(visit('grandchild', 'child', { index: 7 }, 2));
    expect(firstMoveOnPath(model, 'grandchild')).toBeNull();
  });
});
