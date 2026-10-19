import { describe, expect, it } from 'vitest';
import type { CutInfo, SearchWindow, VisitEvent, VisitKind } from '../contracts.js';
import { createModelRecorder, TreeModel } from './tree-model.js';

interface FakeState {
  readonly id: string;
}

function visit(
  nodeKey: string,
  parentKey: string | null,
  move: string | null,
  depth: number,
  kind: VisitKind = 'expand',
  stateId = nodeKey,
  bounds: { window?: SearchWindow; cut?: CutInfo } = {},
): VisitEvent<FakeState> {
  return {
    nodeKey,
    parentKey,
    move,
    state: { id: stateId },
    depth,
    kind,
    ...bounds,
  };
}

function cutOf(bound: 'alpha' | 'beta', triggerKey: string | null = null): CutInfo {
  return { bound, strict: true, bestScore: 0.5, limit: 0.2, triggerKey };
}

describe('TreeModel insertion', () => {
  it('builds a scripted DFS tree with correct parents, depths and orders', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('a', 'root', 'mA', 1));
    model.insert(visit('a1', 'a', 'mA1', 2));
    model.insert(visit('a2', 'a', 'mA2', 2));
    model.insert(visit('b', 'root', 'mB', 1));
    model.insert(visit('b1', 'b', 'mB1', 2));

    expect(model.size()).toBe(6);
    expect(model.root()?.nodeKey).toBe('root');
    expect(model.children('root').map((n) => n.nodeKey)).toEqual(['a', 'b']);
    expect(model.children('a').map((n) => n.nodeKey)).toEqual(['a1', 'a2']);
    expect(model.children('b').map((n) => n.nodeKey)).toEqual(['b1']);

    const root = model.node('root');
    expect(root?.parentKey).toBeNull();
    expect(root?.depth).toBe(0);
    expect(root?.order).toBe(0);
    expect(model.node('a')?.parentKey).toBe('root');
    expect(model.node('a')?.depth).toBe(1);
    expect(model.node('a')?.order).toBe(1);
    expect(model.node('a1')?.order).toBe(2);
    expect(model.node('a2')?.order).toBe(3);
    expect(model.node('b')?.order).toBe(4);
    expect(model.node('b1')?.order).toBe(5);
    for (const [parent, child] of [
      ['root', 'a'],
      ['root', 'b'],
      ['a', 'a1'],
      ['a', 'a2'],
      ['b', 'b1'],
    ] as const) {
      const childNode = model.node(child);
      const parentNode = model.node(parent);
      expect(childNode?.parentKey).toBe(parent);
      expect(childNode && parentNode ? childNode.order > parentNode.order : false).toBe(true);
    }
  });

  it('keeps states and moves on nodes', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('a', 'root', 'mA', 1, 'leaf', 'state-a'));
    expect(model.node('a')?.state).toEqual({ id: 'state-a' });
    expect(model.node('a')?.move).toBe('mA');
  });
});

describe('TreeModel merge by nodeKey', () => {
  it('keeps a single entry at the first insertion position on re-visit', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('x', 'root', 'mX', 1));
    model.insert(visit('y', 'root', 'mY', 1));
    model.insert(visit('z', 'x', 'mZ', 2));
    expect(model.size()).toBe(4);

    model.insert(visit('y', 'z', 'mY-again', 3));
    expect(model.size()).toBe(4);
    expect(model.node('y')?.parentKey).toBe('root');
    expect(model.node('y')?.move).toBe('mY');
    expect(model.children('z')).toEqual([]);
    expect(model.children('root').map((n) => n.nodeKey)).toEqual(['x', 'y']);
  });

  it('upgrades kind to the richer visit', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('n', null, null, 0, 'leaf'));
    expect(model.node('n')?.kind).toBe('leaf');
    model.insert(visit('n', null, null, 0, 'expand'));
    expect(model.node('n')?.kind).toBe('expand');
    model.insert(visit('n', null, null, 0, 'leaf'));
    expect(model.node('n')?.kind).toBe('expand');
  });

  it('tracks lastVisited as the most recently inserted or revisited node', () => {
    const model = new TreeModel<FakeState>();
    expect(model.lastVisited()).toBeNull();
    model.insert(visit('a', null, null, 0));
    model.insert(visit('b', 'a', 'm', 1));
    expect(model.lastVisited()?.nodeKey).toBe('b');
    model.insert(visit('a', null, null, 0));
    expect(model.lastVisited()?.nodeKey).toBe('a');
  });
});

describe('TreeModel applyValue', () => {
  it('updates score without duplicating the node', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('n', null, null, 0));
    model.applyValue({ nodeKey: 'n', score: 0.5, depth: 1, exact: false });
    model.applyValue({ nodeKey: 'n', score: 0.9, depth: 1, exact: false });
    expect(model.size()).toBe(1);
    expect(model.node('n')?.score).toBe(0.9);
    expect(model.node('n')?.exact).toBe(false);
    expect(model.resolved('n')).toBe(false);
  });

  it('marks exactness and resolved', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('n', null, null, 0));
    model.applyValue({ nodeKey: 'n', score: -1, depth: 2, exact: true });
    expect(model.node('n')?.exact).toBe(true);
    expect(model.resolved('n')).toBe(true);
  });

  it('drops exactness when overwritten by a bound value', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('n', null, null, 0));
    model.applyValue({ nodeKey: 'n', score: 0.5, depth: 1, exact: true });
    model.applyValue({ nodeKey: 'n', score: 0.9, depth: 1, exact: false });
    expect(model.node('n')?.score).toBe(0.9);
    expect(model.node('n')?.exact).toBe(false);
    expect(model.resolved('n')).toBe(false);
  });

  it('creates the node defensively when absent', () => {
    const model = new TreeModel<FakeState>();
    model.applyValue({ nodeKey: 'ghost', score: 0.3, depth: 4, exact: false });
    expect(model.size()).toBe(1);
    expect(model.has('ghost')).toBe(true);
    expect(model.node('ghost')?.score).toBe(0.3);
  });
});

describe('TreeModel alpha-bêta fields', () => {
  const windowA: SearchWindow = { alpha: 0.1, beta: 0.4 };
  const windowB: SearchWindow = { alpha: 0.2, beta: 0.9 };

  it('stores window and cut on insert', () => {
    const model = new TreeModel<FakeState>();
    const cut = cutOf('beta', 'child');
    model.insert(visit('root', null, null, 0));
    model.insert(visit('n', 'root', 'mN', 1, 'expand', 'n', { window: windowA, cut }));
    expect(model.node('n')?.window).toEqual(windowA);
    expect(model.node('n')?.cut).toEqual(cut);
  });

  it('refreshes window and cut on re-visit', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('n', 'root', 'mN', 1, 'expand', 'n', { window: windowA }));
    const cut = cutOf('alpha');
    model.insert(visit('n', 'root', 'mN', 1, 'expand', 'n', { window: windowB, cut }));
    expect(model.node('n')?.window).toEqual(windowB);
    expect(model.node('n')?.cut).toEqual(cut);
  });

  it('keeps stored bounds when a re-visit carries none', () => {
    const model = new TreeModel<FakeState>();
    const cut = cutOf('beta');
    model.insert(visit('root', null, null, 0));
    model.insert(visit('n', 'root', 'mN', 1, 'expand', 'n', { window: windowA, cut }));
    model.insert(visit('n', 'root', 'mN', 1, 'expand', 'n'));
    expect(model.node('n')?.window).toEqual(windowA);
    expect(model.node('n')?.cut).toEqual(cut);
  });

  it('stores window and cut from an inexact value', () => {
    const model = new TreeModel<FakeState>();
    const cut = cutOf('beta', 'child');
    model.insert(visit('root', null, null, 0));
    model.insert(visit('n', 'root', 'mN', 1));
    model.applyValue({
      nodeKey: 'n',
      score: 0.5,
      depth: 1,
      exact: false,
      window: windowA,
      cut,
    });
    expect(model.node('n')?.window).toEqual(windowA);
    expect(model.node('n')?.cut).toEqual(cut);
  });

  it('clears cut but keeps window on an exact value', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('n', 'root', 'mN', 1));
    model.applyValue({
      nodeKey: 'n',
      score: 0.5,
      depth: 1,
      exact: false,
      window: windowA,
      cut: cutOf('beta'),
    });
    model.applyValue({ nodeKey: 'n', score: 0.7, depth: 1, exact: true, window: windowB });
    expect(model.node('n')?.cut).toBeUndefined();
    expect(model.node('n')?.window).toEqual(windowB);
  });

  it('stores bounds in the defensive branch when the node is absent', () => {
    const model = new TreeModel<FakeState>();
    const cut = cutOf('alpha', 'ghost-child');
    model.applyValue({
      nodeKey: 'ghost',
      score: 0.3,
      depth: 4,
      exact: false,
      window: windowA,
      cut,
    });
    expect(model.node('ghost')?.window).toEqual(windowA);
    expect(model.node('ghost')?.cut).toEqual(cut);
  });
});

describe('TreeModel pruned and tt-hit nodes', () => {
  it('never lets a pruned node acquire descendants', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('p', 'root', 'mP', 1, 'pruned'));
    model.insert(visit('child', 'p', 'mC', 2));
    expect(model.has('child')).toBe(false);
    expect(model.children('p')).toEqual([]);
  });

  it('never lets a tt-hit node acquire descendants', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('h', 'root', 'mH', 1, 'tt-hit'));
    model.insert(visit('child', 'h', 'mC', 2));
    expect(model.has('child')).toBe(false);
  });

  it('ignores re-insertion of a pruned node under another parent', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('p', 'root', 'mP', 1, 'pruned'));
    model.insert(visit('other', 'root', 'mO', 1));
    model.insert(visit('p', 'other', 'mP-again', 2));
    expect(model.size()).toBe(3);
    expect(model.node('p')?.parentKey).toBe('root');
    expect(model.node('p')?.kind).toBe('pruned');
    expect(model.children('other')).toEqual([]);
  });
});

describe('TreeModel edge cases', () => {
  it('drops a child whose parent was never inserted', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('orphan', 'missing', 'm', 1));
    expect(model.size()).toBe(0);
    expect(model.root()).toBeNull();
  });

  it('clears everything', () => {
    const model = new TreeModel<FakeState>();
    model.insert(visit('root', null, null, 0));
    model.insert(visit('a', 'root', 'm', 1));
    model.applyValue({ nodeKey: 'a', score: 1, depth: 1, exact: true });
    model.clear();
    expect(model.size()).toBe(0);
    expect(model.root()).toBeNull();
    expect(model.has('root')).toBe(false);
    expect(model.node('a')).toBeNull();
    expect(model.lastVisited()).toBeNull();
    expect(model.resolved('a')).toBe(false);
    model.insert(visit('root2', null, null, 0));
    expect(model.root()?.nodeKey).toBe('root2');
    expect(model.root()?.order).toBe(0);
  });
});

describe('createModelRecorder', () => {
  it('feeds the model through the TreeRecorder contract', () => {
    const model = new TreeModel<FakeState>();
    const recorder = createModelRecorder(model);
    recorder.onVisit(visit('root', null, null, 0));
    recorder.onVisit(visit('a', 'root', 'mA', 1));
    recorder.onValue({ nodeKey: 'a', score: 0.75, depth: 1, exact: true });
    recorder.onVisit(visit('b', 'root', 'mB', 1, 'pruned'));
    expect(model.size()).toBe(3);
    expect(model.children('root').map((n) => n.nodeKey)).toEqual(['a', 'b']);
    expect(model.node('a')?.score).toBe(0.75);
    expect(model.resolved('a')).toBe(true);
    expect(model.node('b')?.kind).toBe('pruned');
  });
});
