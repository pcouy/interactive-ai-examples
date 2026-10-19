import { describe, expect, it } from 'vitest';
import type {
  SearchAgent,
  SearchStats,
  SearchStatus,
  TreeRecorder,
  ValueEvent,
  VisitEvent,
  VisitKind,
} from '../contracts.js';
import { createStepper, type Stepper } from './stepper.js';
import { createModelRecorder, TreeModel } from './tree-model.js';

interface ScriptStep {
  readonly visits?: VisitEvent<string>[];
  readonly values?: ValueEvent[];
  readonly done?: boolean;
}

const visit = (
  nodeKey: string,
  parentKey: string | null,
  depth: number,
  kind: VisitKind,
  move: number | null = null,
): VisitEvent<string> => ({
  nodeKey,
  parentKey,
  move,
  state: nodeKey,
  depth,
  kind,
});

const value = (nodeKey: string, score: number, depth: number, exact: boolean): ValueEvent => ({
  nodeKey,
  score,
  depth,
  exact,
});

class ScriptedAgent implements SearchAgent<string, number> {
  private index = 0;

  private readonly script: ScriptStep[];

  private readonly recorder: TreeRecorder<string>;

  private readonly root: VisitEvent<string>;

  constructor(script: ScriptStep[], recorder: TreeRecorder<string>, root: VisitEvent<string>) {
    this.script = script;
    this.recorder = recorder;
    this.root = root;
  }

  start(_state: string): void {
    this.index = 0;
    this.recorder.onVisit(this.root);
  }

  step(): SearchStatus {
    const step = this.script[this.index];
    this.index += 1;
    if (!step) {
      return 'done';
    }
    for (const event of step.visits ?? []) {
      this.recorder.onVisit(event);
    }
    for (const event of step.values ?? []) {
      this.recorder.onValue(event);
    }
    return step.done ? 'done' : 'running';
  }

  bestMove(): number | null {
    return null;
  }

  stats(): SearchStats {
    return {
      nodesVisited: 0,
      nodesPruned: 0,
      ttHits: 0,
      ttMisses: 0,
      leavesEvaluated: 0,
      elapsedSteps: 0,
      effectiveDepth: 0,
    };
  }
}

interface Harness {
  model: TreeModel<string>;
  stepper: Stepper<string>;
}

const harness = (script: ScriptStep[], root: VisitEvent<string>): Harness => {
  const model = new TreeModel<string>();
  const agent = new ScriptedAgent(script, createModelRecorder(model), root);
  return { model, stepper: createStepper(agent, model) };
};

const mainScript: ScriptStep[] = [
  { visits: [visit('a', 'r', 1, 'expand', 0)] },
  {
    visits: [visit('a1', 'a', 2, 'terminal', 0)],
    values: [value('a1', 0.5, 2, true), value('a', 0.5, 1, false)],
  },
  {
    visits: [visit('a2', 'a', 2, 'leaf', 1)],
    values: [value('a2', -0.2, 2, true), value('a', 0.5, 1, true)],
  },
  { visits: [visit('b', 'r', 1, 'expand', 1)] },
  {
    visits: [visit('b1', 'b', 2, 'leaf', 0)],
    values: [value('b1', 0.1, 2, true), value('b', 0.1, 1, true)],
  },
  {
    visits: [visit('c', 'r', 1, 'leaf', 2)],
    values: [value('c', -0.3, 2, true), value('r', 0.5, 0, true)],
  },
  { done: true },
];

const prunedScript: ScriptStep[] = [
  { visits: [visit('x', 'r2', 1, 'expand', 0)] },
  {
    visits: [visit('x1', 'x', 2, 'leaf', 0)],
    values: [value('x1', 0.9, 2, true), value('x', 0.9, 1, false)],
  },
  { visits: [visit('x2', 'x', 2, 'pruned', 1)] },
  { values: [value('x', 0.9, 1, false)] },
  { visits: [visit('y', 'r2', 1, 'expand', 1)], values: [value('r2', 0.9, 0, false)] },
  {
    visits: [visit('y1', 'y', 2, 'leaf', 0)],
    values: [value('y1', -0.5, 2, true), value('y', -0.5, 1, true), value('r2', 0.9, 0, true)],
  },
  { done: true },
];

describe('stepper', () => {
  it('adopts the root node as the initial cursor on start', () => {
    const { model, stepper } = harness(mainScript, visit('r', null, 0, 'expand'));
    stepper.start('R');
    expect(model.root()?.nodeKey).toBe('r');
    expect(stepper.current()).toBe('r');
    expect(stepper.path()).toEqual(['r']);
    expect(stepper.isDone()).toBe(false);
  });

  it('steps in by descending into each newly visited node', () => {
    const { model, stepper } = harness(mainScript, visit('r', null, 0, 'expand'));
    stepper.start('R');
    stepper.stepIn();
    expect(stepper.current()).toBe('a');
    expect(stepper.path()).toEqual(['r', 'a']);
    stepper.stepIn();
    expect(stepper.current()).toBe('a1');
    expect(stepper.path()).toEqual(['r', 'a', 'a1']);
    expect(model.resolved('a1')).toBe(true);
    expect(model.resolved('a')).toBe(false);
  });

  it('steps over to the next sibling once the subtree completes', () => {
    const { model, stepper } = harness(mainScript, visit('r', null, 0, 'expand'));
    stepper.start('R');
    stepper.stepIn();
    stepper.stepIn();
    stepper.stepOver();
    expect(stepper.current()).toBe('a2');
    expect(stepper.path()).toEqual(['r', 'a', 'a2']);
    expect(model.resolved('a')).toBe(true);
  });

  it('steps over to the parent when the node is the last sibling', () => {
    const { stepper } = harness(mainScript, visit('r', null, 0, 'expand'));
    stepper.start('R');
    stepper.stepIn();
    stepper.stepIn();
    stepper.stepIn();
    stepper.stepOver();
    expect(stepper.current()).toBe('a');
    expect(stepper.path()).toEqual(['r', 'a']);
  });

  it('steps out through the remaining siblings and returns to the parent', () => {
    const { stepper } = harness(mainScript, visit('r', null, 0, 'expand'));
    stepper.start('R');
    stepper.stepIn();
    stepper.stepOut();
    expect(stepper.current()).toBe('r');
    expect(stepper.path()).toEqual(['r']);
    expect(stepper.isDone()).toBe(true);
  });

  it('steps out from the root by running the search to completion', () => {
    const { stepper } = harness(mainScript, visit('r', null, 0, 'expand'));
    stepper.start('R');
    stepper.stepOut();
    expect(stepper.isDone()).toBe(true);
    expect(stepper.current()).toBe('r');
  });

  it('steps over from the root by running the whole search', () => {
    const { stepper } = harness(mainScript, visit('r', null, 0, 'expand'));
    stepper.start('R');
    stepper.stepOver();
    expect(stepper.isDone()).toBe(true);
    expect(stepper.current()).toBe('r');
  });

  it('lands on pruned placeholders and resumes stepping from them', () => {
    const { model, stepper } = harness(prunedScript, visit('r2', null, 0, 'expand'));
    stepper.start('R');
    stepper.stepIn();
    stepper.stepIn();
    stepper.stepOver();
    expect(stepper.current()).toBe('x2');
    expect(stepper.node()?.kind).toBe('pruned');
    stepper.stepOver();
    expect(stepper.current()).toBe('x');
    expect(model.resolved('x')).toBe(false);
  });

  it('leaves the cursor in place when a step emits no visit', () => {
    const { stepper } = harness(prunedScript, visit('r2', null, 0, 'expand'));
    stepper.start('R');
    stepper.stepIn();
    stepper.stepIn();
    stepper.stepIn();
    expect(stepper.current()).toBe('x2');
    stepper.stepIn();
    expect(stepper.current()).toBe('x2');
  });

  it('gates every primitive once the search is done', () => {
    const { stepper } = harness(mainScript, visit('r', null, 0, 'expand'));
    stepper.start('R');
    stepper.stepOut();
    expect(stepper.isDone()).toBe(true);
    expect(stepper.canStepIn()).toBe(false);
    expect(stepper.canStepOver()).toBe(false);
    expect(stepper.canStepOut()).toBe(false);
    stepper.stepIn();
    stepper.stepOver();
    stepper.stepOut();
    expect(stepper.current()).toBe('r');
  });

  it('ignores every primitive before start', () => {
    const { stepper } = harness(mainScript, visit('r', null, 0, 'expand'));
    expect(stepper.canStepIn()).toBe(false);
    expect(stepper.canStepOver()).toBe(false);
    expect(stepper.canStepOut()).toBe(false);
    stepper.stepIn();
    stepper.stepOver();
    stepper.stepOut();
    expect(stepper.current()).toBeNull();
    expect(stepper.isDone()).toBe(false);
  });
});
