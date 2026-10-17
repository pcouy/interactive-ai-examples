import type { SearchAgent, SearchStatus } from '../contracts.js';
import type { TreeModel, TreeNode } from './tree-model.js';

export interface Stepper<S> {
  start(state: S): void;
  stepIn(): void;
  stepOver(): void;
  stepOut(): void;
  canStepIn(): boolean;
  canStepOver(): boolean;
  canStepOut(): boolean;
  isDone(): boolean;
  current(): string | null;
  node(): TreeNode<S> | null;
  path(): readonly string[];
}

export function createStepper<S, M>(agent: SearchAgent<S, M>, model: TreeModel<S>): Stepper<S> {
  let stack: string[] = [];
  let started = false;
  let done = false;

  const advance = (): SearchStatus => {
    const status = agent.step();
    if (status === 'done') {
      done = true;
    }
    return status;
  };

  const currentKey = (): string | null => stack.at(-1) ?? null;

  const setCursor = (key: string): void => {
    const chain: string[] = [];
    let cursor: string | null = key;
    while (cursor !== null) {
      chain.push(cursor);
      cursor = model.node(cursor)?.parentKey ?? null;
    }
    chain.reverse();
    stack = chain;
  };

  const isOutsideSubtree = (key: string, subtreeRoot: string): boolean => {
    let cursor: string | null = key;
    while (cursor !== null) {
      if (cursor === subtreeRoot) {
        return false;
      }
      cursor = model.node(cursor)?.parentKey ?? null;
    }
    return true;
  };

  const nextSiblingKey = (key: string): string | null => {
    const node = model.node(key);
    if (!node || node.parentKey === null) {
      return null;
    }
    const siblings = model.children(node.parentKey);
    const index = siblings.findIndex((child) => child.nodeKey === key);
    if (index < 0 || index + 1 >= siblings.length) {
      return null;
    }
    return siblings[index + 1]?.nodeKey ?? null;
  };

  const start = (state: S): void => {
    agent.start(state);
    started = true;
    done = false;
    const root = model.root();
    stack = root ? [root.nodeKey] : [];
  };

  const stepIn = (): void => {
    if (!started || done) {
      return;
    }
    advance();
    const visited = model.lastVisited();
    if (visited) {
      setCursor(visited.nodeKey);
      return;
    }
    if (stack.length === 0) {
      const root = model.root();
      if (root) {
        setCursor(root.nodeKey);
      }
    }
  };

  const stepOver = (): void => {
    const key = currentKey();
    if (!started || done || key === null) {
      return;
    }
    const node = model.node(key);
    if (!node) {
      return;
    }
    for (;;) {
      const status = advance();
      const visited = model.lastVisited();
      if (visited && isOutsideSubtree(visited.nodeKey, key)) {
        const isNextSibling =
          node.parentKey !== null &&
          visited.parentKey === node.parentKey &&
          visited.order > node.order;
        setCursor(isNextSibling ? visited.nodeKey : (node.parentKey ?? key));
        return;
      }
      if (status === 'done') {
        break;
      }
    }
    setCursor(nextSiblingKey(key) ?? node.parentKey ?? key);
  };

  const stepOut = (): void => {
    const key = currentKey();
    if (!started || done || key === null) {
      return;
    }
    const node = model.node(key);
    if (!node) {
      return;
    }
    const parentKey = node.parentKey;
    for (;;) {
      const status = advance();
      const visited = model.lastVisited();
      if (parentKey !== null && visited && isOutsideSubtree(visited.nodeKey, parentKey)) {
        break;
      }
      if (status === 'done') {
        break;
      }
    }
    setCursor(parentKey ?? key);
  };

  return {
    start,
    stepIn,
    stepOver,
    stepOut,
    canStepIn: () => started && !done,
    canStepOver: () => started && !done && currentKey() !== null,
    canStepOut: () => started && !done && currentKey() !== null,
    isDone: () => done,
    current: currentKey,
    node: () => {
      const key = currentKey();
      return key === null ? null : model.node(key);
    },
    path: () => stack.slice(),
  };
}
