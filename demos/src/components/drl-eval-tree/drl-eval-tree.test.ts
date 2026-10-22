import { describe, expect, it } from 'vitest';
import { DrlEvalTree } from './drl-eval-tree.js';

/** No DOM in the node test environment: neutralise Lit's async render pass. */
function createTree(): DrlEvalTree<unknown> {
  const tree = new DrlEvalTree<unknown>();
  (tree as unknown as { performUpdate(): void }).performUpdate = () => {};
  return tree;
}

/** Flattens a Lit template into text, including nested templates and values. */
function templateText(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'number') {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value.map(templateText).join('');
  }
  if (typeof value === 'object' && value !== null && 'strings' in value && 'values' in value) {
    const { strings, values } = value as {
      strings: readonly string[];
      values: readonly unknown[];
    };
    let text = '';
    for (let index = 0; index < strings.length; index += 1) {
      text += strings[index];
      if (index < values.length) {
        text += templateText(values[index]);
      }
    }
    return text;
  }
  return '';
}

function renderText(tree: DrlEvalTree<unknown>): string {
  const internals = tree as unknown as { renderContent(): unknown };
  return templateText(internals.renderContent()).replace(/\s+/g, ' ').trim();
}

describe('DrlEvalTree localization', () => {
  it('renders the toolbar in French by default', () => {
    const text = renderText(createTree());
    expect(text).toContain("Débogueur d'arbre de recherche");
    expect(text).toContain('Entrer');
    expect(text).toContain('Passer');
    expect(text).toContain('Sortir');
    expect(text).toContain('Racine = état courant');
    expect(text).toContain('Ajuster auto');
    expect(text).toContain('Plein écran');
    expect(text).toContain('Prof. max');
    expect(text).toContain("Fil d'Ariane");
    expect(text).toContain('lang=fr');
  });

  it('renders the toolbar in English after a locale change', () => {
    const tree = createTree();
    tree.locale = 'en';
    const text = renderText(tree);
    expect(text).toContain('Search tree debugger');
    expect(text).toContain('Enter');
    expect(text).toContain('Skip');
    expect(text).toContain('Exit');
    expect(text).toContain('Root = current state');
    expect(text).toContain('Fit auto');
    expect(text).toContain('Fullscreen');
    expect(text).toContain('Max depth');
    expect(text).toContain('Breadcrumb');
    expect(text).toContain('Step in: go down to a visited node');
    expect(text).toContain('Fit the zoom to the displayed tree');
    expect(text).toContain('lang=en');
    expect(text).not.toContain('Entrer');
    expect(text).not.toContain('Racine = état courant');
  });

  it('translates the bounds title and score labels', () => {
    const tree = createTree();
    const internals = tree as unknown as {
      boundsText: {
        key: string;
        title: string;
        alpha: string;
        beta: string;
        score: string;
        exact: boolean;
        cut: string | null;
      };
    };
    internals.boundsText = {
      key: 'node',
      title: 'Coup 4',
      alpha: '0,10',
      beta: '0,40',
      score: '+0,50',
      exact: true,
      cut: null,
    };
    expect(renderText(tree)).toContain('Développement du nœud');

    tree.locale = 'en';
    internals.boundsText = { ...internals.boundsText, title: 'Move 4' };
    const text = renderText(tree);
    expect(text).toContain('Move 4');
    expect(text).toContain('(exact)');
    expect(text).toContain('Node expansion');
    expect(text).not.toContain('Développement du nœud');
  });

  it('updates the imperative status line in the selected locale', () => {
    const tree = createTree();
    const internals = tree as unknown as {
      statusText: { textContent: string } | null;
      driver: { current(): string | null; isDone(): boolean } | null;
      updateStatus(): void;
    };
    internals.statusText = { textContent: '' };
    internals.updateStatus();
    expect(internals.statusText.textContent).toBe(
      'Nœuds : 0 · Masqués : 0 · Aucune recherche liée',
    );

    internals.driver = { current: () => 'node-1', isDone: () => true };
    tree.locale = 'en';
    internals.updateStatus();
    expect(internals.statusText.textContent).toBe(
      'Nodes: 0 · Hidden: 0 · Cursor: node-1 · Search complete',
    );

    internals.driver = { current: () => null, isDone: () => false };
    internals.updateStatus();
    expect(internals.statusText.textContent).toBe(
      'Nodes: 0 · Hidden: 0 · Cursor: – · Search in progress',
    );
  });

  it('translates node tooltips', () => {
    const tree = createTree();
    const internals = tree as unknown as {
      nodeTitle(node: unknown): string;
    };
    const node = {
      nodeKey: 'node',
      parentKey: null,
      move: { index: 4 },
      state: undefined,
      depth: 1,
      kind: 'expand',
      score: 0.5,
      exact: true,
      window: { alpha: 0.1, beta: 0.4 },
    };
    expect(internals.nodeTitle(node)).toBe('Score +0,50 (exact) · α +0,10 · β +0,40');

    tree.locale = 'en';
    expect(internals.nodeTitle(node)).toBe('Score +0.50 (exact) · α +0.10 · β +0.40');

    const unknown = { ...node, score: null, exact: false, kind: 'terminal' };
    expect(internals.nodeTitle(unknown)).toBe('Unknown score · α +0.10 · β +0.40');
  });
});
