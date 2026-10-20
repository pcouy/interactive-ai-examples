import { describe, expect, it, vi } from 'vitest';
import type { VisitEvent } from '../../core/contracts.js';
import { TreeModel } from '../../core/search/tree-model.js';
import { DrlDeepBluePage, firstMoveOnPath } from './drl-deep-blue-page.js';

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

/** Valeur liée à une propriété Lit (`.nom=`) dans un template, sans DOM. */
function boundProperty(template: unknown, property: string): unknown {
  if (typeof template !== 'object' || template === null) {
    return undefined;
  }
  const { strings, values } = template as {
    strings?: readonly string[];
    values?: readonly unknown[];
  };
  if (strings === undefined || values === undefined) {
    return undefined;
  }
  for (let index = 0; index < values.length; index += 1) {
    if (strings[index]?.endsWith(`.${property}=`)) {
      return values[index];
    }
  }
  return undefined;
}

/** Gestionnaire lié à un événement Lit (`@nom=`) dans un template, sans DOM. */
function boundEvent(template: unknown, event: string): unknown {
  if (typeof template !== 'object' || template === null) {
    return undefined;
  }
  const { strings, values } = template as {
    strings?: readonly string[];
    values?: readonly unknown[];
  };
  if (strings === undefined || values === undefined) {
    return undefined;
  }
  for (let index = 0; index < values.length; index += 1) {
    if (strings[index]?.endsWith(`@${event}=`)) {
      return values[index];
    }
  }
  return undefined;
}

/** Liaison Lit dans un template, avec le nom d'attribut tel que `.prop`, `@event` ou `?bool`. */
interface TemplateBinding {
  readonly attribute: string;
  readonly value: unknown;
}

const BINDING_RE = /([?@.]?[A-Za-z0-9_-]+)=$/;

/** Remonte toutes les liaisons d'un template, en parcourant les templates imbriqués. */
function collectBindings(template: unknown): TemplateBinding[] {
  if (Array.isArray(template)) {
    return template.flatMap((item) => collectBindings(item));
  }
  if (typeof template !== 'object' || template === null) {
    return [];
  }
  const { strings, values } = template as {
    strings?: readonly string[];
    values?: readonly unknown[];
  };
  if (strings === undefined || values === undefined) {
    return [];
  }
  const bindings: TemplateBinding[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const match = BINDING_RE.exec(strings[index] ?? '');
    if (match !== null) {
      bindings.push({ attribute: match[1], value: values[index] });
    }
    bindings.push(...collectBindings(values[index]));
  }
  return bindings;
}

/** Texte statique d'un template, templates imbriqués inclus, dans l'ordre du document. */
function collectStrings(template: unknown): string {
  if (typeof template !== 'object' || template === null) {
    return '';
  }
  const { strings, values } = template as {
    strings?: readonly string[];
    values?: readonly unknown[];
  };
  if (strings === undefined || values === undefined) {
    return '';
  }
  let text = '';
  for (let index = 0; index < strings.length; index += 1) {
    text += strings[index];
    if (index < values.length) {
      text += collectStrings(values[index]);
    }
  }
  return text;
}

/** Page sans DOM : `performUpdate` neutralisé pour inspecter les templates. */
function createPage(): DrlDeepBluePage {
  const page = new DrlDeepBluePage();
  (page as unknown as { performUpdate(): void }).performUpdate = () => {};
  return page;
}

describe('DrlDeepBluePage strict pruning forwarding', () => {
  it('binds strictPruning on both game elements', () => {
    const page = new DrlDeepBluePage();
    (page as unknown as { performUpdate(): void }).performUpdate = () => {};
    page.strictPruning = true;
    const internals = page as unknown as { renderGame(): unknown };
    // Les deux branches partagent le même binding : on vérifie le morpion puis
    // la puissance 4, la page ne devant jamais diverger de la recherche rejouée.
    expect(boundProperty(internals.renderGame(), 'strictPruning')).toBe(true);
    page.gameKey = 'connect4';
    expect(boundProperty(internals.renderGame(), 'strictPruning')).toBe(true);
  });
});

describe('DrlDeepBluePage step forwarding', () => {
  function newPage(): DrlDeepBluePage {
    const page = new DrlDeepBluePage();
    (page as unknown as { performUpdate(): void }).performUpdate = () => {};
    return page;
  }

  /** Remplace le getter `treeElement` par un faux arbre, sans DOM. */
  function stubTree(page: DrlDeepBluePage, tree: unknown): void {
    Object.defineProperty(page, 'treeElement', { configurable: true, value: tree });
  }

  it('binds drl-step on both game elements', () => {
    const page = newPage();
    const internals = page as unknown as { renderGame(): unknown };
    // Identité vérifiée par rapport au gestionnaire de l'instance : un mauvais
    // handler lié sur un jeu ne doit pas passer inaperçu.
    const handler = (page as unknown as { onGameStep: unknown }).onGameStep;
    // Les deux branches partagent le même binding : on vérifie le morpion puis
    // la puissance 4, la page ne devant jamais diverger entre les deux jeux.
    expect(boundEvent(internals.renderGame(), 'drl-step')).toBe(handler);
    page.gameKey = 'connect4';
    expect(boundEvent(internals.renderGame(), 'drl-step')).toBe(handler);
  });

  it('forwards one step to the tree when the tree is paused', () => {
    const page = newPage();
    const step = vi.fn();
    stubTree(page, { running: false, step });
    (page as unknown as { onGameStep(): void }).onGameStep();
    expect(step).toHaveBeenCalledTimes(1);
  });

  it('does not forward a step when the tree is running', () => {
    const page = newPage();
    const step = vi.fn();
    stubTree(page, { running: true, step });
    (page as unknown as { onGameStep(): void }).onGameStep();
    expect(step).not.toHaveBeenCalled();
  });

  it('does nothing when the tree element is missing', () => {
    const page = newPage();
    stubTree(page, null);
    expect(() => (page as unknown as { onGameStep(): void }).onGameStep()).not.toThrow();
  });
});

describe('DrlDeepBluePage configuration controls', () => {
  interface Internals {
    render(): unknown;
    renderGame(): unknown;
    onPruningChange(event: Event): void;
  }

  function internals(page: DrlDeepBluePage): Internals {
    return page as unknown as Internals;
  }

  it('defaults pruning to false and forwards it to both game elements', () => {
    const page = createPage();
    expect(boundProperty(internals(page).renderGame(), 'pruning')).toBe(false);
    page.pruning = true;
    expect(boundProperty(internals(page).renderGame(), 'pruning')).toBe(true);
    page.gameKey = 'connect4';
    expect(boundProperty(internals(page).renderGame(), 'pruning')).toBe(true);
  });

  it('defaults rolloutSamples to 10 and forwards it to both game elements', () => {
    const page = createPage();
    expect(boundProperty(internals(page).renderGame(), 'rolloutSamples')).toBe(10);
    page.rolloutSamples = 25;
    expect(boundProperty(internals(page).renderGame(), 'rolloutSamples')).toBe(25);
    page.gameKey = 'connect4';
    expect(boundProperty(internals(page).renderGame(), 'rolloutSamples')).toBe(25);
  });

  it('binds showSearchFields to false on both game elements', () => {
    const page = createPage();
    expect(boundProperty(internals(page).renderGame(), 'showSearchFields')).toBe(false);
    page.gameKey = 'connect4';
    expect(boundProperty(internals(page).renderGame(), 'showSearchFields')).toBe(false);
  });

  it('binds autoFit to the tree and reflects its value', () => {
    const page = createPage();
    expect(boundProperty(internals(page).render(), 'autoFit')).toBe(true);
    page.autoFit = false;
    expect(boundProperty(internals(page).render(), 'autoFit')).toBe(false);
  });

  it('sets a Boolean from the pruning select', () => {
    const page = createPage();
    const onPruningChange = internals(page).onPruningChange;
    onPruningChange({ target: { value: 'minimax' } } as unknown as Event);
    expect(page.pruning).toBe(false);
    onPruningChange({ target: { value: 'alpha-beta' } } as unknown as Event);
    expect(page.pruning).toBe(true);
  });
});

describe('DrlDeepBluePage vs-mode controls', () => {
  it('lists the three player modes and updates players on click', () => {
    const page = createPage();
    const internals = page as unknown as {
      renderVsModeRow(): unknown;
      onModePick(event: Event): void;
    };
    const modes = collectBindings(internals.renderVsModeRow())
      .filter((binding) => binding.attribute === 'data-players')
      .map((binding) => binding.value);
    expect(modes).toEqual(['["human","agent"]', '["agent","human"]', '["agent","agent"]']);

    const pressed = (): unknown[] =>
      collectBindings(internals.renderVsModeRow())
        .filter((binding) => binding.attribute === 'aria-pressed')
        .map((binding) => binding.value);
    expect(pressed()).toEqual([true, false, false]);

    internals.onModePick({
      target: { dataset: { players: '["agent","agent"]' } },
    } as unknown as Event);
    expect(page.players).toBe('["agent","agent"]');
    expect(pressed()).toEqual([false, false, true]);
  });
});

describe('DrlDeepBluePage search fields', () => {
  function searchFields(page: DrlDeepBluePage): string {
    const internals = page as unknown as { renderSearchFields(): unknown };
    return collectStrings(internals.renderSearchFields());
  }

  it('shows rollout samples only for the rollout evaluator', () => {
    const page = createPage();
    page.evaluator = 'windows';
    expect(searchFields(page)).not.toContain('Échantillons de rollout');
    page.evaluator = 'rollout';
    expect(searchFields(page)).toContain('Échantillons de rollout');
  });

  it('hides discount and strict pruning for the random evaluator', () => {
    const page = createPage();
    page.evaluator = 'windows';
    const windowsText = searchFields(page);
    expect(windowsText).toContain('Actualisation');
    expect(windowsText).toContain('Élagage strict');
    page.evaluator = 'random';
    const randomText = searchFields(page);
    expect(randomText).not.toContain('Actualisation');
    expect(randomText).not.toContain('Élagage strict');
  });

  it('orders the controls depth, evaluator, samples, discount, pruning, strict, transposition', () => {
    const page = createPage();
    page.evaluator = 'rollout';
    const text = searchFields(page);
    const markers = [
      'Profondeur',
      'Évaluateur',
      'Échantillons de rollout',
      'Actualisation',
      'Élagage',
      'Élagage strict',
      'Transposition',
    ];
    let previous = -1;
    for (const marker of markers) {
      const index = text.indexOf(marker);
      expect(index).toBeGreaterThan(previous);
      previous = index;
    }
  });
});
