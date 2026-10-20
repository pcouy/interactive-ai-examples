import { describe, expect, it, vi } from 'vitest';
import type { VisitEvent } from '../../core/contracts.js';
import { defaultCustomSource } from '../../core/search/evaluators/custom.js';
import { TreeModel } from '../../core/search/tree-model.js';
import { TIC_TAC_TOE_GAME_CONFIG } from '../shared/create-game-component.js';
import { type DeepBlueGameId, DrlDeepBluePage, firstMoveOnPath } from './drl-deep-blue-page.js';

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
  if (typeof template === 'string') {
    return template;
  }
  if (Array.isArray(template)) {
    return template.map((item) => collectStrings(item)).join('');
  }
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

  it('highlights the active mode even when the JSON has extra spaces', () => {
    const page = createPage();
    page.players = '[ "human" , "agent" ]';
    const internals = page as unknown as { renderVsModeRow(): unknown };
    const pressed = collectBindings(internals.renderVsModeRow())
      .filter((binding) => binding.attribute === 'aria-pressed')
      .map((binding) => binding.value);
    expect(pressed).toEqual([true, false, false]);
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

describe('DrlDeepBluePage input handlers', () => {
  interface Internals {
    onDepthChange(event: Event): void;
    onSamplesChange(event: Event): void;
    onDiscountChange(event: Event): void;
    onEvaluatorChange(event: Event): void;
  }

  function internals(page: DrlDeepBluePage): Internals {
    return page as unknown as Internals;
  }

  /** Faux événement de changement portant une valeur texte. */
  function change(value: string): Event {
    return { target: { value } } as unknown as Event;
  }

  it('sets depth to null for an empty value', () => {
    const page = createPage();
    internals(page).onDepthChange(change(''));
    expect(page.depth).toBeNull();
  });

  it('falls back to the game default for a non-finite depth', () => {
    const page = createPage();
    internals(page).onDepthChange(change('not-a-number'));
    expect(page.depth).toBe(TIC_TAC_TOE_GAME_CONFIG.defaultDepth);
  });

  it('clamps a below-minimum depth to the minimum', () => {
    const page = createPage();
    internals(page).onDepthChange(change('0'));
    expect(page.depth).toBe(1);
  });

  it('falls back to the page default for a non-finite rollout sample count', () => {
    const page = createPage();
    internals(page).onSamplesChange(change('not-a-number'));
    expect(page.rolloutSamples).toBe(10);
  });

  it('clamps a below-minimum rollout sample count to the minimum', () => {
    const page = createPage();
    internals(page).onSamplesChange(change('0'));
    expect(page.rolloutSamples).toBe(1);
  });

  it('falls back to 1 for an invalid or out-of-range discount', () => {
    const page = createPage();
    internals(page).onDiscountChange(change('not-a-number'));
    expect(page.discount).toBe(1);
    internals(page).onDiscountChange(change('1.5'));
    expect(page.discount).toBe(1);
    internals(page).onDiscountChange(change('0'));
    expect(page.discount).toBe(1);
  });

  it('parses the evaluator value from the select', () => {
    const page = createPage();
    internals(page).onEvaluatorChange(change('rollout'));
    expect(page.evaluator).toBe('rollout');
    internals(page).onEvaluatorChange(change('nope'));
    expect(page.evaluator).toBe('windows');
  });
});

describe('DrlDeepBluePage render order', () => {
  function renderText(page: DrlDeepBluePage): string {
    const internals = page as unknown as { render(): unknown };
    return collectStrings(internals.render());
  }

  it('renders the header, the modes row, the search row, then the layout row', () => {
    const text = renderText(createPage());
    const order = ['drl-page-header', 'drl-page-modes', 'drl-page-search', 'drl-page-layout'];
    let previous = -1;
    for (const marker of order) {
      const index = text.indexOf(marker);
      expect(index).toBeGreaterThan(previous);
      previous = index;
    }
  });

  it('keeps the search row outside the layout row and the modes row in its own block', () => {
    const text = renderText(createPage());
    const modesStart = text.indexOf('drl-page-modes');
    const searchStart = text.indexOf('drl-page-search');
    const layoutStart = text.indexOf('drl-page-layout');
    // La rangée des modes est un bloc propre, entre l'en-tête et la recherche.
    expect(text.indexOf('Humain vs Agent')).toBeGreaterThan(modesStart);
    expect(searchStart).toBeGreaterThan(text.indexOf('Agent vs Agent'));
    // La rangée de recherche précède la mise en page et n'est pas imbriquée dedans.
    expect(layoutStart).toBeGreaterThan(text.indexOf('Profondeur'));
  });
});

describe('DrlDeepBluePage custom evaluator', () => {
  interface Internals {
    renderSearchFields(): unknown;
    renderCustomBlock(): unknown;
    renderGame(): unknown;
    render(): unknown;
    onEvaluatorChange(event: Event): void;
    onCustomInput(event: Event): void;
    onCustomApply(): void;
    onCustomReset(): void;
    onGamePick(event: Event): void;
  }

  interface PrivateState {
    customSources: Record<DeepBlueGameId, string>;
    customEvaluators: Partial<Record<DeepBlueGameId, unknown>>;
    customError: string | null;
  }

  function internals(page: DrlDeepBluePage): Internals {
    return page as unknown as Internals;
  }

  function priv(page: DrlDeepBluePage): PrivateState {
    return page as unknown as PrivateState;
  }

  /** Faux arbre minimal, installé sans DOM comme dans les tests de pas. */
  function stubTree(page: DrlDeepBluePage): { clearTree: () => void } {
    const tree = { clearTree: vi.fn() };
    Object.defineProperty(page, 'treeElement', { configurable: true, value: tree });
    return tree;
  }

  function change(value: string): Event {
    return { target: { value } } as unknown as Event;
  }

  it('offers the custom option in the evaluator select', () => {
    const page = createPage();
    expect(collectStrings(internals(page).renderSearchFields())).toContain('custom');
  });

  it('renders the custom block only for the custom evaluator', () => {
    const page = createPage();
    const block = (): string => collectStrings(internals(page).renderCustomBlock());
    page.evaluator = 'windows';
    expect(block()).not.toContain('Évaluateur personnalisé');
    page.evaluator = 'custom';
    expect(block()).toContain('Évaluateur personnalisé');
  });

  it('renders the custom block between the search row and the layout', () => {
    const page = createPage();
    page.evaluator = 'custom';
    const text = collectStrings(internals(page).render());
    const searchStart = text.indexOf('drl-page-search');
    const customStart = text.indexOf('drl-page-custom');
    const layoutStart = text.indexOf('drl-page-layout');
    expect(customStart).toBeGreaterThan(searchStart);
    expect(layoutStart).toBeGreaterThan(customStart);
  });

  it('warns that the custom code runs in the page without isolation', () => {
    const page = createPage();
    page.evaluator = 'custom';
    // Le texte du modèle est replié sur plusieurs lignes : on normalise les
    // espaces, comme le rendu HTML le ferait.
    const text = collectStrings(internals(page).renderCustomBlock()).replace(/\s+/g, ' ');
    expect(text).toContain(
      'Code exécuté dans la page, sans isolation. À utiliser uniquement en local avec du code de confiance.',
    );
  });

  it('shows the game default source in the textarea', () => {
    const page = createPage();
    page.evaluator = 'custom';
    expect(boundProperty(internals(page).renderCustomBlock(), 'value')).toBe(
      defaultCustomSource('tic-tac-toe'),
    );
    page.gameKey = 'connect4';
    expect(boundProperty(internals(page).renderCustomBlock(), 'value')).toBe(
      defaultCustomSource('connect4'),
    );
  });

  it('stores the draft for the active game without touching the other game', () => {
    const page = createPage();
    internals(page).onCustomInput(change('return 1;'));
    expect(priv(page).customSources['tic-tac-toe']).toBe('return 1;');
    expect(priv(page).customSources.connect4).toBe(defaultCustomSource('connect4'));
  });

  it('compiles a valid body on apply and clears the error', () => {
    const page = createPage();
    page.evaluator = 'custom';
    const tree = stubTree(page);
    internals(page).onCustomInput(change('return 0.5;'));
    internals(page).onCustomApply();
    expect(priv(page).customEvaluators['tic-tac-toe']).toBeDefined();
    expect(priv(page).customError).toBeNull();
    expect(tree.clearTree).toHaveBeenCalledTimes(1);
  });

  it('compiles the default evaluator when created already in custom mode', () => {
    const page = createPage();
    // Sans DOM, `syncTreeGame` doit pouvoir sortir tôt : arbre et jeu absents.
    Object.defineProperty(page, 'treeElement', { configurable: true, value: null });
    Object.defineProperty(page, 'gameElement', { configurable: true, value: null });
    page.evaluator = 'custom';
    (page as unknown as { updated(changed: Map<string, unknown>): void }).updated(new Map());
    expect(priv(page).customEvaluators['tic-tac-toe']).toBeDefined();
  });

  it('reports an error on apply and keeps the previous evaluator', () => {
    const page = createPage();
    page.evaluator = 'custom';
    stubTree(page);
    internals(page).onCustomApply();
    const previous = priv(page).customEvaluators['tic-tac-toe'];
    expect(previous).toBeDefined();
    internals(page).onCustomInput(change('return 0 +;'));
    internals(page).onCustomApply();
    expect(priv(page).customError).not.toBeNull();
    expect(priv(page).customEvaluators['tic-tac-toe']).toBe(previous);
  });

  it('restores and compiles the default source on reset', () => {
    const page = createPage();
    page.evaluator = 'custom';
    const tree = stubTree(page);
    internals(page).onCustomInput(change('return 0.25;'));
    internals(page).onCustomApply();
    internals(page).onCustomReset();
    expect(priv(page).customSources['tic-tac-toe']).toBe(defaultCustomSource('tic-tac-toe'));
    expect(priv(page).customEvaluators['tic-tac-toe']).toBeDefined();
    expect(priv(page).customError).toBeNull();
    expect(tree.clearTree).toHaveBeenCalledTimes(2);
  });

  it('switches to the new game draft and compiles it when custom is selected', () => {
    const page = createPage();
    internals(page).onEvaluatorChange(change('custom'));
    expect(page.evaluator).toBe('custom');
    expect(priv(page).customEvaluators['tic-tac-toe']).toBeDefined();
    internals(page).onGamePick(change('connect4'));
    expect(page.gameKey).toBe('connect4');
    expect(priv(page).customEvaluators.connect4).toBeDefined();
    expect(boundProperty(internals(page).renderCustomBlock(), 'value')).toBe(
      defaultCustomSource('connect4'),
    );
  });

  it('keeps drafts and evaluators when leaving custom', () => {
    const page = createPage();
    internals(page).onEvaluatorChange(change('custom'));
    const compiled = priv(page).customEvaluators['tic-tac-toe'];
    internals(page).onCustomInput(change('return 0.75;'));
    internals(page).onEvaluatorChange(change('windows'));
    expect(page.evaluator).toBe('windows');
    expect(priv(page).customSources['tic-tac-toe']).toBe('return 0.75;');
    expect(priv(page).customEvaluators['tic-tac-toe']).toBe(compiled);
    internals(page).onEvaluatorChange(change('custom'));
    expect(priv(page).customEvaluators['tic-tac-toe']).toBe(compiled);
  });

  it('binds customEvaluator to the active compiled evaluator and null otherwise', () => {
    const page = createPage();
    expect(boundProperty(internals(page).renderGame(), 'customEvaluator')).toBeNull();
    page.evaluator = 'custom';
    internals(page).onEvaluatorChange(change('custom'));
    const active = priv(page).customEvaluators['tic-tac-toe'];
    expect(active).toBeDefined();
    expect(boundProperty(internals(page).renderGame(), 'customEvaluator')).toBe(active);
    internals(page).onGamePick(change('connect4'));
    expect(boundProperty(internals(page).renderGame(), 'customEvaluator')).toBe(
      priv(page).customEvaluators.connect4,
    );
  });
});
