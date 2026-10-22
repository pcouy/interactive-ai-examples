import { describe, expect, it } from 'vitest';
import type { FineStepState, FrozenLakeAction } from '../../core/frozen-lake.js';
import type { QUpdateResult, Transition } from '../../core/q-learning.js';
import { FrozenLake } from './drl-frozen-lake.js';
import { FrozenLakeReplay } from './drl-frozen-lake-replay.js';
import {
  ACTION_GLYPHS,
  FROZEN_LAKE_REPLAY_EVENT,
  type FrozenLakeReplayState,
  replayHeader,
  replayRows,
  schematicView,
  updateView,
} from './frozen-lake-ui.js';
import {
  FROZEN_LAKE_MESSAGES,
  frozenLakeActionName,
  frozenLakeMessage,
  frozenLakePhaseLabel,
} from './messages.js';
import { REPLAY_MESSAGES } from './replay-messages.js';

/** Complete transition, with defaults that a test can override. */
function transition(
  overrides: Partial<Transition<number, FrozenLakeAction>> = {},
): Transition<number, FrozenLakeAction> {
  return {
    observation: 0,
    action: 'right',
    reward: 0,
    nextObservation: 1,
    done: false,
    ...overrides,
  };
}

/** Replay state around one batch, with a non-empty buffer. */
function replayState(
  batch: readonly Transition<number, FrozenLakeAction>[],
  bufferSize = 120,
  capacity = 1000,
): FrozenLakeReplayState {
  return { enabled: true, batch, bufferSize, capacity };
}

/** No DOM in the node test environment: neutralise Lit's async render pass. */
function preventRender<T extends object>(element: T): T {
  (element as unknown as { performUpdate(): void }).performUpdate = () => {};
  return element;
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

/** Renders an element template to flat text without connecting it to a DOM. */
function renderText(element: object): string {
  return templateText((element as { render(): unknown }).render());
}

/** Fine-step state in the observe phase, with no action yet. */
function fineStep(overrides: Partial<FineStepState> = {}): FineStepState {
  return {
    phase: 'observe',
    nextPhase: 'act',
    observation: 0,
    action: null,
    nextObservation: null,
    reward: null,
    done: null,
    ...overrides,
  };
}

describe('ACTION_GLYPHS', () => {
  it('maps every action to its arrow', () => {
    expect(ACTION_GLYPHS).toEqual({ left: '←', down: '↓', right: '→', up: '↑' });
  });
});

describe('FROZEN_LAKE_REPLAY_EVENT', () => {
  it('uses the shared event name', () => {
    expect(FROZEN_LAKE_REPLAY_EVENT).toBe('drl-replay-state');
  });
});

describe('replayRows', () => {
  it('returns the newest batch entry first and marks it as dernier pas', () => {
    const rows = replayRows(
      replayState([
        transition({ observation: 0, nextObservation: 1 }),
        transition({ observation: 1, nextObservation: 2 }),
        transition({ observation: 2, nextObservation: 3 }),
      ]),
    );
    expect(rows).toEqual([
      { text: 'case 2 → case 3 · r 0 · dernier pas', newest: true },
      { text: 'case 0 → case 1 · r 0', newest: false },
      { text: 'case 1 → case 2 · r 0', newest: false },
    ]);
  });

  it('adds the fin marker to a terminal row', () => {
    const rows = replayRows(
      replayState([transition({ observation: 14, nextObservation: 15, reward: 1, done: true })]),
    );
    expect(rows[0].text).toBe('case 14 → case 15 · r 1 · fin · dernier pas');
  });

  it('keeps the fin marker before the dernier pas marker', () => {
    const rows = replayRows(
      replayState([
        transition({ observation: 5, nextObservation: 6, done: false }),
        transition({ observation: 9, nextObservation: 10, reward: 1, done: true }),
      ]),
    );
    expect(rows[0].text.endsWith(' · fin · dernier pas')).toBe(true);
    expect(rows[0].newest).toBe(true);
    expect(rows[1].text).not.toContain('fin');
    expect(rows[1].newest).toBe(false);
  });

  it('returns no row for an empty batch', () => {
    expect(replayRows(replayState([]))).toEqual([]);
  });
});

describe('replayHeader', () => {
  it('states the batch length, the buffer size and the capacity', () => {
    const header = replayHeader(replayState([transition(), transition(), transition()], 42, 500));
    expect(header).toBe('Rejeu · échantillon 3 · mémoire 42/500');
  });

  it('states a zero batch length for an empty batch', () => {
    expect(replayHeader(replayState([], 0, 1000))).toBe('Rejeu · échantillon 0 · mémoire 0/1000');
  });
});

describe('schematicView', () => {
  it('shows only the observation in the observe phase', () => {
    expect(schematicView(fineStep({ observation: 5 }))).toEqual({
      observation: 5,
      action: null,
      nextObservation: null,
      reward: null,
    });
  });

  it('ignores stale transition fields in the observe phase', () => {
    const state = fineStep({ observation: 6, nextObservation: 9, reward: 0.5 });
    expect(schematicView(state)).toEqual({
      observation: 6,
      action: null,
      nextObservation: null,
      reward: null,
    });
  });

  it('shows the observation and the action in the act phase', () => {
    const state = fineStep({
      phase: 'act',
      nextPhase: 'transition',
      observation: 4,
      action: 'up',
    });
    expect(schematicView(state)).toEqual({
      observation: 4,
      action: 'up',
      nextObservation: null,
      reward: null,
    });
  });

  it('shows the full transition in the transition phase', () => {
    const state = fineStep({
      phase: 'transition',
      nextPhase: 'observe',
      observation: 4,
      action: 'down',
      nextObservation: 9,
      reward: 0.5,
      done: false,
    });
    expect(schematicView(state)).toEqual({
      observation: 4,
      action: 'down',
      nextObservation: 9,
      reward: 0.5,
    });
  });
});

describe('updateView', () => {
  /** Fixed update whose target and updated value follow the Q-learning rule. */
  const update: QUpdateResult<number, FrozenLakeAction> = {
    transition: transition({ observation: 0, action: 'right', reward: 1, nextObservation: 1 }),
    previousValue: 0.25,
    nextValue: 0.5,
    discountFactor: 0.9,
    bootstrap: 0.45,
    target: 1.45,
    tdError: 1.2,
    learningRate: 0.1,
    updatedValue: 0.37,
  };

  it('formats every value with three decimals', () => {
    expect(updateView(update)).toMatchObject({
      reward: '1.000',
      gamma: '0.900',
      nextValue: '0.500',
      bootstrap: '0.450',
      target: '1.450',
      previous: '0.250',
      alpha: '0.100',
      updated: '0.370',
    });
  });

  it('builds both French formulas with the substituted values', () => {
    const view = updateView(update);
    expect(view.targetFormula).toBe("r + γ · max Q(s', a') = 1.000 + 0.900 × 0.500 = 1.450");
    expect(view.updateFormula).toBe('Q(s, a) ← 0.250 + 0.100 × (1.450 − 0.250) = 0.370');
  });

  it('keeps the same mathematical formulas in English', () => {
    const french = updateView(update);
    const english = updateView(update, 'en');
    expect(english.targetFormula).toBe(french.targetFormula);
    expect(english.updateFormula).toBe(french.updateFormula);
  });
});

describe('Frozen Lake catalog', () => {
  it('keeps the French strings unchanged', () => {
    expect(FROZEN_LAKE_MESSAGES.fr.loopTitle).toBe('Boucle agent-environnement');
    expect(FROZEN_LAKE_MESSAGES.fr.updateTitle).toBe('Dernière mise à jour');
    expect(FROZEN_LAKE_MESSAGES.fr.replayToggle).toBe('Rejeu');
    expect(FROZEN_LAKE_MESSAGES.fr.status).toBe(
      'Épisodes {episodes} · Réussite {success} % · Pas {steps}',
    );
    expect(frozenLakeMessage('fr', 'status', { episodes: 3, success: 50, steps: 7 })).toBe(
      'Épisodes 3 · Réussite 50 % · Pas 7',
    );
    expect(frozenLakeMessage('fr', 'caseLabel', { observation: 5, action: '→ droite' })).toBe(
      'Case 5 · → droite',
    );
  });

  it('translates the panel labels and the status to English', () => {
    expect(frozenLakeMessage('en', 'loopTitle')).toBe('Agent-environment loop');
    expect(frozenLakeMessage('en', 'updateTitle')).toBe('Latest update');
    expect(frozenLakeMessage('en', 'replayToggle')).toBe('Replay');
    expect(frozenLakeMessage('en', 'nextPhase')).toBe('Next phase');
    expect(frozenLakeMessage('en', 'status', { episodes: 3, success: 50, steps: 7 })).toBe(
      'Episodes 3 · Success 50 % · Steps 7',
    );
  });

  it('translates action names, phase labels and the case label', () => {
    expect(frozenLakeActionName('fr', 'right')).toBe('droite');
    expect(frozenLakeActionName('fr', 'left')).toBe('gauche');
    expect(frozenLakeActionName('fr', 'up')).toBe('haut');
    expect(frozenLakeActionName('fr', 'down')).toBe('bas');
    expect(frozenLakeActionName('en', 'right')).toBe('right');

    expect(frozenLakePhaseLabel('fr', 'observe')).toBe('Observation');
    expect(frozenLakePhaseLabel('en', 'transition')).toBe('Transition');
    expect(frozenLakeMessage('en', 'caseLabel', { observation: 5, action: '→ right' })).toBe(
      'Cell 5 · → right',
    );
  });
});

describe('Frozen Lake replay catalog', () => {
  it('keeps the French aria label and empty message unchanged', () => {
    expect(REPLAY_MESSAGES.fr.ariaLabel).toBe('Rejeu');
    expect(REPLAY_MESSAGES.fr.empty).toBe("Aucun échantillon pour l'instant.");
  });
});

describe('replay localization', () => {
  it('renders the English header when the locale is en', () => {
    const header = replayHeader(
      replayState([transition(), transition(), transition()], 42, 500),
      'en',
    );
    expect(header).toBe('Replay · sample 3 · memory 42/500');
  });

  it('renders the English end and last-step markers when the locale is en', () => {
    const rows = replayRows(
      replayState([transition({ observation: 14, nextObservation: 15, reward: 1, done: true })]),
      'en',
    );
    expect(rows[0].text).toBe('cell 14 → cell 15 · r 1 · end · last step');
  });
});

describe('Frozen Lake element localization', () => {
  it('renders the French labels by default', () => {
    const demo = preventRender(new FrozenLake());
    const text = renderText(demo);
    expect(text).toContain('Frozen Lake — Q-learning tabulaire');
    expect(text).toContain('Contrôles de Frozen Lake');
    expect(text).toContain('Boucle agent-environnement');
    expect(text).toContain('Dernière mise à jour');
    expect(text).toContain('Phase suivante');
  });

  it('renders complete English labels after a locale change', () => {
    const demo = preventRender(new FrozenLake());
    demo.locale = 'en';
    const text = renderText(demo);
    expect(text).toContain('Frozen Lake — tabular Q-learning');
    expect(text).toContain('Frozen Lake controls');
    expect(text).toContain('Frozen Lake grid');
    expect(text).toContain('Reward curve');
    expect(text).toContain('Agent-environment loop');
    expect(text).toContain('Environment');
    expect(text).toContain('Reward r');
    expect(text).toContain('Action a');
    expect(text).toContain('Step phases');
    expect(text).toContain('Next phase');
    expect(text).toContain('Latest update');
    expect(text).toContain('No update yet.');
    expect(text).toContain('Factor γ');
    expect(text).toContain("Max next value Q(s', a')");
    expect(text).toContain('Bootstrapping γ');
    expect(text).toContain('Target');
    expect(text).toContain('Previous value Q(s, a)');
    expect(text).toContain('Rate α');
    expect(text).toContain('Updated value');
    expect(text).not.toContain('Boucle agent-environnement');
    expect(text).not.toContain('Dernière mise à jour');
    expect(text).not.toContain('Phase suivante');
    expect(text).not.toContain('Récompense');
    expect(text).not.toContain('Épisodes');
    expect(text).not.toContain('Aucune mise à jour');
  });
});

describe('Frozen Lake replay element localization', () => {
  const batch = [transition({ observation: 14, nextObservation: 15, reward: 1, done: true })];

  it('renders its French header, row and aria label by default', () => {
    const replay = preventRender(new FrozenLakeReplay());
    replay.replayState = replayState(batch);
    const text = renderText(replay);
    expect(text).toContain('aria-label=Rejeu');
    expect(text).toContain('Rejeu · échantillon 1 · mémoire 120/1000');
    expect(text).toContain('case 14 → case 15 · r 1 · fin · dernier pas');
  });

  it('renders its English header, row and aria label after a locale change', () => {
    const replay = preventRender(new FrozenLakeReplay());
    replay.replayState = replayState(batch);
    replay.locale = 'en';
    const text = renderText(replay);
    expect(text).toContain('aria-label=Replay');
    expect(text).toContain('Replay · sample 1 · memory 120/1000');
    expect(text).toContain('cell 14 → cell 15 · r 1 · end · last step');
    expect(text).not.toContain('dernier pas');
  });

  it('localizes the empty message', () => {
    const replay = preventRender(new FrozenLakeReplay());
    replay.replayState = replayState([]);
    expect(renderText(replay)).toContain("Aucun échantillon pour l'instant.");
    replay.locale = 'en';
    expect(renderText(replay)).toContain('No sample yet.');
  });
});
