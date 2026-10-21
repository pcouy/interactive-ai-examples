import { describe, expect, it } from 'vitest';
import type { FineStepState, FrozenLakeAction } from '../../core/frozen-lake.js';
import type { QUpdateResult, Transition } from '../../core/q-learning.js';
import {
  ACTION_GLYPHS,
  FROZEN_LAKE_REPLAY_EVENT,
  type FrozenLakeReplayState,
  replayHeader,
  replayRows,
  schematicView,
  updateView,
} from './frozen-lake-ui.js';

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
  /** Dummy update whose transition fields are easy to tell apart from a state. */
  function updateResult(
    overrides: Partial<Transition<number, FrozenLakeAction>> = {},
  ): QUpdateResult<number, FrozenLakeAction> {
    return {
      transition: transition(overrides),
      previousValue: 0,
      nextValue: 0,
      discountFactor: 0.9,
      bootstrap: 0,
      target: 0,
      tdError: 0,
      learningRate: 0.1,
      updatedValue: 0,
    };
  }

  it('shows only the observation in the observe phase without an update', () => {
    expect(schematicView(fineStep({ observation: 5 }), null)).toEqual({
      observation: 5,
      action: null,
      nextObservation: null,
      reward: null,
      fallback: false,
    });
  });

  it('shows the observation and the action in the act phase', () => {
    const state = fineStep({
      phase: 'act',
      nextPhase: 'transition',
      observation: 4,
      action: 'up',
    });
    // Live state wins over the stale update.
    const staleUpdate = updateResult({
      observation: 30,
      action: 'left',
      reward: -2,
      nextObservation: 31,
      done: true,
    });
    expect(schematicView(state, staleUpdate)).toEqual({
      observation: 4,
      action: 'up',
      nextObservation: null,
      reward: null,
      fallback: false,
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
    // Live state wins over the stale update.
    const staleUpdate = updateResult({
      observation: 30,
      action: 'left',
      reward: -2,
      nextObservation: 31,
      done: true,
    });
    expect(schematicView(state, staleUpdate)).toEqual({
      observation: 4,
      action: 'down',
      nextObservation: 9,
      reward: 0.5,
      fallback: false,
    });
  });

  it('uses one whole transition as the fallback in the observe phase', () => {
    const lastUpdate = updateResult({
      observation: 3,
      action: 'right',
      reward: 1,
      nextObservation: 7,
      done: true,
    });
    // The state observation differs, so a mixed mapping would be visible.
    const state = fineStep({ observation: 6 });
    expect(schematicView(state, lastUpdate)).toEqual({
      observation: 3,
      action: 'right',
      nextObservation: 7,
      reward: 1,
      fallback: true,
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
});
