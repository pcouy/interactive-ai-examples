import type { FineStepState, FrozenLakeAction } from '../../core/frozen-lake.js';
import type { QUpdateResult, Transition } from '../../core/q-learning.js';
import type { Locale } from '../../i18n/locale.js';
import { frozenLakeMessage } from './messages.js';
import { replayMessage } from './replay-messages.js';

/** Event name the Frozen Lake demo dispatches with the current replay state. */
export const FROZEN_LAKE_REPLAY_EVENT = 'drl-replay-state';

/** Snapshot of the replay buffer and its last sampled batch. */
export interface FrozenLakeReplayState {
  readonly enabled: boolean;
  readonly batch: readonly Transition<number, FrozenLakeAction>[];
  readonly bufferSize: number;
  readonly capacity: number;
}

/** One rendered line of the replay panel. */
export interface ReplayRow {
  readonly text: string;
  readonly newest: boolean;
}

export const ACTION_GLYPHS: Record<FrozenLakeAction, string> = {
  left: '←',
  down: '↓',
  right: '→',
  up: '↑',
};

/** Header line of the replay panel. */
export function replayHeader(state: FrozenLakeReplayState, locale: Locale = 'fr'): string {
  return replayMessage(locale, 'header', {
    count: state.batch.length,
    used: state.bufferSize,
    capacity: state.capacity,
  });
}

/**
 * Replay rows, newest first: the last batch entry comes first, then the older
 * entries in their current order.
 */
export function replayRows(state: FrozenLakeReplayState, locale: Locale = 'fr'): ReplayRow[] {
  const batch = state.batch;
  if (batch.length === 0) {
    return [];
  }
  const ordered = [batch[batch.length - 1], ...batch.slice(0, -1)];
  return ordered.map((transition, position) => {
    const newest = position === 0;
    let text = replayMessage(locale, 'row', {
      from: transition.observation,
      glyph: ACTION_GLYPHS[transition.action],
      to: transition.nextObservation,
      reward: transition.reward,
    });
    if (transition.done) {
      text += ` ${replayMessage(locale, 'end')}`;
    }
    if (newest) {
      text += ` ${replayMessage(locale, 'lastStep')}`;
    }
    return { text, newest };
  });
}

/** Formatted values and formulas for the last Q-learning update. */
export interface UpdateView {
  readonly reward: string;
  readonly gamma: string;
  readonly nextValue: string;
  readonly bootstrap: string;
  readonly target: string;
  readonly previous: string;
  readonly alpha: string;
  readonly updated: string;
  readonly targetFormula: string;
  readonly updateFormula: string;
}

/** Formats a number with three decimals and a decimal point. */
function formatNumber(value: number): string {
  return value.toFixed(3);
}

/** Builds the localized labels and formulas of the update panel. */
export function updateView(
  update: QUpdateResult<number, FrozenLakeAction>,
  locale: Locale = 'fr',
): UpdateView {
  const reward = formatNumber(update.transition.reward);
  const gamma = formatNumber(update.discountFactor);
  const nextValue = formatNumber(update.nextValue);
  const bootstrap = formatNumber(update.bootstrap);
  const target = formatNumber(update.target);
  const previous = formatNumber(update.previousValue);
  const alpha = formatNumber(update.learningRate);
  const updated = formatNumber(update.updatedValue);
  return {
    reward,
    gamma,
    nextValue,
    bootstrap,
    target,
    previous,
    alpha,
    updated,
    targetFormula: frozenLakeMessage(locale, 'targetFormula', {
      reward,
      gamma,
      nextValue,
      target,
    }),
    updateFormula: frozenLakeMessage(locale, 'updateFormula', {
      previous,
      alpha,
      target,
      updated,
    }),
  };
}

/** Values shown on the fine-step schematic. */
export interface SchematicView {
  readonly observation: number | null;
  readonly action: FrozenLakeAction | null;
  readonly nextObservation: number | null;
  readonly reward: number | null;
}

/**
 * Builds the schematic of the current fine step. The observe phase shows the
 * current observation only. Every other field stays empty until the act phase.
 */
export function schematicView(state: FineStepState): SchematicView {
  if (state.action !== null) {
    return {
      observation: state.observation,
      action: state.action,
      nextObservation: state.nextObservation,
      reward: state.reward,
    };
  }
  return {
    observation: state.observation,
    action: null,
    nextObservation: null,
    reward: null,
  };
}
