import { describe, expect, it } from 'vitest';
import {
  badgeText,
  type DemoStatusInput,
  formatPercent,
  formatSeconds,
  statusLine,
  weightsOriginText,
} from './demo-status.js';

const BASE_INPUT: DemoStatusInput = {
  state: 'deterministic',
  weightsEdited: false,
  manifestError: null,
  batchesProcessed: 0,
  loss: null,
  validationAccuracy: null,
  elapsedMs: 0,
  metricsStarted: false,
  trainExampleCount: 600,
  validationExampleCount: 100,
  parameters: 1234,
  predictedClass: null,
  inputSource: 'Dessin',
};

function input(overrides: Partial<DemoStatusInput> = {}): DemoStatusInput {
  return { ...BASE_INPUT, ...overrides };
}

const INACTIVE_STATES: readonly DemoStatusInput['state'][] = [
  'deterministic',
  'pretrained',
  'paused',
  'trained',
  'random',
];

const ALL_STATES: readonly DemoStatusInput['state'][] = [...INACTIVE_STATES, 'training'];

const STATE_BADGES: Readonly<Record<DemoStatusInput['state'], string>> = {
  deterministic: 'Poids déterministes',
  pretrained: 'Pré-entraîné',
  training: 'Entraînement',
  paused: 'En pause',
  trained: 'Entraîné',
  random: 'Poids aléatoires',
};

describe('badgeText', () => {
  it('returns « Modifié » for hand-edited weights when no run is active', () => {
    for (const state of INACTIVE_STATES) {
      expect(badgeText(input({ state, weightsEdited: true }))).toBe('Modifié');
    }
  });

  it('prefers the state badge when a run is active and the weights were edited', () => {
    expect(badgeText(input({ state: 'training', weightsEdited: true }))).toBe('Entraînement');
  });

  it('returns the state badge when the weights were not edited', () => {
    for (const state of ALL_STATES) {
      expect(badgeText(input({ state }))).toBe(STATE_BADGES[state]);
    }
  });
});

describe('weightsOriginText', () => {
  it('returns « Modifié » for hand-edited weights whatever the state', () => {
    for (const state of ALL_STATES) {
      expect(weightsOriginText(input({ state, weightsEdited: true }))).toBe('Modifié');
    }
  });

  it('returns « Aléatoires » for a random reset', () => {
    expect(weightsOriginText(input({ state: 'random' }))).toBe('Aléatoires');
  });

  it('returns « Entraîné » once a run has started', () => {
    for (const state of ['training', 'paused', 'trained'] as const) {
      expect(weightsOriginText(input({ state, metricsStarted: true }))).toBe('Entraîné');
    }
  });

  it('ignores a finished run before the first progress event', () => {
    expect(weightsOriginText(input({ state: 'trained' }))).toBe('Déterministes');
  });

  it('returns « Pré-entraîné » or « Déterministes » from the state', () => {
    expect(weightsOriginText(input({ state: 'pretrained' }))).toBe('Pré-entraîné');
    expect(weightsOriginText(input({ state: 'deterministic' }))).toBe('Déterministes');
  });
});

describe('statusLine', () => {
  it('omits the run metrics before a run', () => {
    expect(statusLine(input())).toBe(
      'Entrée : Dessin · classe prédite : – · 1234 paramètres · entraînement : 600 exemples · validation : 100 exemples',
    );
  });

  it('shows the run metrics once the run has started', () => {
    expect(
      statusLine(
        input({
          state: 'training',
          metricsStarted: true,
          batchesProcessed: 12,
          loss: 0.4567,
          validationAccuracy: 0.964,
          elapsedMs: 1500,
          predictedClass: 3,
          inputSource: 'Galerie',
          parameters: 11898,
        }),
      ),
    ).toBe(
      'Entrée : Galerie · classe prédite : 3 · 11898 paramètres · lots : 12 · perte : 0,457 · précision validation (sous-ensemble) : 96,4 % · temps : 1,5 s · entraînement : 600 exemples · validation : 100 exemples',
    );
  });

  it('shows dashes for a null loss and a null validation accuracy', () => {
    expect(statusLine(input({ state: 'training', metricsStarted: true }))).toBe(
      'Entrée : Dessin · classe prédite : – · 1234 paramètres · lots : 0 · perte : – · précision validation (sous-ensemble) : – · temps : 0,0 s · entraînement : 600 exemples · validation : 100 exemples',
    );
  });

  it('shows a dash for a null accuracy when the loss is present', () => {
    expect(
      statusLine(
        input({ state: 'training', metricsStarted: true, loss: 0.42, validationAccuracy: null }),
      ),
    ).toBe(
      'Entrée : Dessin · classe prédite : – · 1234 paramètres · lots : 0 · perte : 0,420 · précision validation (sous-ensemble) : – · temps : 0,0 s · entraînement : 600 exemples · validation : 100 exemples',
    );
  });

  it('shows a dash for a non-finite loss and a null validation accuracy', () => {
    const line = statusLine(
      input({
        state: 'training',
        metricsStarted: true,
        loss: Number.NaN,
        validationAccuracy: null,
      }),
    );
    expect(line).toContain('perte : –');
    expect(line).not.toContain('NaN');
    expect(line).toBe(
      'Entrée : Dessin · classe prédite : – · 1234 paramètres · lots : 0 · perte : – · précision validation (sous-ensemble) : – · temps : 0,0 s · entraînement : 600 exemples · validation : 100 exemples',
    );
  });

  it('shows a dash for a non-finite validation accuracy', () => {
    const nanLine = statusLine(
      input({ state: 'training', metricsStarted: true, validationAccuracy: Number.NaN }),
    );
    expect(nanLine).toContain('précision validation (sous-ensemble) : –');
    expect(nanLine).not.toContain('NaN');
    const infinityLine = statusLine(
      input({
        state: 'training',
        metricsStarted: true,
        validationAccuracy: Number.POSITIVE_INFINITY,
      }),
    );
    expect(infinityLine).toContain('précision validation (sous-ensemble) : –');
    expect(infinityLine).not.toContain('Infinity');
  });

  it('shows the metrics of a finished run', () => {
    expect(
      statusLine(
        input({
          state: 'trained',
          metricsStarted: true,
          batchesProcessed: 5,
          loss: 0.1,
          validationAccuracy: 0.9,
          elapsedMs: 2500,
        }),
      ),
    ).toBe(
      'Entrée : Dessin · classe prédite : – · 1234 paramètres · lots : 5 · perte : 0,100 · précision validation (sous-ensemble) : 90,0 % · temps : 2,5 s · entraînement : 600 exemples · validation : 100 exemples',
    );
  });

  it('shows the epoch, round and shard window of a segmented run', () => {
    const line = statusLine(
      input({
        state: 'training',
        metricsStarted: true,
        batchesProcessed: 12,
        loss: 0.4567,
        validationAccuracy: 0.964,
        elapsedMs: 1500,
        completedEpochs: 2,
        roundIndex: 1,
        roundsPerEpoch: 4,
        completedRounds: 5,
        windowShards: 8,
      }),
    );
    expect(line).toContain('époque 2');
    expect(line).toContain('round 1/4');
    expect(line).toContain('fenêtre 8 fragments');
    // Les métriques existantes restent affichées.
    expect(line).toContain('perte : 0,457');
    expect(line).toContain('précision validation (sous-ensemble) : 96,4 %');
    expect(line).toContain('temps : 1,5 s');
  });

  it('omits the segmented parts when their fields are absent or null', () => {
    const line = statusLine(
      input({
        state: 'training',
        metricsStarted: true,
        completedEpochs: null,
        roundIndex: null,
        roundsPerEpoch: null,
        completedRounds: null,
        windowShards: null,
      }),
    );
    expect(line).not.toContain('époque');
    expect(line).not.toContain('round');
    expect(line).not.toContain('fenêtre');
  });

  it('appends the hand-edited note after the example counts', () => {
    expect(statusLine(input({ state: 'pretrained', weightsEdited: true, predictedClass: 7 }))).toBe(
      'Entrée : Dessin · classe prédite : 7 · 1234 paramètres · entraînement : 600 exemples · validation : 100 exemples · poids modifiés à la main',
    );
  });

  it('appends the manifest error last', () => {
    expect(statusLine(input({ manifestError: 'HTTP 404', weightsEdited: true }))).toBe(
      'Entrée : Dessin · classe prédite : – · 1234 paramètres · entraînement : 600 exemples · validation : 100 exemples · poids modifiés à la main · manifeste inaccessible : HTTP 404',
    );
  });

  it('joins the parts with a spaced middle dot', () => {
    expect(statusLine(input()).split(' · ')).toEqual([
      'Entrée : Dessin',
      'classe prédite : –',
      '1234 paramètres',
      'entraînement : 600 exemples',
      'validation : 100 exemples',
    ]);
  });
});

describe('formatPercent', () => {
  it('formats zero, a normal value and a rounding boundary', () => {
    expect(formatPercent(0)).toBe('0,0 %');
    expect(formatPercent(96.4)).toBe('96,4 %');
    expect(formatPercent(12.25)).toBe('12,3 %');
    expect(formatPercent(12.24)).toBe('12,2 %');
  });
});

describe('formatSeconds', () => {
  it('formats zero, a normal value and a rounding boundary', () => {
    expect(formatSeconds(0)).toBe('0,0 s');
    expect(formatSeconds(1500)).toBe('1,5 s');
    expect(formatSeconds(1250)).toBe('1,3 s');
    expect(formatSeconds(999)).toBe('1,0 s');
  });

  it('formats the minute boundary', () => {
    expect(formatSeconds(60000)).toBe('60,0 s');
  });
});
