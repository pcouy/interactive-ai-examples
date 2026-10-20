import { describe, expect, it } from 'vitest';
import {
  initialTrainingState,
  isTrainingActive,
  onComponentDisconnected,
  onFiltersChanged,
  onManifestFailed,
  onManifestLoaded,
  onPauseRequested,
  onProgressReceived,
  onRunCancelled,
  onRunCompleted,
  onRunStarted,
  onWeightsEdited,
  onWeightsReset,
  type TrainingState,
  trainingStateBadge,
} from './training-state.js';

const ALL_STATES: readonly TrainingState[] = [
  'deterministic',
  'pretrained',
  'training',
  'paused',
  'trained',
  'random',
];

const INACTIVE_STATES: readonly TrainingState[] = [
  'deterministic',
  'pretrained',
  'paused',
  'trained',
  'random',
];

describe('trainingStateBadge', () => {
  it('maps every state to its French badge', () => {
    expect(trainingStateBadge('deterministic')).toBe('Poids déterministes');
    expect(trainingStateBadge('pretrained')).toBe('Pré-entraîné');
    expect(trainingStateBadge('training')).toBe('Entraînement');
    expect(trainingStateBadge('paused')).toBe('En pause');
    expect(trainingStateBadge('trained')).toBe('Entraîné');
    expect(trainingStateBadge('random')).toBe('Poids aléatoires');
  });
});

describe('initialTrainingState', () => {
  it('is pretrained when the manifest matches and deterministic otherwise', () => {
    expect(initialTrainingState(true)).toBe('pretrained');
    expect(initialTrainingState(false)).toBe('deterministic');
  });
});

describe('onRunStarted', () => {
  it('moves every state to training', () => {
    for (const state of ALL_STATES) {
      expect(onRunStarted(state)).toBe('training');
    }
  });
});

describe('onProgressReceived', () => {
  it('never changes the state', () => {
    for (const state of ALL_STATES) {
      expect(onProgressReceived(state)).toBe(state);
    }
  });
});

describe('onPauseRequested', () => {
  it('moves training to paused', () => {
    expect(onPauseRequested('training')).toBe('paused');
  });

  it('leaves inactive states unchanged', () => {
    for (const state of INACTIVE_STATES) {
      expect(onPauseRequested(state)).toBe(state);
    }
  });
});

describe('onRunCompleted', () => {
  it('moves training to trained', () => {
    expect(onRunCompleted('training')).toBe('trained');
  });

  it('leaves inactive states unchanged', () => {
    for (const state of INACTIVE_STATES) {
      expect(onRunCompleted(state)).toBe(state);
    }
  });
});

describe('onRunCancelled', () => {
  it('moves training to paused', () => {
    expect(onRunCancelled('training')).toBe('paused');
  });

  it('leaves inactive states unchanged', () => {
    for (const state of INACTIVE_STATES) {
      expect(onRunCancelled(state)).toBe(state);
    }
  });
});

describe('onWeightsReset', () => {
  it('returns to the initial state for every current state', () => {
    for (const state of ALL_STATES) {
      expect(onWeightsReset(state, true)).toBe('pretrained');
      expect(onWeightsReset(state, false)).toBe('deterministic');
    }
  });

  it('returns random for every current state with a random origin', () => {
    for (const state of ALL_STATES) {
      expect(onWeightsReset(state, true, 'random')).toBe('random');
      expect(onWeightsReset(state, false, 'random')).toBe('random');
    }
  });
});

describe('onWeightsEdited', () => {
  it('never returns training and returns the initial state for every current state', () => {
    for (const state of ALL_STATES) {
      expect(onWeightsEdited(state, true)).toBe('pretrained');
      expect(onWeightsEdited(state, false)).toBe('deterministic');
    }
  });
});

describe('onFiltersChanged', () => {
  it('returns to the initial state for every current state', () => {
    for (const state of ALL_STATES) {
      expect(onFiltersChanged(state, true)).toBe('pretrained');
      expect(onFiltersChanged(state, false)).toBe('deterministic');
    }
  });
});

describe('onManifestLoaded', () => {
  it('moves idle states to pretrained when the manifest matches', () => {
    expect(onManifestLoaded('deterministic', true)).toBe('pretrained');
    expect(onManifestLoaded('pretrained', true)).toBe('pretrained');
  });

  it('moves idle states to deterministic when the manifest does not match', () => {
    expect(onManifestLoaded('pretrained', false)).toBe('deterministic');
  });

  it('never disturbs an active or finished run', () => {
    for (const state of ['training', 'paused', 'trained'] as const) {
      expect(onManifestLoaded(state, true)).toBe(state);
      expect(onManifestLoaded(state, false)).toBe(state);
    }
  });

  it('never disturbs a random reset', () => {
    expect(onManifestLoaded('random', true)).toBe('random');
    expect(onManifestLoaded('random', false)).toBe('random');
  });
});

describe('onManifestFailed', () => {
  it('moves idle states to deterministic', () => {
    expect(onManifestFailed('deterministic')).toBe('deterministic');
    expect(onManifestFailed('pretrained')).toBe('deterministic');
  });

  it('never disturbs an active or finished run', () => {
    for (const state of ['training', 'paused', 'trained'] as const) {
      expect(onManifestFailed(state)).toBe(state);
    }
  });

  it('never disturbs a random reset', () => {
    expect(onManifestFailed('random')).toBe('random');
  });
});

describe('onComponentDisconnected', () => {
  it('moves training to paused', () => {
    expect(onComponentDisconnected('training')).toBe('paused');
  });

  it('leaves inactive states unchanged', () => {
    for (const state of INACTIVE_STATES) {
      expect(onComponentDisconnected(state)).toBe(state);
    }
  });
});

describe('inactive-run guarantee', () => {
  it('no transition leaves training when no run is active', () => {
    const endOfRun: ReadonlyArray<(state: TrainingState) => TrainingState> = [
      onPauseRequested,
      onRunCancelled,
      (state) => onWeightsReset(state, true),
      (state) => onWeightsEdited(state, true),
      (state) => onFiltersChanged(state, true),
      onComponentDisconnected,
    ];
    for (const transition of endOfRun) {
      expect(transition('training')).not.toBe('training');
    }
  });

  it('isTrainingActive is true only for training', () => {
    expect(isTrainingActive('training')).toBe(true);
    expect(isTrainingActive('random')).toBe(false);
    for (const state of INACTIVE_STATES) {
      expect(isTrainingActive(state)).toBe(false);
    }
  });
});
