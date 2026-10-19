import './ui/drl-controls.js';
import './ui/drl-slider.js';
import './ui/drl-card.js';

import './components/drl-perceptron/drl-perceptron.js';

import './components/drl-gradient-descent/drl-gradient-descent.js';

import './components/drl-tic-tac-toe/drl-tic-tac-toe.js';

import './components/drl-connect4/drl-connect4.js';

import './components/drl-deep-blue-page/drl-deep-blue-page.js';

import './components/drl-frozen-lake/drl-frozen-lake.js';

// @demos-imports

export * from './base/demo-element.js';
export type {
  FrozenLakeAction,
  FrozenLakeCell,
  FrozenLakeCoreOptions,
  FrozenLakeEnvOptions,
  FrozenLakeSize,
  FrozenLakeStats,
  FrozenLakeStep,
} from './core/frozen-lake.js';
export { FROZEN_LAKE_ACTIONS, FrozenLakeCore, FrozenLakeEnv } from './core/frozen-lake.js';
export type { QLearningOptions, Transition } from './core/q-learning.js';
export { QLearningAgent } from './core/q-learning.js';
export type { ReplayBufferOptions } from './core/replay-buffer.js';
export { ReplayBuffer } from './core/replay-buffer.js';
export { DrlCard } from './ui/drl-card.js';
export { DrlControls } from './ui/drl-controls.js';
export { DrlSlider } from './ui/drl-slider.js';
