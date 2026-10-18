import { customElement } from 'lit/decorators.js';
import type { Connect4Move, Connect4State } from '../../core/games/connect4.js';
import {
  CONNECT4_GAME_CONFIG,
  createGameComponent,
  type GameComponentApi,
} from '../shared/create-game-component.js';

const Connect4Base = createGameComponent(CONNECT4_GAME_CONFIG);

export type DrlConnect4Api = GameComponentApi<Connect4State, Connect4Move>;

@customElement('drl-connect4')
export class DrlConnect4 extends Connect4Base {}
