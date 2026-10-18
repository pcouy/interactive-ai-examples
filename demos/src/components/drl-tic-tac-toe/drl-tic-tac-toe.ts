import { customElement } from 'lit/decorators.js';
import type { TicTacToeMove, TicTacToeState } from '../../core/games/tic-tac-toe.js';
import {
  createGameComponent,
  type GameComponentApi,
  TIC_TAC_TOE_GAME_CONFIG,
} from '../shared/create-game-component.js';

const TicTacToeBase = createGameComponent(TIC_TAC_TOE_GAME_CONFIG);

export type DrlTicTacToeApi = GameComponentApi<TicTacToeState, TicTacToeMove>;

@customElement('drl-tic-tac-toe')
export class DrlTicTacToe extends TicTacToeBase {}
