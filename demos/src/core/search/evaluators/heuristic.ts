/**
 * Évaluateur artisanal « à la Deep Blue » : comptage de fenêtres sur Connect4.
 *
 * Avant l'apprentissage par renforcement, la valeur d'une position s'écrivait
 * à la main. C'est le cœur de la démarche de Deep Blue (1997) : des
 * programmeurs et des grands maîtres ont codifié, fonction par fonction, ce
 * qui fait la force d'une position. Cet évaluateur reproduit cette philosophie
 * sur Connect4 avec la technique classique du comptage de fenêtres.
 *
 * Le plateau 6×7 contient exactement 69 fenêtres de 4 cases adjacentes :
 * 24 horizontales, 21 verticales, 12 diagonales ↘ et 12 diagonales ↗. Une
 * fenêtre compte pour un joueur si elle ne contient AUCUN pion adverse :
 * 3 pions à soi valent +50 (menace immédiate), 2 valent +10, 1 vaut +1, une
 * fenêtre vide ou mixte vaut 0. Le score brut est la somme des fenêtres du
 * joueur racine moins celles de son adversaire, du point de vue de
 * `ctx.rootPlayer`, divisée par WINDOW_SCORE_SCALE pour tenir dans la plage
 * heuristique imposée par le contrat.
 *
 * Les poids sont codés en dur, sous vos yeux, en constantes nommées. Ce sont
 * des choix HUMAINS, arbitraires et contestables — c'est le point narratif de
 * cette démo : ils ne sont PAS réglables depuis l'interface, et pour de
 * bonnes raisons. Régler ces nombres à la main est un travail d'expert, long
 * et fragile ; les poids « vrais » d'AlphaGo ne sont pas écrits par des
 * humains mais appris par un réseau de neurones (voir la suite de la confé-
 * rence). L'heuristique figée est ici le contrepoids qui fait sentir pourquoi
 * l'apprentissage par renforcement a changé la donne.
 *
 * Deux garde-fous :
 * - une fenêtre à 4 pions d'un même joueur est un état terminal, que le
 *   moteur ne confie jamais à l'évaluateur ; on lui attribue malgré tout une
 *   grande constante signée, purement défensive ;
 * - le score final est borné strictement à l'intérieur de
 *   ]−WIN_SCORE_MARGIN, +WIN_SCORE_MARGIN[ pour ne jamais masquer les scores
 *   exacts (±1) d'un état terminal.
 *
 * Ce fichier est écrit pour être lu pendant la démo : chaque constante porte
 * un nom, le calcul tient en quelques dizaines de lignes.
 */

import { DRAW_SCORE, type Evaluator, WIN_SCORE_MARGIN } from '../../contracts.js';
import {
  CONNECT4_COLS,
  CONNECT4_ROWS,
  type Connect4Move,
  type Connect4State,
} from '../../games/connect4.js';
import type { TicTacToeMove, TicTacToeState } from '../../games/tic-tac-toe.js';

const WINDOW_ONE = 1;
const WINDOW_TWO = 10;
const WINDOW_THREE = 50;
const WINDOW_FOUR = 10000;
const WINDOW_SCORE_SCALE = 1000;
const HEURISTIC_BOUND = WIN_SCORE_MARGIN - 0.0001;

const CONNECT4_DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

type WindowCells = ReadonlyArray<readonly [number, number]>;

function buildConnect4Windows(): ReadonlyArray<WindowCells> {
  const windows: WindowCells[] = [];
  for (const [dRow, dCol] of CONNECT4_DIRECTIONS) {
    for (let row = 0; row < CONNECT4_ROWS; row += 1) {
      for (let col = 0; col < CONNECT4_COLS; col += 1) {
        const endRow = row + dRow * 3;
        const endCol = col + dCol * 3;
        if (endRow < 0 || endRow >= CONNECT4_ROWS || endCol < 0 || endCol >= CONNECT4_COLS) {
          continue;
        }
        const cells: Array<readonly [number, number]> = [];
        for (let step = 0; step < 4; step += 1) {
          cells.push([row + dRow * step, col + dCol * step]);
        }
        windows.push(cells);
      }
    }
  }
  return windows;
}

const CONNECT4_WINDOWS: ReadonlyArray<WindowCells> = buildConnect4Windows();

function windowValue(pawnCount: number): number {
  switch (pawnCount) {
    case 4:
      return WINDOW_FOUR;
    case 3:
      return WINDOW_THREE;
    case 2:
      return WINDOW_TWO;
    case 1:
      return WINDOW_ONE;
    default:
      return 0;
  }
}

function clampHeuristicScore(score: number): number {
  return Math.min(HEURISTIC_BOUND, Math.max(-HEURISTIC_BOUND, score));
}

export const connect4Heuristic: Evaluator<Connect4State, Connect4Move> = {
  id: 'windows',
  evaluate(state, ctx) {
    const root = ctx.rootPlayer;
    const opponent = root === 0 ? 1 : 0;
    let rootTotal = 0;
    let opponentTotal = 0;
    for (const window of CONNECT4_WINDOWS) {
      let rootPawns = 0;
      let opponentPawns = 0;
      for (const [row, col] of window) {
        const cell = state.grid[row][col];
        if (cell === root) {
          rootPawns += 1;
        } else if (cell === opponent) {
          opponentPawns += 1;
        }
      }
      if (opponentPawns === 0) {
        rootTotal += windowValue(rootPawns);
      } else if (rootPawns === 0) {
        opponentTotal += windowValue(opponentPawns);
      }
    }
    return clampHeuristicScore((rootTotal - opponentTotal) / WINDOW_SCORE_SCALE);
  },
};

export const ticTacToeHeuristic: Evaluator<TicTacToeState, TicTacToeMove> = {
  id: 'zero',
  evaluate() {
    return DRAW_SCORE;
  },
};
