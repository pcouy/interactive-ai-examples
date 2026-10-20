/**
 * Évaluateur personnalisable : le corps d'une fonction `evaluate(state, ctx)`
 * écrit à la main est compilé une seule fois puis réutilisé par la recherche.
 *
 * Le corps est compilé avec `new Function('state', 'ctx', '"use strict";\n' +
 * source)`. Le contexte `ctx` expose, en plus de `rootPlayer` et `game` :
 * `opponent`, `rows`, `cols`, puis `lines` (morpion) ou `windows` (puissance 4).
 *
 * Deux garde-fous :
 * - la compilation et une sonde sur `game.initialState()` (pour `rootPlayer`
 *   0 et 1) rejettent toute source invalide avant qu'elle n'entre en recherche ;
 * - chaque évaluation borne le résultat à ±0.9499 et transforme un résultat
 *   non fini en 0. Une exception levée pendant la recherche retombe sur 0 plutôt
 *   que de faire échouer la recherche entière.
 *
 * Ce module ne touche pas au DOM : il ne sert que le moteur de recherche.
 */

import type { Evaluator, Game, PlayerId } from '../../contracts.js';
import { CONNECT4_COLS, CONNECT4_ROWS, connect4 } from '../../games/connect4.js';
import { ticTacToe } from '../../games/tic-tac-toe.js';

export type CustomGameId = 'tic-tac-toe' | 'connect4';

/** Corps par défaut du puissance 4 : comptage des fenêtres de 4 cases. */
export const CUSTOM_CONNECT4_SOURCE = `// Défaut « windows » du Puissance 4 : comptage des fenêtres de 4 cases.
const ONE = 1, TWO = 10, THREE = 50, FOUR = 10000, SCALE = 1000;
const root = ctx.rootPlayer;
const opp = ctx.opponent;
let rootTotal = 0;
let oppTotal = 0;
for (const window of ctx.windows) {
  let r = 0;
  let o = 0;
  for (const [row, col] of window) {
    const cell = state.grid[row][col];
    if (cell === root) r += 1;
    else if (cell === opp) o += 1;
  }
  const value = (n) => n === 4 ? FOUR : n === 3 ? THREE : n === 2 ? TWO : n === 1 ? ONE : 0;
  if (o === 0) rootTotal += value(r);
  else if (r === 0) oppTotal += value(o);
}
const raw = (rootTotal - oppTotal) / SCALE;
const BOUND = 0.9499;
return Math.min(BOUND, Math.max(-BOUND, raw));
`;

/** Corps par défaut du morpion : le moteur résout la partie, l'heuristique est nulle. */
export const CUSTOM_TIC_TAC_TOE_SOURCE = `// Défaut « windows » du Morpion : le moteur résout la partie au terminal,
// donc l'évaluateur windows est l'heuristique nulle (DRAW_SCORE = 0).
return 0;
`;

/** Source par défaut d'un jeu donné. */
export function defaultCustomSource(gameId: CustomGameId): string {
  return gameId === 'connect4' ? CUSTOM_CONNECT4_SOURCE : CUSTOM_TIC_TAC_TOE_SOURCE;
}

type WindowCells = ReadonlyArray<readonly [number, number]>;

export interface CustomContext {
  readonly rootPlayer: PlayerId;
  readonly opponent: PlayerId;
  readonly rows: number;
  readonly cols: number;
  readonly game: Game<unknown, unknown>;
  readonly lines?: ReadonlyArray<readonly [number, number, number]>;
  readonly windows?: ReadonlyArray<WindowCells>;
}

const CONNECT4_DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

const TIC_TAC_TOE_LINES: ReadonlyArray<readonly [number, number, number]> = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
];

const TIC_TAC_TOE_SIZE = 3;

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

function gameForId(gameId: CustomGameId): Game<unknown, unknown> {
  return gameId === 'connect4' ? connect4 : ticTacToe;
}

/**
 * Construit le contexte passé au corps personnalisé. Les dimensions viennent
 * des constantes du jeu ; `state` n'intervient pas car le contexte est le
 * même pour tout état d'un jeu donné.
 */
export function buildCustomContext(
  _state: unknown,
  rootPlayer: PlayerId,
  gameId: CustomGameId,
): CustomContext {
  const opponent = rootPlayer === 0 ? 1 : 0;
  if (gameId === 'connect4') {
    return {
      rootPlayer,
      opponent,
      rows: CONNECT4_ROWS,
      cols: CONNECT4_COLS,
      game: gameForId(gameId),
      windows: CONNECT4_WINDOWS,
    };
  }
  return {
    rootPlayer,
    opponent,
    rows: TIC_TAC_TOE_SIZE,
    cols: TIC_TAC_TOE_SIZE,
    game: gameForId(gameId),
    lines: TIC_TAC_TOE_LINES,
  };
}

const CUSTOM_BOUND = 0.9499;

function clampCustomScore(value: number): number {
  return Math.min(CUSTOM_BOUND, Math.max(-CUSTOM_BOUND, value));
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export type CompileCustomResult =
  | { readonly ok: true; readonly evaluator: Evaluator<unknown, unknown> }
  | { readonly ok: false; readonly error: string };

/**
 * Compile le corps `source` en un évaluateur `{ id: 'custom' }`. La fonction
 * est compilée une fois ici, jamais à chaque évaluation. La sonde sur l'état
 * initial rejette une source qui lève ou qui ne retourne pas un nombre fini.
 */
export function compileCustomEvaluator(
  source: string,
  game: Game<unknown, unknown>,
  gameId: CustomGameId,
): CompileCustomResult {
  let body: (state: unknown, ctx: CustomContext) => unknown;
  try {
    body = new Function('state', 'ctx', `"use strict";\n${source}`) as (
      state: unknown,
      ctx: CustomContext,
    ) => unknown;
  } catch (error) {
    return { ok: false, error: errorMessage(error) };
  }

  const initial = game.initialState();
  for (const rootPlayer of [0, 1] as const) {
    let value: unknown;
    try {
      value = body(initial, buildCustomContext(initial, rootPlayer, gameId));
    } catch (error) {
      return { ok: false, error: errorMessage(error) };
    }
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return { ok: false, error: 'the evaluator must return a finite number' };
    }
  }

  return {
    ok: true,
    evaluator: {
      id: 'custom',
      evaluate(state, ctx) {
        try {
          const value = body(state, buildCustomContext(state, ctx.rootPlayer, gameId));
          if (typeof value !== 'number' || !Number.isFinite(value)) {
            return 0;
          }
          return clampCustomScore(value);
        } catch {
          // Exception pendant une recherche réelle : on retombe sur 0 pour ne
          // pas interrompre la recherche (le corps est du code fourni par
          // l'utilisateur, il n'est pas garanti sûr à l'exécution).
          return 0;
        }
      },
    },
  };
}
