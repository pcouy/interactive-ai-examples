import {
  type BoardView,
  type Cell,
  type Game,
  type GameOutcome,
  type PlayerId,
  WIN_SCORE,
} from '../contracts.js';
import { makeBoardView } from './board-view.js';

export interface TicTacToeState {
  readonly cells: ReadonlyArray<PlayerId | null>;
  readonly currentPlayer: PlayerId;
}

export interface TicTacToeMove {
  readonly index: number;
}

const WIN_LINES: ReadonlyArray<readonly [number, number, number]> = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
];

const SYMMETRIES: ReadonlyArray<readonly number[]> = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8],
  [2, 5, 8, 1, 4, 7, 0, 3, 6],
  [8, 7, 6, 5, 4, 3, 2, 1, 0],
  [6, 3, 0, 7, 4, 1, 8, 5, 2],
  [2, 1, 0, 5, 4, 3, 8, 7, 6],
  [6, 7, 8, 3, 4, 5, 0, 1, 2],
  [0, 3, 6, 1, 4, 7, 2, 5, 8],
  [8, 5, 2, 7, 4, 1, 6, 3, 0],
];

function otherPlayer(player: PlayerId): PlayerId {
  return player === 0 ? 1 : 0;
}

function requireInRange(index: number): void {
  if (!Number.isInteger(index) || index < 0 || index > 8) {
    throw new RangeError(`tic-tac-toe move index out of range: ${index}`);
  }
}

function requireEmptyCell(state: TicTacToeState, index: number): void {
  requireInRange(index);
  if (state.cells[index] !== null) {
    throw new Error(`tic-tac-toe cell ${index} is already occupied`);
  }
}

function requireOccupiedCell(state: TicTacToeState, index: number): void {
  requireInRange(index);
  if (state.cells[index] === null) {
    throw new Error(`tic-tac-toe cell ${index} is empty and cannot be undone`);
  }
}

function buildInitialState(): TicTacToeState {
  return { cells: Array.from({ length: 9 }, () => null), currentPlayer: 0 };
}

function listLegalMoves(state: TicTacToeState): TicTacToeMove[] {
  const moves: TicTacToeMove[] = [];
  for (let index = 0; index < 9; index += 1) {
    if (state.cells[index] === null) {
      moves.push({ index });
    }
  }
  return moves;
}

function applyMove(state: TicTacToeState, move: TicTacToeMove): TicTacToeState {
  requireEmptyCell(state, move.index);
  const cells = state.cells.slice();
  cells[move.index] = state.currentPlayer;
  return { cells, currentPlayer: otherPlayer(state.currentPlayer) };
}

function undoMove(state: TicTacToeState, move: TicTacToeMove): TicTacToeState {
  requireOccupiedCell(state, move.index);
  const cells = state.cells.slice();
  cells[move.index] = null;
  return { cells, currentPlayer: otherPlayer(state.currentPlayer) };
}

function readCurrentPlayer(state: TicTacToeState): PlayerId {
  return state.currentPlayer;
}

function findWinner(state: TicTacToeState): GameOutcome {
  for (const [a, b, c] of WIN_LINES) {
    const cell = state.cells[a];
    if (cell !== null && cell === state.cells[b] && cell === state.cells[c]) {
      return cell;
    }
  }
  return state.cells.every((cell) => cell !== null) ? 'draw' : null;
}

function cellMasks(cells: ReadonlyArray<Cell>): [number, number] {
  let mask0 = 0;
  let mask1 = 0;
  for (let index = 0; index < 9; index += 1) {
    const cell = cells[index];
    if (cell === 0) {
      mask0 |= 1 << index;
    } else if (cell === 1) {
      mask1 |= 1 << index;
    }
  }
  return [mask0, mask1];
}

function toBits(mask: number): string {
  return mask.toString(2).padStart(9, '0');
}

function canonicalKey(state: TicTacToeState): string {
  const [mask0, mask1] = cellMasks(state.cells);
  let best: string | null = null;
  for (const symmetry of SYMMETRIES) {
    let movedMask0 = 0;
    let movedMask1 = 0;
    for (let index = 0; index < 9; index += 1) {
      const source = symmetry[index];
      if ((mask0 & (1 << source)) !== 0) {
        movedMask0 |= 1 << index;
      }
      if ((mask1 & (1 << source)) !== 0) {
        movedMask1 |= 1 << index;
      }
    }
    const variant = toBits(movedMask0) + toBits(movedMask1);
    if (best === null || variant < best) {
      best = variant;
    }
  }
  return `${best}|${state.currentPlayer}`;
}

function buildBoardView(state: TicTacToeState): BoardView {
  const rows: Cell[][] = [];
  for (let row = 0; row < 3; row += 1) {
    rows.push(state.cells.slice(row * 3, row * 3 + 3));
  }
  return makeBoardView(3, 3, rows);
}

export const ticTacToe: Game<TicTacToeState, TicTacToeMove> = {
  winScore: WIN_SCORE,
  initialState: buildInitialState,
  legalMoves: listLegalMoves,
  apply: applyMove,
  undo: undoMove,
  currentPlayer: readCurrentPlayer,
  winner: findWinner,
  stateKey: canonicalKey,
  boardView: buildBoardView,
};
