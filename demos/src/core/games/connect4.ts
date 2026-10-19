import {
  type BoardView,
  type Cell,
  type CellCoord,
  type Game,
  type GameOutcome,
  type PlayerId,
  WIN_SCORE,
} from '../contracts.js';
import { makeBoardView } from './board-view.js';

export const CONNECT4_ROWS = 6;
export const CONNECT4_COLS = 7;

export interface Connect4State {
  readonly grid: ReadonlyArray<ReadonlyArray<Cell>>;
  readonly currentPlayer: PlayerId;
  readonly lastMove: CellCoord | null;
  readonly history: ReadonlyArray<CellCoord>;
}

export interface Connect4Move {
  readonly col: number;
}

const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

function otherPlayer(player: PlayerId): PlayerId {
  return player === 0 ? 1 : 0;
}

function requireValidColumn(col: number): void {
  if (!Number.isInteger(col) || col < 0 || col >= CONNECT4_COLS) {
    throw new RangeError(`connect4 move col out of range: ${col}`);
  }
}

function landingRow(grid: ReadonlyArray<ReadonlyArray<Cell>>, col: number): number {
  for (let row = CONNECT4_ROWS - 1; row >= 0; row -= 1) {
    if (grid[row][col] === null) {
      return row;
    }
  }
  throw new Error(`connect4 column ${col} is full and cannot receive a pawn`);
}

function topOccupiedRow(grid: ReadonlyArray<ReadonlyArray<Cell>>, col: number): number {
  for (let row = 0; row < CONNECT4_ROWS; row += 1) {
    if (grid[row][col] !== null) {
      return row;
    }
  }
  throw new Error(`connect4 column ${col} is empty and cannot be undone`);
}

function buildInitialState(): Connect4State {
  const grid: Cell[][] = Array.from({ length: CONNECT4_ROWS }, () =>
    Array.from({ length: CONNECT4_COLS }, () => null),
  );
  return { grid, currentPlayer: 0, lastMove: null, history: [] };
}

function listLegalMoves(state: Connect4State): Connect4Move[] {
  const moves: Connect4Move[] = [];
  for (let col = 0; col < CONNECT4_COLS; col += 1) {
    if (state.grid[0][col] === null) {
      moves.push({ col });
    }
  }
  return moves;
}

function applyMove(state: Connect4State, move: Connect4Move): Connect4State {
  requireValidColumn(move.col);
  const row = landingRow(state.grid, move.col);
  const grid = state.grid.map((line) => line.slice());
  grid[row][move.col] = state.currentPlayer;
  const landing: CellCoord = { row, col: move.col };
  return {
    grid,
    currentPlayer: otherPlayer(state.currentPlayer),
    lastMove: landing,
    history: [...state.history, landing],
  };
}

function undoMove(state: Connect4State, move: Connect4Move): Connect4State {
  requireValidColumn(move.col);
  const row = topOccupiedRow(state.grid, move.col);
  const grid = state.grid.map((line) => line.slice());
  grid[row][move.col] = null;
  const history = state.history.slice(0, -1);
  const lastMove: CellCoord | null = history.length === 0 ? null : history[history.length - 1];
  return { grid, currentPlayer: otherPlayer(state.currentPlayer), lastMove, history };
}

function readCurrentPlayer(state: Connect4State): PlayerId {
  return state.currentPlayer;
}

function findWinner(state: Connect4State): GameOutcome {
  for (let row = 0; row < CONNECT4_ROWS; row += 1) {
    for (let col = 0; col < CONNECT4_COLS; col += 1) {
      const pawn = state.grid[row][col];
      if (pawn === null) {
        continue;
      }
      for (const [dRow, dCol] of DIRECTIONS) {
        let aligned = true;
        for (let step = 1; step <= 3; step += 1) {
          const nextRow = row + dRow * step;
          const nextCol = col + dCol * step;
          const inGrid =
            nextRow >= 0 && nextRow < CONNECT4_ROWS && nextCol >= 0 && nextCol < CONNECT4_COLS;
          if (!inGrid || state.grid[nextRow][nextCol] !== pawn) {
            aligned = false;
            break;
          }
        }
        if (aligned) {
          return pawn;
        }
      }
    }
  }
  const full = state.grid.every((line) => line.every((cell) => cell !== null));
  return full ? 'draw' : null;
}

function serializeGrid(grid: ReadonlyArray<ReadonlyArray<Cell>>, mirrored: boolean): string {
  let serial = '';
  for (let row = 0; row < CONNECT4_ROWS; row += 1) {
    for (let col = 0; col < CONNECT4_COLS; col += 1) {
      const cell = grid[row][mirrored ? CONNECT4_COLS - 1 - col : col];
      serial += cell === null ? '.' : String(cell);
    }
  }
  return serial;
}

function canonicalKey(state: Connect4State): string {
  const normal = serializeGrid(state.grid, false);
  const mirrored = serializeGrid(state.grid, true);
  const best = mirrored < normal ? mirrored : normal;
  return `${best}|${state.currentPlayer}`;
}

function buildBoardView(state: Connect4State): BoardView {
  const highlight: CellCoord[] | undefined = state.lastMove === null ? undefined : [state.lastMove];
  return makeBoardView(CONNECT4_ROWS, CONNECT4_COLS, state.grid, { highlight });
}

// Une case non nulle correspond à un pion posé : c'est le tour intra-partie.
function countPieces(state: Connect4State): number {
  let count = 0;
  for (const line of state.grid) {
    for (const cell of line) {
      if (cell !== null) {
        count += 1;
      }
    }
  }
  return count;
}

export const connect4: Game<Connect4State, Connect4Move> = {
  winScore: WIN_SCORE,
  initialState: buildInitialState,
  legalMoves: listLegalMoves,
  apply: applyMove,
  undo: undoMove,
  currentPlayer: readCurrentPlayer,
  winner: findWinner,
  stateKey: canonicalKey,
  boardView: buildBoardView,
  pieceCount: countPieces,
};
