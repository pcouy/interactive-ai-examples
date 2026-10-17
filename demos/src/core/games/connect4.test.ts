import { describe, expect, it } from 'vitest';
import type { Cell, CellCoord, PlayerId } from '../contracts.js';
import { describeGameContract } from '../testing/game-contract.js';
import { createRng, randomStates } from '../testing/playouts.js';
import { CONNECT4_COLS, CONNECT4_ROWS, type Connect4State, connect4 } from './connect4.js';

const CONTRACT_SEED = 20261017;
const PROPERTY_SEED = 424242;

function dropAll(cols: readonly number[]): Connect4State {
  let state = connect4.initialState();
  for (const col of cols) {
    state = connect4.apply(state, { col });
  }
  return state;
}

function stateFromColumns(
  columns: ReadonlyArray<ReadonlyArray<PlayerId>>,
  currentPlayer: PlayerId,
): Connect4State {
  if (columns.length !== CONNECT4_COLS) {
    throw new RangeError(`expected ${CONNECT4_COLS} columns, received ${columns.length}`);
  }
  const grid: Cell[][] = Array.from({ length: CONNECT4_ROWS }, () =>
    Array.from({ length: CONNECT4_COLS }, () => null),
  );
  const history: CellCoord[] = [];
  for (let col = 0; col < CONNECT4_COLS; col += 1) {
    const stack = columns[col];
    if (stack.length > CONNECT4_ROWS) {
      throw new RangeError(`column ${col} overflows the grid`);
    }
    for (let height = 0; height < stack.length; height += 1) {
      const row = CONNECT4_ROWS - 1 - height;
      grid[row][col] = stack[height];
      history.push({ row, col });
    }
  }
  const lastMove: CellCoord | null = history.length === 0 ? null : history[history.length - 1];
  return { grid, currentPlayer, lastMove, history };
}

function serializeGrid(grid: ReadonlyArray<ReadonlyArray<Cell>>): string {
  let serial = '';
  for (let row = 0; row < CONNECT4_ROWS; row += 1) {
    for (let col = 0; col < CONNECT4_COLS; col += 1) {
      const cell = grid[row][col];
      serial += cell === null ? '.' : String(cell);
    }
  }
  return serial;
}

function mirrorGrid(grid: ReadonlyArray<ReadonlyArray<Cell>>): Cell[][] {
  return grid.map((line) => line.slice().reverse());
}

function smallCorpus(maxPlies: number): Connect4State[] {
  const seen = new Set<string>();
  const states: Connect4State[] = [];
  const visit = (state: Connect4State, ply: number): void => {
    const serial = `${serializeGrid(state.grid)}|${state.currentPlayer}`;
    if (!seen.has(serial)) {
      seen.add(serial);
      states.push(state);
    }
    if (ply >= maxPlies) {
      return;
    }
    for (const move of connect4.legalMoves(state)) {
      visit(connect4.apply(state, move), ply + 1);
    }
  };
  visit(connect4.initialState(), 0);
  return states;
}

describeGameContract(connect4, 'connect4', {
  corpusSize: 60,
  maxPly: 20,
  seed: CONTRACT_SEED,
});

describe('connect4 rules', () => {
  it('starts with an empty grid, player 0 to move and no winner', () => {
    const state = connect4.initialState();
    expect(connect4.currentPlayer(state)).toBe(0);
    expect(connect4.winner(state)).toBeNull();
    expect(state.lastMove).toBeNull();
    expect(connect4.legalMoves(state)).toEqual(
      Array.from({ length: CONNECT4_COLS }, (_, col) => ({ col })),
    );
  });

  it('drops the first pawn at the bottom row of an empty column', () => {
    const state = connect4.apply(connect4.initialState(), { col: 4 });
    expect(state.grid[5][4]).toBe(0);
    for (let row = 0; row < CONNECT4_ROWS - 1; row += 1) {
      expect(state.grid[row][4]).toBeNull();
    }
    expect(state.lastMove).toEqual({ row: 5, col: 4 });
    expect(connect4.currentPlayer(state)).toBe(1);
  });

  it('stacks pawns from the bottom up in a column already played', () => {
    const state = dropAll([2, 2, 2]);
    expect(state.grid[5][2]).toBe(0);
    expect(state.grid[4][2]).toBe(1);
    expect(state.grid[3][2]).toBe(0);
    expect(state.lastMove).toEqual({ row: 3, col: 2 });
  });

  it('excludes full columns from legal moves', () => {
    const state = stateFromColumns([[1, 0, 1, 0, 1, 0], [], [], [], [], [], []], 0);
    expect(connect4.legalMoves(state)).toEqual([1, 2, 3, 4, 5, 6].map((col) => ({ col })));
  });

  it('applies moves without mutating the source state', () => {
    const state = dropAll([3, 3]);
    const snapshot = structuredClone(state);
    connect4.apply(state, { col: 0 });
    expect(state).toEqual(snapshot);
  });

  it('restores the exact parent state through repeated undo calls', () => {
    let state = dropAll([0, 3, 1, 4, 2, 5]);
    for (const col of [5, 2, 4, 1, 3, 0]) {
      state = connect4.undo(state, { col });
    }
    expect(state).toEqual(connect4.initialState());
    expect(state.lastMove).toBeNull();
    expect(connect4.currentPlayer(state)).toBe(0);
  });

  it('restores the parent lastMove exactly when undoing one move', () => {
    const parent = dropAll([2, 3, 2, 3, 0]);
    const child = connect4.apply(parent, { col: 1 });
    expect(child.lastMove).toEqual({ row: 5, col: 1 });
    expect(connect4.undo(child, { col: 1 })).toEqual(parent);
  });

  it('detects a horizontal win on the bottom row', () => {
    const state = stateFromColumns([[], [], [0], [0], [0], [0], []], 1);
    expect(connect4.winner(state)).toBe(0);
  });

  it('detects a vertical win in the rightmost column', () => {
    const state = stateFromColumns([[], [], [], [], [], [], [1, 1, 1, 1]], 0);
    expect(connect4.winner(state)).toBe(1);
  });

  it('detects a rising diagonal win reaching the top-right corner', () => {
    const state = stateFromColumns([[], [], [], [0], [1, 0], [1, 1, 0], [0, 1, 1, 0]], 0);
    expect(connect4.winner(state)).toBe(0);
  });

  it('detects a falling diagonal win on the left side', () => {
    const state = stateFromColumns([[1, 1, 0, 0], [1, 1, 0], [1, 0], [0], [], [], []], 0);
    expect(connect4.winner(state)).toBe(0);
  });

  it('reports a draw on a full grid without any alignment', () => {
    const state = stateFromColumns(
      [
        [1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
      ],
      0,
    );
    expect(connect4.winner(state)).toBe('draw');
    expect(connect4.legalMoves(state)).toHaveLength(0);
  });

  it('returns null for an unfinished position without any alignment', () => {
    expect(connect4.winner(dropAll([0, 1, 2]))).toBeNull();
  });

  it('awards the win to the player who just moved, not the one to play', () => {
    const playerZeroWins = dropAll([0, 6, 1, 6, 2, 6, 3]);
    expect(connect4.currentPlayer(playerZeroWins)).toBe(1);
    expect(connect4.winner(playerZeroWins)).toBe(0);
    const playerOneWins = dropAll([0, 6, 1, 6, 2, 6, 3, 6]);
    expect(connect4.currentPlayer(playerOneWins)).toBe(0);
    expect(connect4.winner(playerOneWins)).toBe(1);
  });
});

describe('connect4 stateKey', () => {
  it('gives the same key to a position and its left-right mirror', () => {
    const state = stateFromColumns([[], [], [1, 0], [1], [0, 1], [], []], 0);
    const mirror = stateFromColumns([[], [], [0, 1], [1], [1, 0], [], []], 0);
    expect(connect4.stateKey(state)).toBe(connect4.stateKey(mirror));
  });

  it('separates positions that are not mirrors of each other', () => {
    const left = stateFromColumns([[0], [], [], [], [], [], []], 1);
    const center = stateFromColumns([[], [], [], [0], [], [], []], 1);
    const right = stateFromColumns([[], [], [], [], [], [], [0]], 1);
    const pairWide = stateFromColumns([[0], [1], [], [], [], [], []], 0);
    const pairFar = stateFromColumns([[0], [], [1], [], [], [], []], 0);
    expect(connect4.stateKey(left)).toBe(connect4.stateKey(right));
    expect(connect4.stateKey(left)).not.toBe(connect4.stateKey(center));
    expect(connect4.stateKey(center)).not.toBe(connect4.stateKey(right));
    expect(connect4.stateKey(pairWide)).not.toBe(connect4.stateKey(pairFar));
  });

  it('includes the player to move in the key', () => {
    const columns: ReadonlyArray<ReadonlyArray<PlayerId>> = [[0], [], [], [], [], [], []];
    expect(connect4.stateKey(stateFromColumns(columns, 0))).not.toBe(
      connect4.stateKey(stateFromColumns(columns, 1)),
    );
  });

  it('agrees with an independent mirror orbit across early and random states', () => {
    const rng = createRng(PROPERTY_SEED + 1);
    const corpus = [
      ...smallCorpus(3),
      ...randomStates(connect4, 24, { rng, minPly: 4, maxPly: 20 }),
    ];
    const entries = corpus.map((state) => ({
      key: connect4.stateKey(state),
      player: state.currentPlayer,
      orbit: new Set([serializeGrid(state.grid), serializeGrid(mirrorGrid(state.grid))]),
      serial: serializeGrid(state.grid),
    }));
    const mismatches: string[] = [];
    for (const a of entries) {
      for (const b of entries) {
        const equivalent = a.player === b.player && a.orbit.has(b.serial);
        if ((a.key === b.key) !== equivalent) {
          mismatches.push(`${a.serial}/${a.player} vs ${b.serial}/${b.player}`);
        }
      }
    }
    expect(mismatches).toEqual([]);
  });
});

describe('connect4 boardView', () => {
  it('renders the grid as 6 rows x 7 columns in row order', () => {
    const view = connect4.boardView(dropAll([3, 3, 0]));
    expect(view.rows).toBe(CONNECT4_ROWS);
    expect(view.cols).toBe(CONNECT4_COLS);
    expect(view.cells).toEqual([
      [null, null, null, null, null, null, null],
      [null, null, null, null, null, null, null],
      [null, null, null, null, null, null, null],
      [null, null, null, null, null, null, null],
      [null, null, null, 1, null, null, null],
      [0, null, null, 0, null, null, null],
    ]);
  });

  it('highlights the last played cell only', () => {
    const view = connect4.boardView(dropAll([3, 3, 0]));
    expect(view.highlight).toEqual([{ row: 5, col: 0 }]);
    expect(connect4.boardView(connect4.initialState()).highlight).toBeUndefined();
  });
});
