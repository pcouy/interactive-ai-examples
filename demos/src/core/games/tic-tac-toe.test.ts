import { describe, expect, it } from 'vitest';
import type { Cell, PlayerId } from '../contracts.js';
import { describeGameContract } from '../testing/game-contract.js';
import { createRng, randomStates } from '../testing/playouts.js';
import { type TicTacToeState, ticTacToe } from './tic-tac-toe.js';

const CONTRACT_SEED = 20261017;
const PROPERTY_SEED = 424242;

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

function applyAll(indices: readonly number[]): TicTacToeState {
  let state = ticTacToe.initialState();
  for (const index of indices) {
    state = ticTacToe.apply(state, { index });
  }
  return state;
}

function cellsWith(marks: ReadonlyArray<readonly [number, PlayerId]>): (PlayerId | null)[] {
  const cells: (PlayerId | null)[] = Array.from({ length: 9 }, () => null);
  for (const [index, player] of marks) {
    cells[index] = player;
  }
  return cells;
}

function stateWith(
  marks: ReadonlyArray<readonly [number, PlayerId]>,
  currentPlayer: PlayerId,
): TicTacToeState {
  return { cells: cellsWith(marks), currentPlayer };
}

function serializeCells(cells: ReadonlyArray<Cell>): string {
  return cells.map((cell) => (cell === null ? '-' : String(cell))).join('');
}

function orbitCells(cells: ReadonlyArray<Cell>): string[] {
  const variants: string[] = [];
  for (let quarters = 0; quarters < 4; quarters += 1) {
    for (const mirrored of [false, true]) {
      const transformed: Cell[] = Array.from({ length: 9 }, () => null);
      for (let row = 0; row < 3; row += 1) {
        for (let col = 0; col < 3; col += 1) {
          const sourceRow = [row, 2 - col, 2 - row, col][quarters];
          const sourceCol = [col, row, 2 - col, 2 - row][quarters];
          const finalCol = mirrored ? 2 - sourceCol : sourceCol;
          transformed[row * 3 + col] = cells[sourceRow * 3 + finalCol];
        }
      }
      variants.push(serializeCells(transformed));
    }
  }
  return variants;
}

function smallCorpus(maxPlies: number): TicTacToeState[] {
  const seen = new Set<string>();
  const states: TicTacToeState[] = [];
  const visit = (state: TicTacToeState, ply: number): void => {
    const serial = serializeCells(state.cells) + state.currentPlayer;
    if (!seen.has(serial)) {
      seen.add(serial);
      states.push(state);
    }
    if (ply >= maxPlies) {
      return;
    }
    for (const move of ticTacToe.legalMoves(state)) {
      visit(ticTacToe.apply(state, move), ply + 1);
    }
  };
  visit(ticTacToe.initialState(), 0);
  return states;
}

describeGameContract(ticTacToe, 'tic-tac-toe', {
  corpusSize: 80,
  maxPly: 9,
  seed: CONTRACT_SEED,
});

describe('tic-tac-toe rules', () => {
  it('starts with an empty board, player 0 to move and no winner', () => {
    const state = ticTacToe.initialState();
    expect(ticTacToe.currentPlayer(state)).toBe(0);
    expect(ticTacToe.winner(state)).toBeNull();
    expect(ticTacToe.legalMoves(state)).toEqual(
      Array.from({ length: 9 }, (_, index) => ({ index })),
    );
  });

  it('detects a win on each of the eight lines', () => {
    for (const [a, b, c] of WIN_LINES) {
      const state = stateWith(
        [
          [a, 0],
          [b, 0],
          [c, 0],
        ],
        1,
      );
      expect(ticTacToe.winner(state)).toBe(0);
    }
  });

  it('detects a win for player 1 on a diagonal', () => {
    const state = stateWith(
      [
        [0, 1],
        [4, 1],
        [8, 1],
      ],
      0,
    );
    expect(ticTacToe.winner(state)).toBe(1);
  });

  it('reports a draw once the grid is full without any alignment', () => {
    const state = stateWith(
      [
        [0, 0],
        [1, 1],
        [2, 0],
        [3, 0],
        [4, 1],
        [5, 1],
        [6, 1],
        [7, 0],
        [8, 0],
      ],
      0,
    );
    expect(ticTacToe.winner(state)).toBe('draw');
    expect(ticTacToe.legalMoves(state)).toHaveLength(0);
  });

  it('lists exactly the empty cells as legal moves, in ascending order', () => {
    const state = stateWith(
      [
        [0, 0],
        [2, 1],
        [4, 0],
        [8, 1],
      ],
      0,
    );
    expect(ticTacToe.legalMoves(state)).toEqual([1, 3, 5, 6, 7].map((index) => ({ index })));
  });

  it('matches legal moves to empty cells across random states', () => {
    const rng = createRng(PROPERTY_SEED);
    for (const state of randomStates(ticTacToe, 40, { rng, minPly: 0, maxPly: 8 })) {
      const empties = state.cells
        .map((cell, index) => (cell === null ? index : -1))
        .filter((index) => index >= 0);
      expect(ticTacToe.legalMoves(state)).toEqual(empties.map((index) => ({ index })));
    }
  });

  it('applies moves without mutating the source state', () => {
    const state = stateWith([[4, 0]], 1);
    const snapshot = structuredClone(state);
    ticTacToe.apply(state, { index: 0 });
    expect(state).toEqual(snapshot);
  });

  it('restores the exact parent state through repeated undo calls', () => {
    let state = applyAll([4, 0, 8, 2]);
    for (const index of [2, 8, 0, 4]) {
      state = ticTacToe.undo(state, { index });
    }
    expect(state).toEqual(ticTacToe.initialState());
    expect(ticTacToe.currentPlayer(state)).toBe(0);
  });

  it('awards the win to the player who just moved, not the one to play', () => {
    const playerZeroWins = applyAll([0, 3, 1, 4, 2]);
    expect(ticTacToe.currentPlayer(playerZeroWins)).toBe(1);
    expect(ticTacToe.winner(playerZeroWins)).toBe(0);
    const playerOneWins = applyAll([0, 3, 1, 4, 8, 5]);
    expect(ticTacToe.currentPlayer(playerOneWins)).toBe(0);
    expect(ticTacToe.winner(playerOneWins)).toBe(1);
  });
});

describe('tic-tac-toe stateKey', () => {
  it('gives the same key to different move orders reaching the same board', () => {
    expect(ticTacToe.stateKey(applyAll([0, 4, 8]))).toBe(ticTacToe.stateKey(applyAll([8, 4, 0])));
  });

  it('gives the same key to a played state and an equal hand-built state', () => {
    const played = applyAll([0, 4, 8]);
    const handmade = stateWith(
      [
        [0, 0],
        [4, 1],
        [8, 0],
      ],
      1,
    );
    expect(played).toEqual(handmade);
    expect(ticTacToe.stateKey(played)).toBe(ticTacToe.stateKey(handmade));
    expect(ticTacToe.stateKey(played)).toBe(ticTacToe.stateKey(played));
  });

  it('merges positions equivalent by rotation or reflection', () => {
    const single = (index: number): TicTacToeState => stateWith([[index, 0]], 1);
    const corners = [0, 2, 6, 8].map((index) => ticTacToe.stateKey(single(index)));
    expect(new Set(corners).size).toBe(1);
    const edges = [1, 3, 5, 7].map((index) => ticTacToe.stateKey(single(index)));
    expect(new Set(edges).size).toBe(1);
    const center = ticTacToe.stateKey(single(4));
    expect(center).not.toBe(corners[0]);
    expect(center).not.toBe(edges[0]);
    const diagonal = ticTacToe.stateKey(
      stateWith(
        [
          [0, 0],
          [4, 1],
        ],
        0,
      ),
    );
    expect(diagonal).toBe(
      ticTacToe.stateKey(
        stateWith(
          [
            [8, 0],
            [4, 1],
          ],
          0,
        ),
      ),
    );
    expect(diagonal).toBe(
      ticTacToe.stateKey(
        stateWith(
          [
            [2, 0],
            [4, 1],
          ],
          0,
        ),
      ),
    );
  });

  it('separates positions that no symmetry relates', () => {
    const cornerCenter = ticTacToe.stateKey(
      stateWith(
        [
          [0, 0],
          [4, 1],
        ],
        0,
      ),
    );
    const cornerNearEdge = ticTacToe.stateKey(
      stateWith(
        [
          [0, 0],
          [1, 1],
        ],
        0,
      ),
    );
    const cornerFarEdge = ticTacToe.stateKey(
      stateWith(
        [
          [0, 0],
          [2, 1],
        ],
        0,
      ),
    );
    expect(cornerCenter).not.toBe(cornerNearEdge);
    expect(cornerNearEdge).not.toBe(cornerFarEdge);
    expect(cornerCenter).not.toBe(cornerFarEdge);
  });

  it('includes the player to move in the key', () => {
    const cells = cellsWith([[4, 0]]);
    expect(ticTacToe.stateKey({ cells, currentPlayer: 0 })).not.toBe(
      ticTacToe.stateKey({ cells, currentPlayer: 1 }),
    );
  });

  it('agrees with an independent symmetry orbit across early and random states', () => {
    const rng = createRng(PROPERTY_SEED + 1);
    const corpus = [
      ...smallCorpus(3),
      ...randomStates(ticTacToe, 24, { rng, minPly: 4, maxPly: 9 }),
    ];
    const entries = corpus.map((state) => ({
      key: ticTacToe.stateKey(state),
      player: state.currentPlayer,
      orbit: new Set(orbitCells(state.cells)),
      serial: serializeCells(state.cells),
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

describe('tic-tac-toe boardView', () => {
  it('renders the cells as a 3x3 grid in row order', () => {
    const view = ticTacToe.boardView(applyAll([0, 4, 8]));
    expect(view.rows).toBe(3);
    expect(view.cols).toBe(3);
    expect(view.cells).toEqual([
      [0, null, null],
      [null, 1, null],
      [null, null, 0],
    ]);
  });
});
