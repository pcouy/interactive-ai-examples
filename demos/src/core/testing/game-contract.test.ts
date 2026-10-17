import { describe, expect, it } from 'vitest';
import type { Game, PlayerId } from '../contracts.js';
import { describeGameContract } from './game-contract.js';

interface MockState {
  readonly cells: ReadonlyArray<PlayerId | null>;
  readonly current: PlayerId;
}

interface MockMove {
  readonly index: number;
}

function other(player: PlayerId): PlayerId {
  return player === 0 ? 1 : 0;
}

const mockGame: Game<MockState, MockMove> = {
  winScore: 1,
  initialState: () => ({ cells: Array.from({ length: 4 }, () => null), current: 0 }),
  legalMoves: (state) => state.cells.flatMap((cell, index) => (cell === null ? [{ index }] : [])),
  apply: (state, move) => ({
    cells: state.cells.map((cell, index) => (index === move.index ? state.current : cell)),
    current: other(state.current),
  }),
  undo: (state, move) => ({
    cells: state.cells.map((cell, index) => (index === move.index ? null : cell)),
    current: other(state.current),
  }),
  currentPlayer: (state) => state.current,
  winner: (state) => {
    const first = state.cells[0];
    if (first !== null && first === state.cells[1]) return first;
    if (state.cells.every((cell) => cell !== null)) return 'draw';
    return null;
  },
  stateKey: (state) => `${state.current}|${state.cells.map((cell) => cell ?? '-').join('')}`,
  boardView: (state) => ({ rows: 1, cols: 4, cells: [state.cells.slice()] }),
};

describeGameContract(mockGame, 'mock', { corpusSize: 25, maxPly: 4 });

describe('mock game sanity', () => {
  it('starts empty with player 0 to move', () => {
    const state = mockGame.initialState();
    expect(mockGame.currentPlayer(state)).toBe(0);
    expect(mockGame.legalMoves(state)).toHaveLength(4);
    expect(mockGame.winner(state)).toBeNull();
  });

  it('awards the win to the player owning both leading cells', () => {
    const winning: MockState = { cells: [0, 0, null, 1], current: 1 };
    expect(mockGame.winner(winning)).toBe(0);
  });
});
