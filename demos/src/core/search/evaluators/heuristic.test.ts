import { describe, expect, it } from 'vitest';
import { DRAW_SCORE, type PlayerId, WIN_SCORE_MARGIN } from '../../contracts.js';
import { type Connect4State, connect4 } from '../../games/connect4.js';
import { type TicTacToeState, ticTacToe } from '../../games/tic-tac-toe.js';
import { createRng, randomStates } from '../../testing/playouts.js';
import { connect4Heuristic, ticTacToeHeuristic } from './heuristic.js';

const CORPUS_SEED = 20261017;

function stateFromGrid(spec: readonly string[], currentPlayer: PlayerId): Connect4State {
  if (spec.length !== 6 || spec.some((line) => line.length !== 7)) {
    throw new RangeError('expected a 6x7 grid spec');
  }
  const grid = spec.map((line) =>
    Array.from(line, (char) => (char === '.' ? null : Number.parseInt(char, 10))),
  );
  return { grid, currentPlayer, lastMove: null, history: [] };
}

function mirrorConnect4(state: Connect4State): Connect4State {
  const grid = state.grid.map((line) => line.slice().reverse());
  return { grid, currentPlayer: state.currentPlayer, lastMove: null, history: [] };
}

function connect4Corpus(rngSeed: number): Connect4State[] {
  return randomStates(connect4, 30, { rng: createRng(rngSeed), minPly: 4, maxPly: 20 });
}

function ticTacToeCorpus(rngSeed: number): TicTacToeState[] {
  return randomStates(ticTacToe, 40, { rng: createRng(rngSeed), minPly: 0, maxPly: 8 });
}

describe('connect4Heuristic', () => {
  it('has id "windows"', () => {
    expect(connect4Heuristic.id).toBe('windows');
  });

  it('scores the empty board exactly 0', () => {
    const state = connect4.initialState();
    expect(connect4Heuristic.evaluate(state, { rootPlayer: 0, game: connect4 })).toBe(0);
    expect(connect4Heuristic.evaluate(state, { rootPlayer: 1, game: connect4 })).toBe(0);
  });

  it('counts an isolated corner pawn in exactly 3 windows (raw +3, scaled 3/1000)', () => {
    const state = connect4.apply(connect4.initialState(), { col: 0 });
    expect(connect4Heuristic.evaluate(state, { rootPlayer: 0, game: connect4 })).toBe(0.003);
    expect(connect4Heuristic.evaluate(state, { rootPlayer: 1, game: connect4 })).toBe(-0.003);
  });

  it('counts a bottom-center pawn in exactly 7 windows (raw +7, scaled 7/1000)', () => {
    const state = connect4.apply(connect4.initialState(), { col: 3 });
    expect(connect4Heuristic.evaluate(state, { rootPlayer: 0, game: connect4 })).toBe(0.007);
    expect(connect4Heuristic.evaluate(state, { rootPlayer: 1, game: connect4 })).toBe(-0.007);
  });

  it('counts a pawn floating mid-board in exactly 13 windows (raw +13, scaled 13/1000)', () => {
    const spec = ['.......', '.......', '...0...', '.......', '.......', '.......'];
    const state = stateFromGrid(spec, 0);
    expect(connect4Heuristic.evaluate(state, { rootPlayer: 0, game: connect4 })).toBe(0.013);
  });

  it('scores a hand-built early midgame position exactly (raw -9, scaled -9/1000)', () => {
    let state = connect4.initialState();
    for (const col of [3, 3, 2, 2, 4, 4]) {
      state = connect4.apply(state, { col });
    }
    expect(connect4Heuristic.evaluate(state, { rootPlayer: 0, game: connect4 })).toBe(-0.009);
    expect(connect4Heuristic.evaluate(state, { rootPlayer: 1, game: connect4 })).toBe(0.009);
  });

  it('is invariant under left-right mirroring', () => {
    let state = connect4.initialState();
    for (const col of [3, 3, 2, 2, 4, 4]) {
      state = connect4.apply(state, { col });
    }
    const mirrored = mirrorConnect4(state);
    expect(connect4Heuristic.evaluate(mirrored, { rootPlayer: 0, game: connect4 })).toBe(
      connect4Heuristic.evaluate(state, { rootPlayer: 0, game: connect4 }),
    );
    expect(connect4Heuristic.evaluate(mirrored, { rootPlayer: 1, game: connect4 })).toBe(
      connect4Heuristic.evaluate(state, { rootPlayer: 1, game: connect4 }),
    );
  });

  it('is exactly antisymmetric in rootPlayer over a random corpus', () => {
    for (const state of connect4Corpus(CORPUS_SEED)) {
      const asRootZero = connect4Heuristic.evaluate(state, { rootPlayer: 0, game: connect4 });
      const asRootOne = connect4Heuristic.evaluate(state, { rootPlayer: 1, game: connect4 });
      expect(asRootZero).toBe(-asRootOne);
    }
  });

  it('clamps to exactly ±0.9499 (strictly inside WIN_SCORE_MARGIN) on window-saturated positions', () => {
    const spec = ['.......', '.......', '000.000', '000.000', '000.000', '.......'];
    const dominated = stateFromGrid(spec, 0);
    expect(connect4Heuristic.evaluate(dominated, { rootPlayer: 0, game: connect4 })).toBe(0.9499);
    expect(connect4Heuristic.evaluate(dominated, { rootPlayer: 1, game: connect4 })).toBe(-0.9499);
    const score = connect4Heuristic.evaluate(dominated, { rootPlayer: 0, game: connect4 });
    expect(Math.abs(score)).toBeLessThan(WIN_SCORE_MARGIN);
  });

  it('saturates in the dominant direction on the defensive four-in-a-window case', () => {
    const spec = ['.......', '.......', '...0...', '...0...', '...0...', '...0...'];
    const won = stateFromGrid(spec, 0);
    expect(connect4Heuristic.evaluate(won, { rootPlayer: 0, game: connect4 })).toBe(0.9499);
    expect(connect4Heuristic.evaluate(won, { rootPlayer: 1, game: connect4 })).toBe(-0.9499);
    const lost = stateFromGrid(
      spec.map((line) => line.replaceAll('0', '1')),
      1,
    );
    expect(connect4Heuristic.evaluate(lost, { rootPlayer: 1, game: connect4 })).toBe(0.9499);
    expect(connect4Heuristic.evaluate(lost, { rootPlayer: 0, game: connect4 })).toBe(-0.9499);
  });

  it('stays strictly inside WIN_SCORE_MARGIN over the whole random corpus', () => {
    for (const state of connect4Corpus(CORPUS_SEED + 1)) {
      const score = connect4Heuristic.evaluate(state, { rootPlayer: 0, game: connect4 });
      expect(Math.abs(score)).toBeLessThan(WIN_SCORE_MARGIN);
    }
  });

  it('never mutates the evaluated state', () => {
    for (const state of connect4Corpus(CORPUS_SEED + 2)) {
      const snapshot = structuredClone(state);
      connect4Heuristic.evaluate(state, { rootPlayer: 0, game: connect4 });
      connect4Heuristic.evaluate(state, { rootPlayer: 1, game: connect4 });
      expect(state).toEqual(snapshot);
    }
  });
});

describe('ticTacToeHeuristic', () => {
  it('has id "zero"', () => {
    expect(ticTacToeHeuristic.id).toBe('zero');
  });

  it('always returns DRAW_SCORE regardless of the state or root player', () => {
    const states = [ticTacToe.initialState(), ...ticTacToeCorpus(CORPUS_SEED + 3)];
    for (const state of states) {
      expect(ticTacToeHeuristic.evaluate(state, { rootPlayer: 0, game: ticTacToe })).toBe(
        DRAW_SCORE,
      );
      expect(ticTacToeHeuristic.evaluate(state, { rootPlayer: 1, game: ticTacToe })).toBe(
        DRAW_SCORE,
      );
    }
  });
});
