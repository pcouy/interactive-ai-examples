import { describe, expect, it } from 'vitest';
import type { Cell, Game, PlayerId } from '../../contracts.js';
import { type Connect4Move, type Connect4State, connect4 } from '../../games/connect4.js';
import { type TicTacToeMove, type TicTacToeState, ticTacToe } from '../../games/tic-tac-toe.js';
import { createRng, randomStates } from '../../testing/playouts.js';
import { createRandomRolloutEvaluator } from './random-rollout.js';

const CORPUS_SEED = 20261017;

interface LoopState {
  readonly turn: PlayerId;
}

const loopGame: Game<LoopState, number> = {
  winScore: 1,
  initialState: () => ({ turn: 0 }),
  legalMoves: () => [0],
  apply: (state) => ({ turn: state.turn === 0 ? 1 : 0 }),
  undo: (state) => state,
  currentPlayer: (state) => state.turn,
  winner: () => null,
  stateKey: (state) => String(state.turn),
  boardView: () => ({ rows: 1, cols: 1, cells: [[]] as Cell[][] }),
};

function connect4Corpus(rngSeed: number): Connect4State[] {
  return randomStates(connect4, 24, { rng: createRng(rngSeed), minPly: 5, maxPly: 24 });
}

function ticTacToeCorpus(rngSeed: number): TicTacToeState[] {
  return randomStates(ticTacToe, 40, { rng: createRng(rngSeed), minPly: 3, maxPly: 8 });
}

describe('createRandomRolloutEvaluator', () => {
  it('has id "random-rollout"', () => {
    const evaluator = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>();
    expect(evaluator.id).toBe('random-rollout');
  });

  it('exposes samplesPerEvaluation with the configured count and the default', () => {
    const configured = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>({
      samples: 3,
    });
    expect(configured.samplesPerEvaluation).toBe(3);
    // Défaut : DEFAULT_SAMPLES = 1 tirage par évaluation.
    const defaulted = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>();
    expect(defaulted.samplesPerEvaluation).toBe(1);
  });

  it('returns scores within [-1, 1] over random tic-tac-toe and connect-four corpora', () => {
    const ticTacToeEvaluator = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>({
      rng: createRng(11),
      samples: 3,
      maxPlies: 60,
    });
    for (const state of ticTacToeCorpus(CORPUS_SEED)) {
      const score = ticTacToeEvaluator.evaluate(state, { rootPlayer: 0, game: ticTacToe });
      expect(score).toBeGreaterThanOrEqual(-1);
      expect(score).toBeLessThanOrEqual(1);
    }
    const connect4Evaluator = createRandomRolloutEvaluator<Connect4State, Connect4Move>({
      rng: createRng(12),
      samples: 3,
      maxPlies: 60,
    });
    for (const state of connect4Corpus(CORPUS_SEED)) {
      const score = connect4Evaluator.evaluate(state, { rootPlayer: 0, game: connect4 });
      expect(score).toBeGreaterThanOrEqual(-1);
      expect(score).toBeLessThanOrEqual(1);
    }
  });

  it('is exactly antisymmetric in rootPlayer with the same seed and budget', () => {
    const makeOptions = (seed: number) => ({ rng: createRng(seed), samples: 4, maxPlies: 80 });
    for (const seed of [1, CORPUS_SEED]) {
      const asRootZero = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>(
        makeOptions(seed),
      );
      const asRootOne = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>(
        makeOptions(seed),
      );
      for (const state of ticTacToeCorpus(CORPUS_SEED + 1)) {
        const zero = asRootZero.evaluate(state, { rootPlayer: 0, game: ticTacToe });
        const one = asRootOne.evaluate(state, { rootPlayer: 1, game: ticTacToe });
        expect(zero + one).toBe(0);
      }
      const connect4AsRootZero = createRandomRolloutEvaluator<Connect4State, Connect4Move>(
        makeOptions(seed),
      );
      const connect4AsRootOne = createRandomRolloutEvaluator<Connect4State, Connect4Move>(
        makeOptions(seed),
      );
      for (const state of connect4Corpus(CORPUS_SEED + 1)) {
        const zero = connect4AsRootZero.evaluate(state, { rootPlayer: 0, game: connect4 });
        const one = connect4AsRootOne.evaluate(state, { rootPlayer: 1, game: connect4 });
        expect(zero + one).toBe(0);
      }
    }
  });

  it('is deterministic for a given seed and differs under another seed', () => {
    const makeCorpusScores = (seed: number): number[] => {
      const evaluator = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>({
        rng: createRng(seed),
        samples: 5,
        maxPlies: 60,
      });
      return ticTacToeCorpus(CORPUS_SEED + 2).map((state) =>
        evaluator.evaluate(state, { rootPlayer: 0, game: ticTacToe }),
      );
    };
    expect(makeCorpusScores(7)).toEqual(makeCorpusScores(7));
    const first = makeCorpusScores(7);
    const second = makeCorpusScores(8);
    expect(first.some((score, index) => score !== second[index])).toBe(true);
  });

  it('stays well inside [-1, 1] on the empty tic-tac-toe board over 500 samples (random play favors X)', () => {
    const evaluator = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>({
      rng: createRng(99),
      samples: 500,
      maxPlies: 9,
    });
    const score = evaluator.evaluate(ticTacToe.initialState(), { rootPlayer: 0, game: ticTacToe });
    expect(Math.abs(score)).toBeLessThan(0.5);
  });

  it('stops runaway playouts at the ply cap and contributes 0 on a non-terminating game', () => {
    const evaluator = createRandomRolloutEvaluator<LoopState, number>({
      rng: createRng(5),
      samples: 4,
      maxPlies: 5,
    });
    expect(evaluator.evaluate(loopGame.initialState(), { rootPlayer: 0, game: loopGame })).toBe(0);
  });

  it('never mutates the evaluated state', () => {
    const ticTacToeEvaluator = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>({
      rng: createRng(13),
      samples: 2,
      maxPlies: 60,
    });
    for (const state of ticTacToeCorpus(CORPUS_SEED + 5)) {
      const snapshot = structuredClone(state);
      ticTacToeEvaluator.evaluate(state, { rootPlayer: 0, game: ticTacToe });
      expect(state).toEqual(snapshot);
    }
    const connect4Evaluator = createRandomRolloutEvaluator<Connect4State, Connect4Move>({
      rng: createRng(14),
      samples: 2,
      maxPlies: 60,
    });
    for (const state of connect4Corpus(CORPUS_SEED + 6)) {
      const snapshot = structuredClone(state);
      connect4Evaluator.evaluate(state, { rootPlayer: 1, game: connect4 });
      expect(state).toEqual(snapshot);
    }
  });
});
