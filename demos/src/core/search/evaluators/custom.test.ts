import { describe, expect, it } from 'vitest';
import type { Connect4State } from '../../games/connect4.js';
import { connect4 } from '../../games/connect4.js';
import type { TicTacToeState } from '../../games/tic-tac-toe.js';
import { ticTacToe } from '../../games/tic-tac-toe.js';
import { createRng, randomStates } from '../../testing/playouts.js';
import {
  buildCustomContext,
  CUSTOM_CONNECT4_SOURCE,
  CUSTOM_TIC_TAC_TOE_SOURCE,
  compileCustomEvaluator,
  defaultCustomSource,
} from './custom.js';
import { connect4Heuristic, ticTacToeHeuristic } from './heuristic.js';

const CORPUS_SEED = 20261017;
const CUSTOM_BOUND = 0.9499;

function connect4Corpus(rngSeed: number): Connect4State[] {
  return randomStates(connect4, 30, { rng: createRng(rngSeed), minPly: 4, maxPly: 20 });
}

function ticTacToeCorpus(rngSeed: number): TicTacToeState[] {
  return randomStates(ticTacToe, 40, { rng: createRng(rngSeed), minPly: 0, maxPly: 8 });
}

describe('defaultCustomSource', () => {
  it('returns the source matching the game id', () => {
    expect(defaultCustomSource('connect4')).toBe(CUSTOM_CONNECT4_SOURCE);
    expect(defaultCustomSource('tic-tac-toe')).toBe(CUSTOM_TIC_TAC_TOE_SOURCE);
  });
});

describe('buildCustomContext', () => {
  it('builds the connect4 context with 69 windows', () => {
    const ctx = buildCustomContext(connect4.initialState(), 0, 'connect4');
    expect(ctx.rootPlayer).toBe(0);
    expect(ctx.opponent).toBe(1);
    expect(ctx.rows).toBe(6);
    expect(ctx.cols).toBe(7);
    expect(ctx.windows).toHaveLength(69);
    expect(ctx.lines).toBeUndefined();
  });

  it('builds the tic-tac-toe context with 8 lines', () => {
    const ctx = buildCustomContext(ticTacToe.initialState(), 1, 'tic-tac-toe');
    expect(ctx.rootPlayer).toBe(1);
    expect(ctx.opponent).toBe(0);
    expect(ctx.rows).toBe(3);
    expect(ctx.cols).toBe(3);
    expect(ctx.lines).toHaveLength(8);
    expect(ctx.windows).toBeUndefined();
  });
});

describe('compileCustomEvaluator', () => {
  it('compiles a valid source and evaluates a sample state', () => {
    const result = compileCustomEvaluator('return 0.5;', ticTacToe, 'tic-tac-toe');
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.evaluator.id).toBe('custom');
    expect(
      result.evaluator.evaluate(ticTacToe.initialState(), { rootPlayer: 0, game: ticTacToe }),
    ).toBe(0.5);
  });

  it('rejects a source with a syntax error', () => {
    const result = compileCustomEvaluator('return (', ticTacToe, 'tic-tac-toe');
    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.error.length).toBeGreaterThan(0);
  });

  it('rejects a source that throws on the probe state', () => {
    const result = compileCustomEvaluator("throw new Error('boom');", ticTacToe, 'tic-tac-toe');
    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.error).toContain('boom');
  });

  it('rejects a source that returns a non-finite value on the probe', () => {
    expect(compileCustomEvaluator('return Number.NaN;', ticTacToe, 'tic-tac-toe').ok).toBe(false);
    expect(compileCustomEvaluator('return Infinity;', ticTacToe, 'tic-tac-toe').ok).toBe(false);
    expect(compileCustomEvaluator('return "1";', ticTacToe, 'tic-tac-toe').ok).toBe(false);
  });

  it('clamps out-of-range results to ±0.9499', () => {
    const high = compileCustomEvaluator('return 5;', ticTacToe, 'tic-tac-toe');
    expect(high.ok).toBe(true);
    if (high.ok) {
      expect(
        high.evaluator.evaluate(ticTacToe.initialState(), { rootPlayer: 0, game: ticTacToe }),
      ).toBe(CUSTOM_BOUND);
    }
    const low = compileCustomEvaluator('return -5;', ticTacToe, 'tic-tac-toe');
    expect(low.ok).toBe(true);
    if (low.ok) {
      expect(
        low.evaluator.evaluate(ticTacToe.initialState(), { rootPlayer: 0, game: ticTacToe }),
      ).toBe(-CUSTOM_BOUND);
    }
  });

  it('returns 0 on a runtime throw or a non-finite result during a real evaluation', () => {
    // La sonde (état initial, cellule 0 vide) retourne 1 ; sur un état joué,
    // la cellule 0 est occupée et le corps lève une exception.
    const result = compileCustomEvaluator(
      'return state.cells[0] === null ? 1 : null.value;',
      ticTacToe,
      'tic-tac-toe',
    );
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const played = ticTacToe.apply(ticTacToe.initialState(), { index: 0 });
    expect(result.evaluator.evaluate(played, { rootPlayer: 0, game: ticTacToe })).toBe(0);

    const nan = compileCustomEvaluator(
      'return state.cells[0] === null ? 1 : Number.NaN;',
      ticTacToe,
      'tic-tac-toe',
    );
    expect(nan.ok).toBe(true);
    if (nan.ok) {
      expect(nan.evaluator.evaluate(played, { rootPlayer: 0, game: ticTacToe })).toBe(0);
    }
  });

  it('matches connect4Heuristic over a corpus for both root players', () => {
    const result = compileCustomEvaluator(CUSTOM_CONNECT4_SOURCE, connect4, 'connect4');
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    for (const state of connect4Corpus(CORPUS_SEED)) {
      for (const rootPlayer of [0, 1] as const) {
        expect(result.evaluator.evaluate(state, { rootPlayer, game: connect4 })).toBe(
          connect4Heuristic.evaluate(state, { rootPlayer, game: connect4 }),
        );
      }
    }
  });

  it('matches ticTacToeHeuristic (zero) over a corpus', () => {
    const result = compileCustomEvaluator(CUSTOM_TIC_TAC_TOE_SOURCE, ticTacToe, 'tic-tac-toe');
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    for (const state of ticTacToeCorpus(CORPUS_SEED + 1)) {
      for (const rootPlayer of [0, 1] as const) {
        expect(result.evaluator.evaluate(state, { rootPlayer, game: ticTacToe })).toBe(
          ticTacToeHeuristic.evaluate(state, { rootPlayer, game: ticTacToe }),
        );
      }
    }
  });
});
