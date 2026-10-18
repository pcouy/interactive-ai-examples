import { describe, expect, it } from 'vitest';
import {
  DRAW_SCORE,
  LOSS_SCORE,
  type PlayerId,
  type SearchAgent,
  type TreeRecorder,
  type ValueEvent,
  type VisitEvent,
  WIN_SCORE,
} from '../contracts.js';
import { type TicTacToeMove, type TicTacToeState, ticTacToe } from '../games/tic-tac-toe.js';
import { createRng, randomStates } from '../testing/playouts.js';
import { ticTacToeHeuristic } from './evaluators/heuristic.js';
import { createMinimaxAgent } from './minimax.js';
import {
  createTranspositionTable,
  isEntryUsable,
  type TranspositionEntry,
  type TranspositionTable,
} from './transposition.js';

const CORPUS_SEED = 20261018;
const STEP_BUDGET = 2_000_000;

interface RecordedSearch<S, M> {
  readonly agent: SearchAgent<S, M>;
  readonly visits: VisitEvent<S>[];
  readonly values: ValueEvent[];
}

function recordSearch<S, M>(
  build: (recorder: TreeRecorder<S>) => SearchAgent<S, M>,
  state: S,
): RecordedSearch<S, M> {
  const visits: VisitEvent<S>[] = [];
  const values: ValueEvent[] = [];
  const recorder: TreeRecorder<S> = {
    onVisit: (event) => visits.push(event),
    onValue: (event) => values.push(event),
  };
  const agent = build(recorder);
  agent.start(state);
  for (let budget = STEP_BUDGET; budget > 0; budget -= 1) {
    if (agent.step() === 'done') {
      break;
    }
  }
  return { agent, visits, values };
}

function finalRootScore(values: ValueEvent[], rootKey: string): number {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index];
    if (value.nodeKey === rootKey && value.exact) {
      return value.score;
    }
  }
  throw new Error('expected a final exact root value');
}

function naiveMinimax(state: TicTacToeState, depth: number, rootPlayer: PlayerId): number {
  const outcome = ticTacToe.winner(state);
  if (outcome !== null) {
    return outcome === 'draw' ? DRAW_SCORE : outcome === rootPlayer ? WIN_SCORE : LOSS_SCORE;
  }
  if (depth === 0) {
    return ticTacToeHeuristic.evaluate(state, { rootPlayer, game: ticTacToe });
  }
  const maximizing = ticTacToe.currentPlayer(state) === rootPlayer;
  let best = maximizing ? -Infinity : Infinity;
  for (const move of ticTacToe.legalMoves(state)) {
    const value = naiveMinimax(ticTacToe.apply(state, move), depth - 1, rootPlayer);
    best = maximizing ? Math.max(best, value) : Math.min(best, value);
  }
  return best;
}

function createTicTacToeAgent(
  recorder: TreeRecorder<TicTacToeState>,
  depth: number,
  prune: boolean,
  transposition?: TranspositionTable<TicTacToeMove>,
): SearchAgent<TicTacToeState, TicTacToeMove> {
  return createMinimaxAgent(ticTacToe, {
    depth,
    evaluator: ticTacToeHeuristic,
    prune,
    recorder,
    transposition,
  });
}

function midgameState(): TicTacToeState {
  return [4, 0, 2, 6].reduce(
    (state, index) => ticTacToe.apply(state, { index }),
    ticTacToe.initialState(),
  );
}

describe('createTranspositionTable', () => {
  it('stores, probes, counts and clears entries', () => {
    const table = createTranspositionTable<TicTacToeMove>();
    expect(table.size()).toBe(0);
    expect(table.probe('missing')).toBeNull();
    const entry: TranspositionEntry<TicTacToeMove> = {
      score: 0.25,
      depth: 3,
      exact: true,
      bestMove: { index: 4 },
    };
    table.store('a', entry);
    expect(table.probe('a')).toEqual(entry);
    table.store('b', { score: -1, depth: 0, exact: false, bestMove: null });
    expect(table.size()).toBe(2);
    expect(table.probe('b')?.score).toBe(-1);
    table.clear();
    expect(table.size()).toBe(0);
    expect(table.probe('a')).toBeNull();
    expect(table.probe('b')).toBeNull();
  });

  it('keeps the deeper entry when storing over an existing one', () => {
    const table = createTranspositionTable<TicTacToeMove>();
    table.store('k', { score: 1, depth: 2, exact: true, bestMove: null });
    table.store('k', { score: 0, depth: 1, exact: true, bestMove: null });
    expect(table.probe('k')?.score).toBe(1);
    table.store('k', { score: 0.5, depth: 2, exact: false, bestMove: null });
    expect(table.probe('k')?.score).toBe(0.5);
    table.store('k', { score: 0.75, depth: 5, exact: false, bestMove: null });
    expect(table.probe('k')?.score).toBe(0.75);
  });
});

describe('isEntryUsable', () => {
  it('rejects an entry shallower than the remaining search depth', () => {
    const entry: TranspositionEntry<TicTacToeMove> = {
      score: 0.5,
      depth: 1,
      exact: true,
      bestMove: null,
    };
    expect(isEntryUsable(entry, 0, -Infinity, Infinity, true)).toBe(true);
    expect(isEntryUsable(entry, 1, -Infinity, Infinity, true)).toBe(true);
    expect(isEntryUsable(entry, 2, -Infinity, Infinity, true)).toBe(false);
  });

  it('accepts an exact entry whatever the window', () => {
    const entry: TranspositionEntry<TicTacToeMove> = {
      score: 0.5,
      depth: 3,
      exact: true,
      bestMove: null,
    };
    expect(isEntryUsable(entry, 3, -Infinity, Infinity, true)).toBe(true);
    expect(isEntryUsable(entry, 2, -0.9, 0.1, true)).toBe(true);
    expect(isEntryUsable(entry, 2, 0.6, Infinity, false)).toBe(true);
  });

  it('accepts a bound entry only when the window is compatible with the node direction', () => {
    const lower: TranspositionEntry<TicTacToeMove> = {
      score: 0.3,
      depth: 3,
      exact: false,
      bestMove: null,
    };
    expect(isEntryUsable(lower, 3, -Infinity, 0.2, true)).toBe(true);
    expect(isEntryUsable(lower, 3, -Infinity, 0.3, true)).toBe(true);
    expect(isEntryUsable(lower, 3, -Infinity, 0.4, true)).toBe(false);
    const upper: TranspositionEntry<TicTacToeMove> = {
      score: -0.3,
      depth: 3,
      exact: false,
      bestMove: null,
    };
    expect(isEntryUsable(upper, 3, -0.2, Infinity, false)).toBe(true);
    expect(isEntryUsable(upper, 3, -0.3, Infinity, false)).toBe(true);
    expect(isEntryUsable(upper, 3, -0.4, Infinity, false)).toBe(false);
  });
});

describe('minimax with a transposition table', () => {
  it('visits strictly fewer nodes with ttHits > 0 on transposition-heavy positions and keeps the same score and bestMove', () => {
    for (const state of [ticTacToe.initialState(), midgameState()]) {
      const rootKey = ticTacToe.stateKey(state);
      const off = recordSearch((recorder) => createTicTacToeAgent(recorder, 5, true), state);
      const on = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 5, true, createTranspositionTable()),
        state,
      );
      const offStats = off.agent.stats();
      const onStats = on.agent.stats();
      expect(onStats.nodesVisited).toBeLessThan(offStats.nodesVisited);
      expect(onStats.ttHits).toBeGreaterThan(0);
      expect(onStats.ttMisses).toBeGreaterThan(0);
      expect(onStats.elapsedSteps).toBe(onStats.nodesVisited);
      expect(finalRootScore(on.values, rootKey)).toBe(finalRootScore(off.values, rootKey));
      expect(on.agent.bestMove()).toEqual(off.agent.bestMove());
    }
  });

  it('is correct with and without pruning and matches the naive score', () => {
    const rootPlayer = ticTacToe.currentPlayer(ticTacToe.initialState());
    const expected = naiveMinimax(ticTacToe.initialState(), 5, rootPlayer);
    for (const prune of [false, true]) {
      const off = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 5, prune),
        ticTacToe.initialState(),
      );
      const on = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 5, prune, createTranspositionTable()),
        ticTacToe.initialState(),
      );
      const rootKey = ticTacToe.stateKey(ticTacToe.initialState());
      expect(finalRootScore(on.values, rootKey)).toBe(expected);
      expect(finalRootScore(off.values, rootKey)).toBe(expected);
      expect(on.agent.bestMove()).toEqual(off.agent.bestMove());
      expect(on.agent.stats().nodesVisited).toBeLessThan(off.agent.stats().nodesVisited);
      expect(on.agent.stats().ttHits).toBeGreaterThan(0);
    }
  });

  it('misses entries from a shallower search when searching deeper, then hits when searching shallower again', () => {
    const shared = createTranspositionTable<TicTacToeMove>();
    recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 2, true, shared),
      ticTacToe.initialState(),
    );
    const deep = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true, shared),
      ticTacToe.initialState(),
    );
    const rootKey = ticTacToe.stateKey(ticTacToe.initialState());
    const deepStats = deep.agent.stats();
    expect(deepStats.ttMisses).toBeGreaterThan(0);
    expect(deepStats.ttHits).toBeGreaterThan(0);
    expect(finalRootScore(deep.values, rootKey)).toBe(naiveMinimax(ticTacToe.initialState(), 4, 0));
    expect(ticTacToe.legalMoves(ticTacToe.initialState())).toContainEqual(deep.agent.bestMove());
    const again = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 2, true, shared),
      ticTacToe.initialState(),
    );
    const againStats = again.agent.stats();
    expect(againStats.nodesVisited).toBe(1);
    expect(againStats.ttHits).toBe(1);
    expect(again.agent.bestMove()).toEqual(deep.agent.bestMove());
    expect(finalRootScore(again.values, rootKey)).toBe(finalRootScore(deep.values, rootKey));
  });

  it('keeps the same score and bestMove as the table-free engine over a random corpus', () => {
    for (const state of randomStates(ticTacToe, 30, {
      rng: createRng(CORPUS_SEED + 4),
      minPly: 0,
      maxPly: 6,
    })) {
      const rootKey = ticTacToe.stateKey(state);
      const off = recordSearch((recorder) => createTicTacToeAgent(recorder, 4, true), state);
      const on = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 4, true, createTranspositionTable()),
        state,
      );
      expect(finalRootScore(on.values, rootKey)).toBe(finalRootScore(off.values, rootKey));
      expect(on.agent.bestMove()).toEqual(off.agent.bestMove());
    }
  });
});
