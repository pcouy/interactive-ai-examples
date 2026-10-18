import { describe, expect, it } from 'vitest';
import {
  DRAW_SCORE,
  type EvalContext,
  type Evaluator,
  LOSS_SCORE,
  type PlayerId,
  type SearchAgent,
  type TreeRecorder,
  type ValueEvent,
  type VisitEvent,
  type VisitKind,
  WIN_SCORE,
  WIN_SCORE_MARGIN,
} from '../contracts.js';
import { type Connect4Move, type Connect4State, connect4 } from '../games/connect4.js';
import { type TicTacToeMove, type TicTacToeState, ticTacToe } from '../games/tic-tac-toe.js';
import { createRng, randomStates } from '../testing/playouts.js';
import { connect4Heuristic, ticTacToeHeuristic } from './evaluators/heuristic.js';
import { createRandomRolloutEvaluator } from './evaluators/random-rollout.js';
import { createMinimaxAgent } from './minimax.js';

const CORPUS_SEED = 20261018;
const CORPUS_DEPTH = 4;
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

function naiveMoveValue(
  state: TicTacToeState,
  move: TicTacToeMove,
  depth: number,
  rootPlayer: PlayerId,
): number {
  return naiveMinimax(ticTacToe.apply(state, move), depth - 1, rootPlayer);
}

function ticTacToeCorpus(rngSeed: number, count = 100): TicTacToeState[] {
  return randomStates(ticTacToe, count, { rng: createRng(rngSeed), minPly: 0, maxPly: 6 });
}

function createTicTacToeAgent(
  recorder: TreeRecorder<TicTacToeState>,
  depth: number,
  prune: boolean,
  evaluator = ticTacToeHeuristic,
): SearchAgent<TicTacToeState, TicTacToeMove> {
  return createMinimaxAgent(ticTacToe, { depth, evaluator, prune, recorder });
}

function createConnect4Agent(
  recorder: TreeRecorder<Connect4State>,
  depth: number,
  prune: boolean,
): SearchAgent<Connect4State, Connect4Move> {
  return createMinimaxAgent(connect4, { depth, evaluator: connect4Heuristic, prune, recorder });
}

describe('createMinimaxAgent', () => {
  it('matches an exhaustive naive reference on 100 random tic-tac-toe positions with pruning on and off', () => {
    for (const state of ticTacToeCorpus(CORPUS_SEED)) {
      const rootPlayer = ticTacToe.currentPlayer(state);
      const expected = naiveMinimax(state, CORPUS_DEPTH, rootPlayer);
      for (const prune of [false, true]) {
        const { agent, values } = recordSearch(
          (recorder) => createTicTacToeAgent(recorder, CORPUS_DEPTH, prune),
          state,
        );
        expect(finalRootScore(values, ticTacToe.stateKey(state))).toBe(expected);
        const move = agent.bestMove();
        if (ticTacToe.winner(state) !== null) {
          expect(move).toBeNull();
        } else if (move === null) {
          expect(ticTacToe.legalMoves(state)).toHaveLength(0);
        } else {
          expect(naiveMoveValue(state, move, CORPUS_DEPTH, rootPlayer)).toBe(expected);
        }
      }
    }
  });

  it('prunes to a strictly smaller visited tree with the same score and the same bestMove', () => {
    const midgame = [4, 0, 2, 6].reduce(
      (state, index) => ticTacToe.apply(state, { index }),
      ticTacToe.initialState(),
    );
    for (const state of [ticTacToe.initialState(), midgame]) {
      const rootKey = ticTacToe.stateKey(state);
      const off = recordSearch((recorder) => createTicTacToeAgent(recorder, 5, false), state);
      const on = recordSearch((recorder) => createTicTacToeAgent(recorder, 5, true), state);
      const offStats = off.agent.stats();
      const onStats = on.agent.stats();
      expect(finalRootScore(on.values, rootKey)).toBe(finalRootScore(off.values, rootKey));
      expect(on.agent.bestMove()).toEqual(off.agent.bestMove());
      expect(onStats.nodesVisited).toBeLessThan(offStats.nodesVisited);
      expect(offStats.nodesPruned).toBe(0);
      expect(onStats.nodesPruned).toBeGreaterThan(0);
    }
  });

  it('returns null before done and a legal move after done over both game corpora', () => {
    for (const state of ticTacToeCorpus(CORPUS_SEED + 1, 40)) {
      const agent = createTicTacToeAgent(
        { onVisit: () => undefined, onValue: () => undefined },
        3,
        true,
      );
      agent.start(state);
      expect(agent.bestMove()).toBeNull();
      let status = agent.step();
      while (status === 'running') {
        expect(agent.bestMove()).toBeNull();
        status = agent.step();
      }
      const move = agent.bestMove();
      if (ticTacToe.winner(state) !== null) {
        expect(move).toBeNull();
      } else if (move === null) {
        expect(ticTacToe.legalMoves(state)).toHaveLength(0);
      } else {
        expect(ticTacToe.legalMoves(state)).toContainEqual(move);
      }
    }
    for (const state of randomStates(connect4, 10, {
      rng: createRng(CORPUS_SEED + 2),
      minPly: 0,
      maxPly: 12,
    })) {
      const agent = createConnect4Agent(
        { onVisit: () => undefined, onValue: () => undefined },
        2,
        true,
      );
      agent.start(state);
      expect(agent.bestMove()).toBeNull();
      while (agent.step() === 'running') {
        expect(agent.bestMove()).toBeNull();
      }
      const move = agent.bestMove();
      if (connect4.winner(state) !== null) {
        expect(move).toBeNull();
      } else if (move === null) {
        expect(connect4.legalMoves(state)).toHaveLength(0);
      } else {
        expect(connect4.legalMoves(state)).toContainEqual(move);
      }
    }
  });

  it('reports coherent stats: elapsedSteps equals nodesVisited and visit kinds partition the tree', () => {
    const { agent, visits } = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true),
      ticTacToe.initialState(),
    );
    const stats = agent.stats();
    expect(stats.elapsedSteps).toBe(stats.nodesVisited);
    expect(stats.nodesVisited).toBe(visits.length);
    expect(stats.nodesPruned).toBe(visits.filter((visit) => visit.kind === 'pruned').length);
    expect(stats.leavesEvaluated).toBe(visits.filter((visit) => visit.kind === 'leaf').length);
    expect(stats.ttHits).toBe(0);
    expect(stats.ttMisses).toBe(0);
    const kinds = new Set<VisitKind>(['expand', 'pruned', 'tt-hit', 'leaf', 'terminal']);
    const seen = new Set<string>();
    for (const visit of visits) {
      expect(kinds.has(visit.kind)).toBe(true);
      if (visit.parentKey === null) {
        expect(seen.size).toBe(0);
      } else {
        expect(seen.has(visit.parentKey)).toBe(true);
      }
      seen.add(visit.nodeKey);
    }
  });

  it('replays the exact same visit and value sequence for the same evaluator seed', () => {
    const build = (seed: number) =>
      recordSearch((recorder) => {
        const evaluator = createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>({
          rng: createRng(seed),
          samples: 2,
          maxPlies: 12,
        });
        return createTicTacToeAgent(recorder, 3, true, evaluator);
      }, ticTacToe.initialState());
    const first = build(7);
    const replay = build(7);
    const other = build(8);
    expect(replay.visits).toEqual(first.visits);
    expect(replay.values).toEqual(first.values);
    expect(replay.agent.bestMove()).toEqual(first.agent.bestMove());
    expect(other.visits).not.toEqual(first.visits);
  });

  it('detects an immediate winning move and scores it exactly', () => {
    const state: TicTacToeState = {
      cells: [0, 0, null, 1, null, null, null, null, null],
      currentPlayer: 0,
    };
    const { agent, visits, values } = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 6, true),
      state,
    );
    expect(agent.bestMove()).toEqual({ index: 2 });
    expect(finalRootScore(values, ticTacToe.stateKey(state))).toBe(WIN_SCORE);
    const winningKey = ticTacToe.stateKey(ticTacToe.apply(state, { index: 2 }));
    const winningVisit = visits.find((visit) => visit.nodeKey === winningKey);
    expect(winningVisit?.kind).toBe('terminal');
  });

  it('never calls the evaluator on a terminal state', () => {
    const spy: Evaluator<TicTacToeState, TicTacToeMove> = {
      id: 'spy',
      evaluate(state: TicTacToeState, ctx: EvalContext<TicTacToeState, TicTacToeMove>) {
        if (ticTacToe.winner(state) !== null) {
          throw new Error('evaluator called on a terminal state');
        }
        return ticTacToeHeuristic.evaluate(state, ctx);
      },
    };
    let calls = 0;
    const counting: Evaluator<TicTacToeState, TicTacToeMove> = {
      id: 'counting',
      evaluate(state, ctx) {
        calls += 1;
        return spy.evaluate(state, ctx);
      },
    };
    for (const state of ticTacToeCorpus(CORPUS_SEED + 3, 12)) {
      recordSearch((recorder) => createTicTacToeAgent(recorder, 5, true, counting), state);
    }
    expect(calls).toBeGreaterThan(0);
  });

  it('abandons a running search cleanly when start is called again', () => {
    const visits: VisitEvent<TicTacToeState>[] = [];
    const recorder: TreeRecorder<TicTacToeState> = {
      onVisit: (event) => visits.push(event),
      onValue: () => undefined,
    };
    const agent = createTicTacToeAgent(recorder, 5, true);
    agent.start(ticTacToe.initialState());
    for (let step = 0; step < 4 && agent.step() === 'running'; step += 1) {
      expect(agent.bestMove()).toBeNull();
    }
    const abortedVisits = visits.length;
    expect(abortedVisits).toBeGreaterThan(0);
    const resumed = [4, 0, 2].reduce(
      (state, index) => ticTacToe.apply(state, { index }),
      ticTacToe.initialState(),
    );
    agent.start(resumed);
    expect(agent.bestMove()).toBeNull();
    while (agent.step() === 'running') {
      expect(agent.bestMove()).toBeNull();
    }
    const move = agent.bestMove();
    expect(move).not.toBeNull();
    expect(ticTacToe.legalMoves(resumed)).toContainEqual(move);
    expect(visits.length).toBeGreaterThan(abortedVisits);
    expect(agent.stats().nodesVisited).toBe(visits.length - abortedVisits);
  });

  it('runs a connect-four smoke search at depth 2', () => {
    const { agent, visits, values } = recordSearch(
      (recorder) => createConnect4Agent(recorder, 2, true),
      connect4.initialState(),
    );
    const stats = agent.stats();
    expect(stats.elapsedSteps).toBe(stats.nodesVisited);
    expect(stats.nodesVisited).toBe(visits.length);
    expect(stats.nodesVisited).toBeGreaterThan(0);
    expect(visits[0]?.nodeKey).toBe(connect4.stateKey(connect4.initialState()));
    expect(visits[0]?.kind).toBe('expand');
    const move = agent.bestMove();
    expect(move).not.toBeNull();
    expect(connect4.legalMoves(connect4.initialState())).toContainEqual(move);
    const score = finalRootScore(values, connect4.stateKey(connect4.initialState()));
    expect(Math.abs(score)).toBeLessThan(WIN_SCORE_MARGIN);
  });
});
