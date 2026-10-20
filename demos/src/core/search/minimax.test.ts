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
import {
  createReadOnlyOverlay,
  createTranspositionTable,
  SOLVED_DEPTH,
  type TranspositionTable,
} from './transposition.js';

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

function rootKeyOf<S>(visits: VisitEvent<S>[]): string {
  const rootKey = visits[0]?.nodeKey;
  if (rootKey === undefined) {
    throw new Error('expected at least one visit');
  }
  return rootKey;
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

function naiveMinimax(
  state: TicTacToeState,
  depth: number,
  rootPlayer: PlayerId,
  gamma = 1,
): number {
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
    const value = naiveMinimax(ticTacToe.apply(state, move), depth - 1, rootPlayer, gamma);
    // Un seul facteur γ par ply de remontée, comme dans accumulate : le
    // nested γ*(γ*(...)) doit être identique bit à bit au moteur.
    best = maximizing ? Math.max(best, gamma * value) : Math.min(best, gamma * value);
  }
  return best;
}

function naiveMoveValue(
  state: TicTacToeState,
  move: TicTacToeMove,
  depth: number,
  rootPlayer: PlayerId,
  gamma = 1,
): number {
  return naiveMinimax(ticTacToe.apply(state, move), depth - 1, rootPlayer, gamma);
}

function ticTacToeCorpus(rngSeed: number, count = 100): TicTacToeState[] {
  return randomStates(ticTacToe, count, { rng: createRng(rngSeed), minPly: 0, maxPly: 6 });
}

function createTicTacToeAgent(
  recorder: TreeRecorder<TicTacToeState>,
  depth: number,
  prune: boolean,
  evaluator: Evaluator<TicTacToeState, TicTacToeMove> = ticTacToeHeuristic,
  extra: {
    discount?: number;
    rng?: () => number;
    transposition?: TranspositionTable;
    probeTurns?: 'all' | 'previous';
    strictPruning?: boolean;
  } = {},
): SearchAgent<TicTacToeState, TicTacToeMove> {
  return createMinimaxAgent(ticTacToe, { depth, evaluator, prune, recorder, ...extra });
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
        const { agent, visits, values } = recordSearch(
          (recorder) => createTicTacToeAgent(recorder, CORPUS_DEPTH, prune),
          state,
        );
        expect(finalRootScore(values, rootKeyOf(visits))).toBe(expected);
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
      // Sans rng, le minimax pur (prune: false) départage désormais au hasard :
      // on fige la source pour comparer les deux modes sur le même tirage
      // (rng ≈ 1 conserve le premier ex æquo visité).
      const rng = () => 0.999999;
      const off = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 5, false, ticTacToeHeuristic, { rng }),
        state,
      );
      const on = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 5, true, ticTacToeHeuristic, { rng }),
        state,
      );
      const rootKey = rootKeyOf(off.visits);
      expect(rootKeyOf(on.visits)).toBe(rootKey);
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
    expect(finalRootScore(values, rootKeyOf(visits))).toBe(WIN_SCORE);
    const winningKey = ticTacToe.stateKey(ticTacToe.apply(state, { index: 2 }));
    const winningVisit = visits.find((visit) => ticTacToe.stateKey(visit.state) === winningKey);
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
    expect(visits[0]?.nodeKey).toBe(`${connect4.stateKey(connect4.initialState())}#1`);
    expect(visits[0]?.kind).toBe('expand');
    const move = agent.bestMove();
    expect(move).not.toBeNull();
    expect(connect4.legalMoves(connect4.initialState())).toContainEqual(move);
    const score = finalRootScore(values, rootKeyOf(visits));
    expect(Math.abs(score)).toBeLessThan(WIN_SCORE_MARGIN);
  });

  it('backpropagates each expanded node final score as the min/max of its children scores', () => {
    // Position avec de nombreuses transpositions : un même état est atteint
    // par plusieurs ordres de coups, chaque occurrence garde son propre nœud.
    const state = [4, 0, 2].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    const { visits, values } = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 5, true),
      state,
    );
    const rootPlayer = ticTacToe.currentPlayer(state);
    const finalValue = new Map<string, number>();
    for (const value of values) {
      finalValue.set(value.nodeKey, value.score);
    }
    const expanded = visits.filter((visit) => visit.kind === 'expand');
    expect(expanded.length).toBeGreaterThan(0);
    for (const visit of expanded) {
      const own = finalValue.get(visit.nodeKey);
      expect(own).toBeDefined();
      const childScores = visits
        .filter((child) => child.parentKey === visit.nodeKey && finalValue.has(child.nodeKey))
        .map((child) => finalValue.get(child.nodeKey) as number);
      expect(childScores.length).toBeGreaterThan(0);
      const maximizing = ticTacToe.currentPlayer(visit.state) === rootPlayer;
      const expected = maximizing ? Math.max(...childScores) : Math.min(...childScores);
      expect(own).toBe(expected);
    }
  });

  it('gives each visit a path-unique node key, even for transposed states', () => {
    const { visits } = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 5, true),
      ticTacToe.initialState(),
    );
    const keys = new Set(visits.map((visit) => visit.nodeKey));
    expect(keys.size).toBe(visits.length);
    const stateKeys = new Set(visits.map((visit) => ticTacToe.stateKey(visit.state)));
    expect(stateKeys.size).toBeLessThan(visits.length);
  });

  // Positions ex æquo : depuis l'état initial à profondeur 3, aucun terminal
  // n'est atteignable (X n'a que deux pions sur ses lignes) et l'heuristique
  // renvoie zéro — chacun des 9 coups initiaux atteint donc la même valeur.
  const TIE_STATE = ticTacToe.initialState();
  const TIE_DEPTH = 3;

  function optimalMoves(state: TicTacToeState, depth: number): TicTacToeMove[] {
    const rootPlayer = ticTacToe.currentPlayer(state);
    const expected = naiveMinimax(state, depth, rootPlayer);
    return ticTacToe
      .legalMoves(state)
      .filter((move) => naiveMoveValue(state, move, depth, rootPlayer) === expected);
  }

  function moveIndex(move: TicTacToeMove | null): number {
    if (move === null) {
      throw new Error('expected a non-null best move');
    }
    return move.index;
  }

  it('resolves root ties with the injected rng: first candidate with rng≈1, last with rng=0', () => {
    // Sans élagage, chaque valeur enfant est exacte : les ex æquo de la racine
    // sont authentiques et peuvent donc être départagés.
    const optimal = optimalMoves(TIE_STATE, TIE_DEPTH);
    expect(optimal.length).toBeGreaterThanOrEqual(2);
    // rng renvoyant presque 1 : 1/i ≤ 0,5 < rng, jamais de remplacement →
    // le PREMIER coup optimal visité est retenu.
    const keepFirst = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, TIE_DEPTH, false, ticTacToeHeuristic, {
          rng: () => 0.999999,
        }),
      TIE_STATE,
    );
    // rng renvoyant 0 : 0 < 1/i pour tout i ≥ 2, remplacement systématique →
    // le DERNIER coup optimal visité est retenu.
    const keepLast = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, TIE_DEPTH, false, ticTacToeHeuristic, { rng: () => 0 }),
      TIE_STATE,
    );
    const first = keepFirst.agent.bestMove();
    const last = keepLast.agent.bestMove();
    expect(first).not.toBeNull();
    expect(last).not.toBeNull();
    const optimalIndexes = new Set(optimal.map((move) => move.index));
    expect(optimalIndexes.has(moveIndex(first))).toBe(true);
    expect(optimalIndexes.has(moveIndex(last))).toBe(true);
    expect(first).not.toEqual(last);
    expect(first).toEqual(optimal[0]);
    expect(last).toEqual(optimal[optimal.length - 1]);
  });

  it('picks at least two distinct optimal moves across repeated runs with a random rng', () => {
    // Sans élagage les ex æquo sont exacts ; le tirage Math.random (la source
    // par défaut du moteur quand on lui fournit un rng) choisit alors un coup
    // optimal varié d'une recherche à l'autre.
    const optimalIndexes = new Set(optimalMoves(TIE_STATE, TIE_DEPTH).map((move) => move.index));
    const expected = naiveMinimax(TIE_STATE, TIE_DEPTH, ticTacToe.currentPlayer(TIE_STATE));
    const picked = new Set<number>();
    for (let run = 0; run < 200; run += 1) {
      const { agent, values, visits } = recordSearch(
        (recorder) =>
          createTicTacToeAgent(recorder, TIE_DEPTH, false, ticTacToeHeuristic, {
            rng: Math.random,
          }),
        TIE_STATE,
      );
      const move = agent.bestMove();
      const index = moveIndex(move);
      // Chaque coup retenu est ex æquo au sommet donc optimal.
      expect(optimalIndexes.has(index)).toBe(true);
      expect(naiveMoveValue(TIE_STATE, move as TicTacToeMove, TIE_DEPTH, 0)).toBe(expected);
      expect(finalRootScore(values, rootKeyOf(visits))).toBe(expected);
      picked.add(index);
    }
    // À 9 ex æquo équiprobables sur 200 tirages, ne voir qu'un seul coup a
    // une probabilité négligeable (9 × (1/9)²⁰⁰).
    expect(picked.size).toBeGreaterThanOrEqual(2);
  });

  it('randomizes plain minimax ties by default when no rng is supplied', () => {
    // Sans élagage, tous les enfants sont exacts : les ex æquo sont donc
    // authentiques et le moteur tire par défaut sur Math.random, sans qu'il
    // faille injecter une source. Le coup retenu reste optimal.
    const optimalIndexes = new Set(optimalMoves(TIE_STATE, TIE_DEPTH).map((move) => move.index));
    const expected = naiveMinimax(TIE_STATE, TIE_DEPTH, ticTacToe.currentPlayer(TIE_STATE));
    const picked = new Set<number>();
    for (let run = 0; run < 200; run += 1) {
      const { agent, values, visits } = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, TIE_DEPTH, false),
        TIE_STATE,
      );
      const move = agent.bestMove();
      const index = moveIndex(move);
      expect(optimalIndexes.has(index)).toBe(true);
      expect(naiveMoveValue(TIE_STATE, move as TicTacToeMove, TIE_DEPTH, 0)).toBe(expected);
      expect(finalRootScore(values, rootKeyOf(visits))).toBe(expected);
      picked.add(index);
    }
    expect(picked.size).toBeGreaterThanOrEqual(2);
  });

  it('keeps classic alpha-beta deterministic without an rng, unlike plain minimax', () => {
    // L'alpha-bêta classique conserve le premier ex æquo visité : ses arbres
    // restent comparables à l'identique. Seuls `prune: false` et l'élagage
    // strict tirent par défaut.
    const run = () =>
      recordSearch((recorder) => createTicTacToeAgent(recorder, 4, true), ticTacToe.initialState());
    const first = run();
    const replay = run();
    expect(replay.visits).toEqual(first.visits);
    expect(replay.values).toEqual(first.values);
    expect(replay.agent.bestMove()).toEqual(first.agent.bestMove());
  });

  it('does not cut on equality in strict mode and still matches the naive reference', () => {
    // Position de milieu de partie où l'élagage standard coupe réellement :
    // c'est le témoin qui prouve que le mode strict change bien le comportement.
    const midgame = [4, 0, 2, 6].reduce(
      (state, index) => ticTacToe.apply(state, { index }),
      ticTacToe.initialState(),
    );
    const rootPlayer = ticTacToe.currentPlayer(midgame);
    const expected = naiveMinimax(midgame, CORPUS_DEPTH, rootPlayer);
    const standard = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, CORPUS_DEPTH, true),
      midgame,
    );
    const strict = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, CORPUS_DEPTH, true, ticTacToeHeuristic, {
          strictPruning: true,
        }),
      midgame,
    );
    expect(standard.agent.stats().nodesPruned).toBeGreaterThan(0);
    // L'égalité ne coupant plus, le mode strict visite au moins autant de
    // nœuds et en élague au plus autant, sans changer la valeur racine.
    expect(strict.agent.stats().nodesVisited).toBeGreaterThanOrEqual(
      standard.agent.stats().nodesVisited,
    );
    expect(strict.agent.stats().nodesPruned).toBeLessThanOrEqual(
      standard.agent.stats().nodesPruned,
    );
    expect(finalRootScore(strict.values, rootKeyOf(strict.visits))).toBe(expected);
    const move = strict.agent.bestMove();
    if (move !== null) {
      expect(naiveMoveValue(midgame, move, CORPUS_DEPTH, rootPlayer)).toBe(expected);
    }
  });

  it('randomizes strict-mode ties with the injected rng: first with rng≈1, last with rng=0', () => {
    const optimal = optimalMoves(TIE_STATE, TIE_DEPTH);
    expect(optimal.length).toBeGreaterThanOrEqual(2);
    const optimalIndexes = new Set(optimal.map((move) => move.index));
    const keepFirst = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, TIE_DEPTH, true, ticTacToeHeuristic, {
          strictPruning: true,
          rng: () => 0.999999,
        }),
      TIE_STATE,
    );
    const keepLast = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, TIE_DEPTH, true, ticTacToeHeuristic, {
          strictPruning: true,
          rng: () => 0,
        }),
      TIE_STATE,
    );
    const first = keepFirst.agent.bestMove();
    const last = keepLast.agent.bestMove();
    expect(optimalIndexes.has(moveIndex(first))).toBe(true);
    expect(optimalIndexes.has(moveIndex(last))).toBe(true);
    expect(first).not.toEqual(last);
    expect(first).toEqual(optimal[0]);
    expect(last).toEqual(optimal[optimal.length - 1]);
  });

  it('falls back to Math.random to break strict-mode ties when no rng is provided', () => {
    const optimalIndexes = new Set(optimalMoves(TIE_STATE, TIE_DEPTH).map((move) => move.index));
    const picked = new Set<number>();
    for (let run = 0; run < 200; run += 1) {
      const { agent } = recordSearch(
        (recorder) =>
          createTicTacToeAgent(recorder, TIE_DEPTH, true, ticTacToeHeuristic, {
            strictPruning: true,
          }),
        TIE_STATE,
      );
      const index = moveIndex(agent.bestMove());
      // Les ex æquo stricts sont authentiques (aucune coupure sur égalité) :
      // chaque coup retenu est donc optimal.
      expect(optimalIndexes.has(index)).toBe(true);
      picked.add(index);
    }
    expect(picked.size).toBeGreaterThanOrEqual(2);
  });

  it('reports cut metadata consistent with the window on every pruned visit', () => {
    const state = ticTacToe.initialState();
    const rootPlayer = ticTacToe.currentPlayer(state);
    // Profondeur 4 pour le mode standard, comme la référence. En mode strict il
    // faut descendre à 5 : à 4 plies aucun terminal n'est atteignable depuis
    // l'état initial et l'heuristique vaut partout zéro, donc aucune inégalité
    // STRICTE n'apparaît et le mode strict ne coupe jamais (0 nœud élagué).
    const cases = [
      { depth: 4, strictPruning: false },
      { depth: 5, strictPruning: true },
    ] as const;
    for (const { depth, strictPruning } of cases) {
      const { visits, values } = recordSearch(
        (recorder) =>
          createTicTacToeAgent(recorder, depth, true, ticTacToeHeuristic, { strictPruning }),
        state,
      );
      // Chaque évènement porte la fenêtre du nœud qui l'émet.
      for (const visit of visits) {
        expect(visit.window).toBeDefined();
      }
      for (const value of values) {
        expect(value.window).toBeDefined();
      }
      const pruned = visits.filter((visit) => visit.kind === 'pruned');
      expect(pruned.length).toBeGreaterThan(0);
      for (const visit of pruned) {
        const cut = visit.cut;
        const window = visit.window;
        if (cut === undefined || window === undefined) {
          throw new Error('expected cut and window on a pruned visit');
        }
        const parent = visits.find((candidate) => candidate.nodeKey === visit.parentKey);
        if (parent === undefined) {
          throw new Error('expected the pruned visit parent to be recorded');
        }
        // Le parent maximise quand c'est le joueur racine qui doit jouer.
        const parentMaximizes = ticTacToe.currentPlayer(parent.state) === rootPlayer;
        expect(cut.bound).toBe(parentMaximizes ? 'beta' : 'alpha');
        expect(cut.strict).toBe(strictPruning);
        expect(cut.limit).toBe(parentMaximizes ? window.beta : window.alpha);
        // Le déclencheur est un frère : même parent, visité avant la coupure.
        const trigger = visits.find((candidate) => candidate.nodeKey === cut.triggerKey);
        expect(trigger).toBeDefined();
        expect(trigger?.parentKey).toBe(visit.parentKey);
      }
    }
    // Une recherche sans coupure n'émet aucun `cut` : le champ reste absent
    // plutôt que présent à undefined. Tous ses évènements portent une fenêtre.
    const cutFree = recordSearch((recorder) => createTicTacToeAgent(recorder, 4, false), state);
    for (const visit of cutFree.visits) {
      expect(visit.window).toBeDefined();
    }
    for (const value of cutFree.values) {
      expect(value.window).toBeDefined();
    }
    const rootKey = rootKeyOf(cutFree.visits);
    const rootValue = cutFree.values
      .filter((value) => value.nodeKey === rootKey && value.exact)
      .at(-1);
    expect(rootValue).toBeDefined();
    if (rootValue !== undefined) {
      expect('cut' in rootValue).toBe(false);
    }
    expect(cutFree.visits.some((visit) => visit.kind === 'pruned')).toBe(false);
  });

  it('discounts the value reaching the root by γ per ply while emitting raw child scores', () => {
    // X gagne immédiatement avec { index: 2 } : l'enfant terminal est à un ply
    // de la racine, la racine voit donc γ × WIN_SCORE.
    const state: TicTacToeState = {
      cells: [0, 0, null, 1, null, null, null, null, null],
      currentPlayer: 0,
    };
    const winningKey = ticTacToe.stateKey(ticTacToe.apply(state, { index: 2 }));
    for (const [gamma, expected] of [
      [1, WIN_SCORE],
      [0.5, WIN_SCORE / 2],
    ] as const) {
      const { agent, visits, values } = recordSearch(
        (recorder) =>
          createTicTacToeAgent(recorder, 6, true, ticTacToeHeuristic, { discount: gamma }),
        state,
      );
      expect(agent.bestMove()).toEqual({ index: 2 });
      expect(finalRootScore(values, rootKeyOf(visits))).toBe(expected);
      // L'évènement de valeur du terminal reste BRUT : l'arbre enregistré n'est
      // pas actualisé, seule la remontée vers le parent l'est.
      const winningVisit = visits.find((visit) => ticTacToe.stateKey(visit.state) === winningKey);
      expect(winningVisit?.kind).toBe('terminal');
      const winningValues = values.filter(
        (value) => value.nodeKey === winningVisit?.nodeKey && value.exact,
      );
      expect(winningValues.at(-1)?.score).toBe(WIN_SCORE);
    }
  });

  it('prefers the fastest win under discount: immediate win over a forced 3-ply win', () => {
    // X sur 0 et 4, O sur 2 et 6. Victoire immédiate : { index: 8 } (diagonale
    // 0-4-8). Fourchette avec { index: 1 } : menaces 7 (colonne 1-4-7) et 8
    // (diagonale), O ne peut parer qu'une → victoire forcée à 3 plies.
    const state: TicTacToeState = {
      cells: [0, null, 1, null, 0, null, 1, null, null],
      currentPlayer: 0,
    };
    const fast = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 5, true, ticTacToeHeuristic, { discount: 0.5 }),
      state,
    );
    // Victoire immédiate : 0,5¹ = 0,5, strictement mieux que 0,5³ = 0,125.
    expect(fast.agent.bestMove()).toEqual({ index: 8 });
    expect(finalRootScore(fast.values, rootKeyOf(fast.visits))).toBe(0.5);
    // Sans actualisation, les deux victoires valent WIN_SCORE : la racine
    // reste gagnante mais le coup retenu n'est plus déterminable.
    const slow = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 5, true, ticTacToeHeuristic, { discount: 1 }),
      state,
    );
    expect(finalRootScore(slow.values, rootKeyOf(slow.visits))).toBe(WIN_SCORE);
  });

  it('matches the discounting naive reference at γ = 0,9 on the corpus with pruning on and off', () => {
    const gamma = 0.9;
    for (const state of ticTacToeCorpus(CORPUS_SEED)) {
      const rootPlayer = ticTacToe.currentPlayer(state);
      const expected = naiveMinimax(state, CORPUS_DEPTH, rootPlayer, gamma);
      for (const prune of [false, true]) {
        const { visits, values } = recordSearch(
          (recorder) =>
            createTicTacToeAgent(recorder, CORPUS_DEPTH, prune, ticTacToeHeuristic, {
              discount: gamma,
            }),
          state,
        );
        expect(finalRootScore(values, rootKeyOf(visits))).toBe(expected);
      }
    }
  });

  it('replays with probeTurns previous expand everything expanded in the current turn', () => {
    const state = [4, 0, 2, 6].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    // Recherche jouée : elle peuple la table au tour courant... et consomme
    // aussi ses propres entrées de transposition internes (politique 'all').
    const table = createTranspositionTable();
    const played = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 4, true, ticTacToeHeuristic, { transposition: table }),
      state,
    );
    // Rejoué : même table, même tour — chaque entrée estampillée du tour
    // courant est refusée, le rejoué redéploie donc tout l'arbre.
    const replay = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 4, true, ticTacToeHeuristic, {
          transposition: table,
          probeTurns: 'previous',
        }),
      state,
    );
    // Toutes les entrées de la table viennent du tour courant : zéro coupure.
    expect(replay.agent.stats().ttHits).toBe(0);
    // Refuser toutes les entrées du tour courant redéploie exactement l'arbre
    // complet — identique (visites et valeurs) à une recherche sans table.
    const untabled = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true, ticTacToeHeuristic),
      state,
    );
    expect(replay.agent.stats().nodesVisited).toBe(untabled.agent.stats().nodesVisited);
    expect(replay.visits).toEqual(untabled.visits);
    expect(replay.values).toEqual(untabled.values);
    // Même valeur et même coup à la racine que la recherche jouée.
    expect(finalRootScore(replay.values, rootKeyOf(replay.visits))).toBe(
      finalRootScore(played.values, rootKeyOf(played.visits)),
    );
    expect(replay.agent.bestMove()).toEqual(played.agent.bestMove());
  });

  it('probeTurns previous still consumes strictly deeper entries from previous turns', () => {
    const state = [4, 0, 2, 6].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    // Recherche profonde au tour 0 : ses entrées (profondeur certifiée ≥ leur
    // profondeur réelle) restent consommables, strictement plus profondes, au
    // tour suivant.
    const table = createTranspositionTable();
    const deeper = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 4, true, ticTacToeHeuristic, { transposition: table }),
      state,
    );
    table.beginTurn();
    // Rejoué moins profond : avec F1 (scores intrinsèques) et F2 (profondeur
    // certifiée), il consomme les entrées antérieures plus profondes que son
    // budget et hérite donc de LEUR valeur, plus profonde que sa propre
    // horizon nominale.
    const shorter = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 2, true, ticTacToeHeuristic, {
          transposition: table,
          probeTurns: 'previous',
        }),
      state,
    );
    // Témoin : même recherche sur une table vierge, bornée à son horizon.
    const control = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 2, true, ticTacToeHeuristic, {
          transposition: createTranspositionTable(),
        }),
      state,
    );
    expect(shorter.agent.stats().ttHits).toBeGreaterThan(0);
    expect(shorter.agent.stats().nodesVisited).toBeLessThan(control.agent.stats().nodesVisited);
    // Le rejoué restitue le résultat PROFOND réutilisé, pas celui du témoin
    // superficiel.
    expect(finalRootScore(shorter.values, rootKeyOf(shorter.visits))).toBe(
      finalRootScore(deeper.values, rootKeyOf(deeper.visits)),
    );
  });

  it('replays through a read-only overlay: same-turn hits appear and the real table is untouched', () => {
    const state = [4, 0, 2, 6].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    // Recherche jouée au tour courant : elle peuple la table réelle d'entrées
    // estampillées du tour courant.
    const real = createTranspositionTable();
    recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 4, true, ticTacToeHeuristic, { transposition: real }),
      state,
    );
    const realBefore = real.snapshot();
    // Rejoué : options par défaut (probeTurns 'all'), table en lecture seule
    // par-dessus la réelle. Ses propres touches du tour courant apparaissent
    // comme tt-hit dans l'arbre.
    const overlay = createReadOnlyOverlay(real);
    const replay = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 4, true, ticTacToeHeuristic, { transposition: overlay }),
      state,
    );
    expect(replay.agent.stats().ttHits).toBeGreaterThan(0);
    const ttHitVisit = replay.visits.find((visit) => visit.kind === 'tt-hit');
    if (ttHitVisit === undefined) {
      throw new Error('expected a tt-hit visit in the replay');
    }
    const entryDepth = ttHitVisit.entryDepth;
    expect(entryDepth).toBeDefined();
    // La profondeur certifiée de l'entrée consommée est celle stockée en table.
    const consumed = overlay
      .snapshot()
      .find((entry) => entry.key === ticTacToe.stateKey(ttHitVisit.state));
    expect(consumed?.depth).toBe(entryDepth);
    // Le tour 0 prouve que la profondeur provient de l'entrée temporaire de
    // l'overlay (du tour courant), pas de l'entrée réelle (toujours turn -1).
    expect(consumed?.turn).toBe(0);
    // L'évènement de valeur du même nœud porte la même profondeur.
    const matchingValue = replay.values.find((value) => value.nodeKey === ttHitVisit.nodeKey);
    expect(matchingValue?.entryDepth).toBe(entryDepth);
    // La table réelle n'a jamais été écrite par le rejoué.
    expect(real.snapshot()).toEqual(realBefore);
  });

  // Contrat de la table partagée : les entrées sont intrinsèques à l'état
  // (perspective du joueur au trait) et portent une profondeur certifiée, si
  // bien qu'elles sont réutilisables d'un tour et d'un joueur racine à l'autre
  // et permettent de dépasser l'horizon nominal d'une recherche superficielle.
  it('reuses a deep search from the opposite perspective by exact negation', () => {
    const table = createTranspositionTable();
    const base = [4, 0, 2, 6].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    // Recherche pleine profondeur, racine = joueur 0.
    recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 9, false, ticTacToeHeuristic, { transposition: table }),
      base,
    );
    table.beginTurn();
    const children = ticTacToe.legalMoves(base).slice(0, 3);
    expect(children.length).toBeGreaterThan(0);
    for (const move of children) {
      const childState = ticTacToe.apply(base, move);
      // L'enfant a le joueur 1 au trait : perspective opposée à la racine
      // profonde, la restitution passe donc par une négation exacte.
      expect(ticTacToe.currentPlayer(childState)).toBe(1);
      const shallow = recordSearch(
        (recorder) =>
          createTicTacToeAgent(recorder, 2, true, ticTacToeHeuristic, { transposition: table }),
        childState,
      );
      expect(shallow.agent.stats().ttHits).toBeGreaterThan(0);
      // La négation d'un zéro produit -0 : on compare la valeur, pas le bit de
      // signe du zéro.
      const reference = naiveMinimax(childState, 9, 1);
      expect(finalRootScore(shallow.values, rootKeyOf(shallow.visits)) + 0).toBe(reference + 0);
    }
  });

  it('reuses a solved position to search deeper than its nominal horizon', () => {
    const table = createTranspositionTable();
    // Position où la valeur pleine profondeur diffère de la valeur à horizon 1
    // (résultat forcé à au moins 2 plies).
    let target: TicTacToeState | null = null;
    let fullValue = 0;
    let shallowValue = 0;
    for (const candidate of randomStates(ticTacToe, 400, {
      rng: createRng(CORPUS_SEED + 11),
      minPly: 4,
      maxPly: 6,
    })) {
      if (ticTacToe.winner(candidate) !== null) {
        continue;
      }
      const player = ticTacToe.currentPlayer(candidate);
      const full = naiveMinimax(candidate, 9, player);
      const atHorizon = naiveMinimax(candidate, 1, player);
      if (full !== atHorizon) {
        target = candidate;
        fullValue = full;
        shallowValue = atHorizon;
        break;
      }
    }
    if (target === null) {
      throw new Error('expected a corpus position with a forced result');
    }
    expect(fullValue).not.toBe(shallowValue);
    // Recherche pleine profondeur : résout la position et peuple la table.
    recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 9, false, ticTacToeHeuristic, { transposition: table }),
      target,
    );
    table.beginTurn();
    // Recherche à profondeur 1 sur la MÊME table : elle réutilise la valeur
    // résolue, donc dépasse son horizon nominal.
    const reused = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 1, true, ticTacToeHeuristic, { transposition: table }),
      target,
    );
    expect(reused.agent.stats().ttHits).toBeGreaterThan(0);
    // Profondeur effective certifiée : strictement au-delà du budget nominal
    // (les enfants racine sont des tt-hits résolus plus profond).
    expect(reused.agent.stats().effectiveDepth).toBeGreaterThan(1);
    expect(finalRootScore(reused.values, rootKeyOf(reused.visits))).toBe(fullValue);
    // Témoin : même recherche sur table vierge, bornée à son horizon.
    const control = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 1, true, ticTacToeHeuristic, {
          transposition: createTranspositionTable(),
        }),
      target,
    );
    expect(finalRootScore(control.values, rootKeyOf(control.visits))).toBe(shallowValue);
  });

  it('stores solved positions with the certified SOLVED_DEPTH and reports it', () => {
    const table = createTranspositionTable();
    const state = [4, 0, 2, 6].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    const solved = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 9, false, ticTacToeHeuristic, { transposition: table }),
      state,
    );
    // Au moins une position entièrement résolue : profondeur certifiée infinie.
    expect(table.snapshot().some((entry) => entry.depth === SOLVED_DEPTH)).toBe(true);
    // La profondeur effective de la racine est elle aussi certifiée résolue.
    expect(solved.agent.stats().effectiveDepth).toBe(SOLVED_DEPTH);
  });

  it('does not understate the certified depth of a pruned, bounded search', () => {
    const depth = 4;
    const state = ticTacToe.initialState();
    const table = createTranspositionTable();
    const bounded = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, depth, true, ticTacToeHeuristic, { transposition: table }),
      state,
    );
    const stats = bounded.agent.stats();
    // L'élagage produit réellement des frames coupés (donc des enfants bornés)…
    expect(stats.nodesPruned).toBeGreaterThan(0);
    // … sans effondrer la profondeur effective à 1 : un enfant borné certifie
    // son budget restant, donc la racine atteint la profondeur nominale ici
    // (position non résolue : aucun terminal atteignable en 4 plies, et pas de
    // réutilisation plus profonde). La profondeur ne se sous-estime jamais.
    expect(stats.effectiveDepth).toBe(depth);
    // L'entrée stockée pour la racine porte cette même profondeur certifiée.
    const rootEntry = table.snapshot().find((entry) => entry.key === ticTacToe.stateKey(state));
    expect(rootEntry?.depth).toBe(depth);
  });

  it('stamps every stored entry with the piece count of its position', () => {
    const state = [4, 0, 2, 6].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    const pieces = ticTacToe.pieceCount?.(state);
    expect(pieces).toBe(4);
    const table = createTranspositionTable();
    recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 4, true, ticTacToeHeuristic, { transposition: table }),
      state,
    );
    // La racine stockée porte le tour intra-partie de sa position.
    const rootEntry = table.snapshot().find((entry) => entry.key === ticTacToe.stateKey(state));
    expect(rootEntry?.pieces).toBe(4);
    // Chaque entrée porte le compte de pièces de sa propre position.
    for (const entry of table.snapshot()) {
      expect(entry.pieces).toBeDefined();
    }
  });

  // Cache-feuille rollout : les feuilles d'horizon des évaluateurs
  // stochastic-échantillonnés sont stockées à profondeur 0 avec leur nombre
  // de tirages, et chaque re-encounter élargit la moyenne accumulée.
  const ROLLOUT_SEED = 21;
  const ROLLOUT_SAMPLES = 2;

  function rolloutEvaluator(): Evaluator<TicTacToeState, TicTacToeMove> {
    return createRandomRolloutEvaluator<TicTacToeState, TicTacToeMove>({
      rng: createRng(ROLLOUT_SEED),
      samples: ROLLOUT_SAMPLES,
      maxPlies: 12,
    });
  }

  it('accumulates rollout samples into depth-0 entries across transpositions', () => {
    const state = [4, 0, 2, 6].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    const table = createTranspositionTable();
    const { agent, visits } = recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 3, true, rolloutEvaluator(), { transposition: table }),
      state,
    );
    const depthZero = table.snapshot().filter((entry) => entry.depth === 0);
    // Au moins une transposition d'horizon a fusionné ses tirages : la
    // moyenne porte alors plus de tirages que les ROLLOUT_SAMPLES initiaux.
    expect(depthZero.some((entry) => (entry.samples ?? 0) > ROLLOUT_SAMPLES)).toBe(true);
    for (const entry of depthZero) {
      // Tout état profondeur 0 est une feuille rollout : pas d'entrée sans
      // samples, et un score de tirage reste dans [-1, 1].
      expect(entry.samples).toBeDefined();
      expect(entry.score).toBeGreaterThanOrEqual(LOSS_SCORE);
      expect(entry.score).toBeLessThanOrEqual(WIN_SCORE);
    }
    // Le cas spécial de fusion émet une vraie feuille : l'invariant
    // « leavesEvaluated = nombre de visites 'leaf' » est préservé.
    expect(agent.stats().leavesEvaluated).toBe(
      visits.filter((visit) => visit.kind === 'leaf').length,
    );
  });

  it('stamps rollout depth-0 entries with the piece count too', () => {
    const state = [4, 0, 2, 6].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    const table = createTranspositionTable();
    recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 3, true, rolloutEvaluator(), { transposition: table }),
      state,
    );
    const snapshot = table.snapshot();
    // La recherche produit des feuilles rollout à profondeur 0, dont certaines
    // issues du cas de fusion (samples élargis) : les DEUX sites de stockage des
    // feuilles rollout — fusion et tirage frais — sont donc exercés ici.
    const depthZero = snapshot.filter((entry) => entry.depth === 0);
    expect(depthZero.length).toBeGreaterThan(0);
    expect(depthZero.some((entry) => (entry.samples ?? 0) > ROLLOUT_SAMPLES)).toBe(true);
    // Chaque entrée stockée, feuille rollout comprise, porte son tour intra-partie.
    for (const entry of snapshot) {
      expect(entry.pieces).toBeDefined();
    }
  });

  it('merges a pre-stored depth-0 entry with the fresh draws in closed form', () => {
    // Position à trois coups joués (joueur 1 à la main). Recherche à
    // profondeur 1 : chaque enfant est une feuille d'horizon. On pré-stocke
    // la moyenne de l'enfant { index: 5 } — exprimée du POINT DE VUE DU
    // JOUEUR AU TRAIT de l'enfant (joueur 0, opposé de la racine) — et on
    // laisse le moteur la fusionner avec ses 2 tirages frais, puis JOUER avec
    // la moyenne élargie ramenée à la perspective racine.
    const state = [4, 0, 2].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    const childState = ticTacToe.apply(state, { index: 5 });
    const childKey = ticTacToe.stateKey(childState);
    // Espion autour de l'évaluateur rollout : capture le tirage frais de
    // chaque état évalué avant toute fusion par le moteur.
    const run = () => {
      const table = createTranspositionTable();
      // Moyenne intrinsèque 0,5 vue par le joueur au trait dans l'enfant :
      // la racine (perspective opposée) la lit comme -0,5.
      table.store(childKey, { score: -0.5, depth: 0, exact: true, samples: 4 });
      const draws = new Map<string, number>();
      const inner = rolloutEvaluator();
      const spyEvaluator: Evaluator<TicTacToeState, TicTacToeMove> = {
        id: inner.id,
        samplesPerEvaluation: inner.samplesPerEvaluation,
        evaluate: (evaluated, ctx) => {
          const score = inner.evaluate(evaluated, ctx);
          draws.set(ticTacToe.stateKey(evaluated), score);
          return score;
        },
      };
      const { visits, values } = recordSearch(
        (recorder) =>
          // Sans élagage : chaque enfant de la racine est visité, garantie
          // que la feuille pré-stockée traverse le cas de fusion.
          createTicTacToeAgent(recorder, 1, false, spyEvaluator, {
            transposition: table,
          }),
        state,
      );
      const childVisit = visits.find(
        (visit) =>
          visit.kind === 'leaf' &&
          ticTacToe.stateKey(visit.state) === ticTacToe.stateKey(childState),
      );
      if (childVisit === undefined) {
        throw new Error('expected the pre-stored child to be visited as a leaf');
      }
      const fresh = draws.get(ticTacToe.stateKey(childState));
      if (fresh === undefined) {
        throw new Error('expected the spy evaluator to capture the fresh draw');
      }
      const played = values.find(
        (value) => value.nodeKey === childVisit.nodeKey && value.exact,
      )?.score;
      if (played === undefined) {
        throw new Error('expected an exact value event for the pre-stored child');
      }
      return { snapshot: table.snapshot(), fresh, played };
    };
    const first = run();
    const replay = run();
    // Le rng est semé : mêmes tirages, donc mêmes moyennes stockées et jouées.
    expect(replay.fresh).toBe(first.fresh);
    expect(replay.played).toBe(first.played);
    expect(replay.snapshot.filter((entry) => entry.depth === 0)).toEqual(
      first.snapshot.filter((entry) => entry.depth === 0),
    );
    // Formule fermée dans la perspective intrinsèque du joueur au trait :
    // (4·(-0,5) + 2·(-frais)) / 6, le tirage frais étant relatif à la racine
    // puis ramené à cette perspective par négation (4 + 2 = 6 tirages).
    const storedExpected = (4 * -0.5 + ROLLOUT_SAMPLES * -first.fresh) / (4 + ROLLOUT_SAMPLES);
    // … et le moteur ramène la moyenne stockée à la racine par négation.
    const playedExpected = -storedExpected;
    const merged = first.snapshot.find((entry) => entry.key === childKey);
    expect(merged?.samples).toBe(4 + ROLLOUT_SAMPLES);
    expect(merged?.score).toBe(storedExpected);
    expect(merged?.exact).toBe(true);
    // Le moteur joue la moyenne élargie ramenée à la racine, pas le tirage
    // frais isolé.
    expect(first.played).toBe(playedExpected);
  });

  it('never pollutes the root state with a depth-0 rollout entry', () => {
    const state = [4, 0, 2].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    const table = createTranspositionTable();
    recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 2, true, rolloutEvaluator(), { transposition: table }),
      state,
    );
    const rootKey = ticTacToe.stateKey(state);
    // La racine est déployée, jamais évaluée comme feuille : elle ne peut
    // porter qu'une entrée de frame complète (profondeur > 0), pas une
    // entrée profondeur 0 de cache-feuille.
    expect(table.snapshot().some((entry) => entry.key === rootKey && entry.depth === 0)).toBe(
      false,
    );
  });

  it('keeps leaf caching inert for evaluators without samplesPerEvaluation', () => {
    const state = [4, 0, 2].reduce(
      (current, index) => ticTacToe.apply(current, { index }),
      ticTacToe.initialState(),
    );
    const table = createTranspositionTable();
    recordSearch(
      (recorder) =>
        createTicTacToeAgent(recorder, 2, true, ticTacToeHeuristic, {
          transposition: table,
        }),
      state,
    );
    // Évaluateur heuristique : aucune entrée profondeur 0 n'est produite,
    // les feuilles d'horizon restent non cachées.
    expect(table.snapshot().some((entry) => entry.depth === 0)).toBe(false);
  });
});
