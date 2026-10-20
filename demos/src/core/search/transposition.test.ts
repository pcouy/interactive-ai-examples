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
  createReadOnlyOverlay,
  createTranspositionTable,
  isEntryUsable,
  type ProbedEntry,
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

function rootKeyOf<S>(visits: VisitEvent<S>[]): string {
  const rootKey = visits[0]?.nodeKey;
  if (rootKey === undefined) {
    throw new Error('expected at least one visit');
  }
  return rootKey;
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
  transposition?: TranspositionTable,
  strictPruning = false,
  rng?: () => number,
): SearchAgent<TicTacToeState, TicTacToeMove> {
  return createMinimaxAgent(ticTacToe, {
    depth,
    evaluator: ticTacToeHeuristic,
    prune,
    recorder,
    transposition,
    strictPruning,
    rng,
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
    const table = createTranspositionTable();
    expect(table.size()).toBe(0);
    expect(table.probe('missing')).toBeNull();
    const entry: TranspositionEntry = {
      score: 0.25,
      depth: 3,
      exact: true,
    };
    table.store('a', entry);
    expect(table.probe('a')).toEqual({ ...entry, turn: 0 });
    table.store('b', { score: -1, depth: 0, exact: false });
    expect(table.size()).toBe(2);
    expect(table.probe('b')?.score).toBe(-1);
    table.clear();
    expect(table.size()).toBe(0);
    expect(table.probe('a')).toBeNull();
    expect(table.probe('b')).toBeNull();
  });

  it('stamps stored entries with the current turn', () => {
    const table = createTranspositionTable();
    table.store('a', { score: 0.5, depth: 2, exact: true });
    expect(table.probe('a')?.turn).toBe(0);
    table.beginTurn();
    expect(table.currentTurn()).toBe(1);
    table.store('b', { score: -0.5, depth: 2, exact: true });
    expect(table.probe('b')?.turn).toBe(1);
    // Un stockage à profondeur égale écrase l'ancienne entrée et porte le
    // tour courant.
    table.store('a', { score: 0.75, depth: 2, exact: true });
    expect(table.probe('a')?.score).toBe(0.75);
    expect(table.probe('a')?.turn).toBe(1);
  });

  it('snapshots all entries sorted by key', () => {
    const table = createTranspositionTable();
    table.beginTurn();
    table.beginTurn();
    table.store('b', { score: 0.5, depth: 2, exact: true });
    table.store('a', { score: -0.25, depth: 3, exact: false });
    const snapshot = table.snapshot();
    expect(snapshot.map((row) => row.key)).toEqual(['a', 'b']);
    expect(snapshot[0]).toEqual({
      key: 'a',
      score: -0.25,
      depth: 3,
      exact: false,
      turn: 2,
    });
    expect(table.probe('missing')).toBeNull();
  });

  it('keeps the deeper entry when storing over an existing one', () => {
    const table = createTranspositionTable();
    table.store('k', { score: 1, depth: 2, exact: true });
    table.store('k', { score: 0, depth: 1, exact: true });
    expect(table.probe('k')?.score).toBe(1);
    table.store('k', { score: 0.5, depth: 2, exact: false });
    expect(table.probe('k')?.score).toBe(0.5);
    table.store('k', { score: 0.75, depth: 5, exact: false });
    expect(table.probe('k')?.score).toBe(0.75);
  });

  it('keeps an optional pieces count through probe and snapshot', () => {
    const table = createTranspositionTable();
    table.store('with', { score: 0.5, depth: 2, exact: true, pieces: 3 });
    expect(table.probe('with')?.pieces).toBe(3);
    const withEntry = table.snapshot().find((row) => row.key === 'with');
    expect(withEntry?.pieces).toBe(3);
  });

  it('leaves the pieces field absent on entries stored without it', () => {
    const table = createTranspositionTable();
    table.store('without', { score: -0.5, depth: 2, exact: true });
    expect(table.probe('without')?.pieces).toBeUndefined();
    const withoutEntry = table.snapshot().find((row) => row.key === 'without');
    if (withoutEntry === undefined) {
      throw new Error('expected the stored entry in the snapshot');
    }
    expect('pieces' in withoutEntry).toBe(false);
  });
});

describe('createReadOnlyOverlay', () => {
  it('exposes a real entry with turn -1', () => {
    const real = createTranspositionTable();
    real.store('k', { score: 0.5, depth: 3, exact: true });
    const overlay = createReadOnlyOverlay(real);
    expect(overlay.probe('k')).toEqual({ score: 0.5, depth: 3, exact: true, turn: -1 });
    expect(overlay.probe('missing')).toBeNull();
  });

  it('returns an overlay store first and leaves the real table unchanged', () => {
    const real = createTranspositionTable();
    real.store('k', { score: 0.1, depth: 2, exact: true });
    const overlay = createReadOnlyOverlay(real);
    const before = real.snapshot();
    overlay.store('k', { score: 0.9, depth: 2, exact: true });
    // L'entrée temporaire masque la réelle…
    expect(overlay.probe('k')?.score).toBe(0.9);
    // … sans modifier la table réelle.
    expect(real.snapshot()).toEqual(before);
    expect(real.probe('k')?.score).toBe(0.1);
  });

  it('prefers a deeper real entry over a shallower temp entry', () => {
    const real = createTranspositionTable();
    real.store('k', { score: 0.7, depth: 5, exact: true });
    const overlay = createReadOnlyOverlay(real);
    // L'entrée temporaire est strictement moins profonde : elle ne doit pas
    // masquer l'entrée réelle, exposée avec turn -1.
    overlay.store('k', { score: 0.1, depth: 2, exact: true });
    expect(overlay.probe('k')).toEqual({ score: 0.7, depth: 5, exact: true, turn: -1 });
  });

  it('prefers a deeper temp entry over a real entry', () => {
    const real = createTranspositionTable();
    real.store('k', { score: 0.1, depth: 2, exact: true });
    const overlay = createReadOnlyOverlay(real);
    overlay.store('k', { score: 0.8, depth: 5, exact: true });
    expect(overlay.probe('k')).toEqual({ score: 0.8, depth: 5, exact: true, turn: 0 });
  });

  it('lets a temp entry win at equal depth', () => {
    const real = createTranspositionTable();
    real.store('k', { score: 0.1, depth: 3, exact: true });
    const overlay = createReadOnlyOverlay(real);
    overlay.store('k', { score: 0.9, depth: 3, exact: true });
    expect(overlay.probe('k')).toEqual({ score: 0.9, depth: 3, exact: true, turn: 0 });
  });

  it('merges the snapshot with the deeper entry winning and temp at equal depth', () => {
    const real = createTranspositionTable();
    real.store('shallower-temp', { score: 0.7, depth: 5, exact: true });
    real.store('deeper-temp', { score: 0.1, depth: 2, exact: true });
    real.store('equal-depth', { score: 0.2, depth: 3, exact: true });
    const overlay = createReadOnlyOverlay(real);
    overlay.store('shallower-temp', { score: 0.1, depth: 2, exact: true });
    overlay.store('deeper-temp', { score: 0.8, depth: 4, exact: true });
    overlay.store('equal-depth', { score: 0.9, depth: 3, exact: true });
    const snapshot = overlay.snapshot();
    // L'entrée réelle plus profonde est conservée, avec turn -1.
    expect(snapshot.find((row) => row.key === 'shallower-temp')).toEqual({
      key: 'shallower-temp',
      score: 0.7,
      depth: 5,
      exact: true,
      turn: -1,
    });
    // L'entrée temporaire plus profonde l'emporte.
    expect(snapshot.find((row) => row.key === 'deeper-temp')?.score).toBe(0.8);
    expect(snapshot.find((row) => row.key === 'deeper-temp')?.turn).toBe(0);
    // À profondeur égale, l'entrée temporaire l'emporte.
    expect(snapshot.find((row) => row.key === 'equal-depth')?.score).toBe(0.9);
    expect(snapshot.find((row) => row.key === 'equal-depth')?.turn).toBe(0);
  });

  it('follows its own temp turn for currentTurn and beginTurn', () => {
    const real = createTranspositionTable();
    const overlay = createReadOnlyOverlay(real);
    expect(overlay.currentTurn()).toBe(0);
    real.beginTurn();
    real.beginTurn();
    // Le tour de la table réelle n'influence pas l'overlay.
    expect(overlay.currentTurn()).toBe(0);
    overlay.beginTurn();
    expect(overlay.currentTurn()).toBe(1);
  });

  it('clears only its temp table', () => {
    const real = createTranspositionTable();
    real.store('real-key', { score: 0.5, depth: 2, exact: true });
    const overlay = createReadOnlyOverlay(real);
    overlay.store('temp-key', { score: 0.7, depth: 2, exact: true });
    expect(overlay.size()).toBe(2);
    overlay.clear();
    expect(overlay.probe('temp-key')).toBeNull();
    // La table réelle reste intacte et reste visible en lecture.
    expect(overlay.probe('real-key')?.score).toBe(0.5);
    expect(overlay.size()).toBe(1);
    expect(real.snapshot()).toHaveLength(1);
  });

  it('merges snapshots with temp entries winning, sorted by key', () => {
    const real = createTranspositionTable();
    real.store('a', { score: 0.1, depth: 2, exact: true });
    real.store('b', { score: 0.2, depth: 2, exact: true });
    const overlay = createReadOnlyOverlay(real);
    overlay.store('b', { score: 0.8, depth: 3, exact: true });
    overlay.store('c', { score: 0.3, depth: 2, exact: true });
    const snapshot = overlay.snapshot();
    expect(snapshot.map((row) => row.key)).toEqual(['a', 'b', 'c']);
    expect(snapshot.find((row) => row.key === 'b')?.score).toBe(0.8);
    // Les entrées réelles exposées par l'overlay portent turn -1.
    expect(snapshot.find((row) => row.key === 'a')?.turn).toBe(-1);
  });

  it('counts unique keys across temp and real', () => {
    const real = createTranspositionTable();
    real.store('a', { score: 0.1, depth: 2, exact: true });
    real.store('b', { score: 0.2, depth: 2, exact: true });
    const overlay = createReadOnlyOverlay(real);
    overlay.store('b', { score: 0.8, depth: 3, exact: true });
    overlay.store('c', { score: 0.3, depth: 2, exact: true });
    expect(overlay.size()).toBe(3);
  });

  it('never treats an equal-depth real entry as current-turn', () => {
    const real = createTranspositionTable();
    real.store('equal', { score: 0.5, depth: 2, exact: true });
    real.store('deeper', { score: 0.5, depth: 4, exact: true });
    const overlay = createReadOnlyOverlay(real);
    const equal = overlay.probe('equal');
    const deeper = overlay.probe('deeper');
    if (equal === null || deeper === null) {
      throw new Error('expected both entries through the overlay');
    }
    // L'entrée réelle à profondeur égale porte turn -1 : isEntryUsable la refuse.
    expect(isEntryUsable(equal, 2, -Infinity, Infinity, true, overlay.currentTurn())).toBe(false);
    // L'entrée strictement plus profonde reste consommable.
    expect(isEntryUsable(deeper, 2, -Infinity, Infinity, true, overlay.currentTurn())).toBe(true);
  });
});

describe('isEntryUsable', () => {
  it('accepts a strictly deeper entry whatever the turn', () => {
    const entry: ProbedEntry = {
      score: 0.5,
      depth: 2,
      exact: true,
      turn: 0,
    };
    expect(isEntryUsable(entry, 1, -Infinity, Infinity, true, 5)).toBe(true);
    expect(isEntryUsable(entry, 1, -Infinity, Infinity, true, 0)).toBe(true);
  });

  it('accepts an equal-depth entry only when it comes from the current turn', () => {
    const entry: ProbedEntry = {
      score: 0.5,
      depth: 1,
      exact: true,
      turn: 3,
    };
    // À profondeur égale, seule une entrée du tour courant sert : les tours
    // précédents ne peuvent fournir que du strictement plus profond.
    expect(isEntryUsable(entry, 1, -Infinity, Infinity, true, 3)).toBe(true);
    expect(isEntryUsable(entry, 1, -Infinity, Infinity, true, 4)).toBe(false);
    expect(isEntryUsable(entry, 0, -Infinity, Infinity, true, 99)).toBe(true);
  });

  it('accepts an exact entry whatever the window when the depth condition holds', () => {
    const entry: ProbedEntry = {
      score: 0.5,
      depth: 3,
      exact: true,
      turn: 7,
    };
    expect(isEntryUsable(entry, 3, -Infinity, Infinity, true, 7)).toBe(true);
    expect(isEntryUsable(entry, 3, -Infinity, Infinity, true, 8)).toBe(false);
    expect(isEntryUsable(entry, 2, -0.9, 0.1, true, 8)).toBe(true);
    expect(isEntryUsable(entry, 2, 0.6, Infinity, false, 8)).toBe(true);
  });

  it('accepts a bound entry only when the window is compatible with the node direction', () => {
    const lower: ProbedEntry = {
      score: 0.3,
      depth: 3,
      exact: false,
      turn: 2,
    };
    expect(isEntryUsable(lower, 2, -Infinity, 0.2, true, 2)).toBe(true);
    expect(isEntryUsable(lower, 2, -Infinity, 0.3, true, 2)).toBe(true);
    expect(isEntryUsable(lower, 2, -Infinity, 0.4, true, 2)).toBe(false);
    expect(isEntryUsable(lower, 3, -Infinity, 0.2, true, 3)).toBe(false);
    expect(isEntryUsable(lower, 3, -Infinity, 0.2, true, 2)).toBe(true);
    const upper: ProbedEntry = {
      score: -0.3,
      depth: 3,
      exact: false,
      turn: 2,
    };
    expect(isEntryUsable(upper, 2, -0.2, Infinity, false, 2)).toBe(true);
    expect(isEntryUsable(upper, 2, -0.3, Infinity, false, 2)).toBe(true);
    expect(isEntryUsable(upper, 2, -0.4, Infinity, false, 2)).toBe(false);
    expect(isEntryUsable(upper, 3, -0.2, Infinity, false, 3)).toBe(false);
    expect(isEntryUsable(upper, 3, -0.2, Infinity, false, 2)).toBe(true);
  });

  it('rejects a bound entry exactly on the bound in strict mode', () => {
    // Le mode strict reflète la règle de coupure du moteur : seule une valeur
    // qui DÉPASSE strictement la borne est équivalente au résultat élagué.
    const atBeta: ProbedEntry = { score: 0.5, depth: 3, exact: false, turn: 2 };
    expect(isEntryUsable(atBeta, 2, -Infinity, 0.5, true, 2)).toBe(true);
    expect(isEntryUsable(atBeta, 2, -Infinity, 0.5, true, 2, false)).toBe(true);
    expect(isEntryUsable(atBeta, 2, -Infinity, 0.5, true, 2, true)).toBe(false);
    const atAlpha: ProbedEntry = { score: -0.5, depth: 3, exact: false, turn: 2 };
    expect(isEntryUsable(atAlpha, 2, -0.5, Infinity, false, 2)).toBe(true);
    expect(isEntryUsable(atAlpha, 2, -0.5, Infinity, false, 2, false)).toBe(true);
    expect(isEntryUsable(atAlpha, 2, -0.5, Infinity, false, 2, true)).toBe(false);
    // Une entrée exacte reste utilisable dans les deux modes, fenêtre ou non.
    const exact: ProbedEntry = { score: 0.5, depth: 3, exact: true, turn: 2 };
    expect(isEntryUsable(exact, 2, 0.5, Infinity, true, 2)).toBe(true);
    expect(isEntryUsable(exact, 2, 0.5, Infinity, true, 2, true)).toBe(true);
    expect(isEntryUsable(exact, 2, -Infinity, -0.5, false, 2, true)).toBe(true);
  });
});

describe('minimax with a transposition table', () => {
  it('visits strictly fewer nodes with ttHits > 0 and keeps the same score and bestMove', () => {
    for (const state of [ticTacToe.initialState(), midgameState()]) {
      const off = recordSearch((recorder) => createTicTacToeAgent(recorder, 5, true), state);
      const on = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 5, true, createTranspositionTable()),
        state,
      );
      // Au sein d'une même recherche, tous les stockages partagent le même
      // tour : les transpositions à profondeur égale sont donc consommées.
      const offStats = off.agent.stats();
      const onStats = on.agent.stats();
      expect(onStats.nodesVisited).toBeLessThan(offStats.nodesVisited);
      expect(onStats.ttHits).toBeGreaterThan(0);
      expect(onStats.ttMisses).toBeGreaterThan(0);
      expect(onStats.elapsedSteps).toBe(onStats.nodesVisited);
      expect(finalRootScore(on.values, rootKeyOf(on.visits))).toBe(
        finalRootScore(off.values, rootKeyOf(off.visits)),
      );
      expect(on.agent.bestMove()).toEqual(off.agent.bestMove());
    }
  });

  it('reuses equal-depth entries only within the same turn across two searches', () => {
    const shared = createTranspositionTable();
    const first = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true, shared),
      ticTacToe.initialState(),
    );
    const firstStats = first.agent.stats();
    expect(firstStats.nodesVisited).toBeGreaterThan(1);
    // Une ré-exécution sans nouveau tour : la racine n'est jamais résolue
    // depuis la table, elle explore toujours ses enfants. En revanche les
    // nœuds internes consomment les entrées à profondeur égale du même tour :
    // les neuf enfants de la racine sont frappés (tt-hit) et la sous-arborescence
    // sous ces frappes disparaît.
    const replay = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true, shared),
      ticTacToe.initialState(),
    );
    const replayStats = replay.agent.stats();
    const replayRootKey = rootKeyOf(replay.visits);
    const replayRootChildren = replay.visits.filter((visit) => visit.parentKey === replayRootKey);
    expect(replayRootChildren).toHaveLength(ticTacToe.legalMoves(ticTacToe.initialState()).length);
    expect(replayRootChildren.some((visit) => visit.kind === 'tt-hit')).toBe(true);
    expect(replayStats.nodesVisited).toBeGreaterThan(1);
    expect(replayStats.nodesVisited).toBeLessThan(firstStats.nodesVisited);
    expect(replayStats.ttHits).toBeGreaterThan(0);
    expect(finalRootScore(replay.values, replayRootKey)).toBe(
      finalRootScore(first.values, rootKeyOf(first.visits)),
    );
    expect(replay.agent.bestMove()).toEqual(first.agent.bestMove());
    // Après beginTurn, la même ré-exécution à profondeur égale ne peut plus
    // consommer les entrées du tour précédent : elle ré-explore tout, exactement
    // comme une recherche à table fraîche. À profondeur 4 depuis l'état initial
    // de tic-tac-toe aucun terminal n'est atteignable (victoire minimale en 5
    // plies) : aucune entrée ne peut donc porter une profondeur certifiée
    // STRICTEMENT supérieure, seule capable de traverser un changement de tour.
    shared.beginTurn();
    const stale = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true, shared),
      ticTacToe.initialState(),
    );
    expect(stale.agent.stats().nodesVisited).toBe(firstStats.nodesVisited);
    expect(finalRootScore(stale.values, rootKeyOf(stale.visits))).toBe(
      finalRootScore(first.values, rootKeyOf(first.visits)),
    );
    expect(stale.agent.bestMove()).toEqual(first.agent.bestMove());
  });

  it('is correct with and without pruning and matches the naive score', () => {
    const rootPlayer = ticTacToe.currentPlayer(ticTacToe.initialState());
    const expected = naiveMinimax(ticTacToe.initialState(), 5, rootPlayer);
    for (const prune of [false, true]) {
      // Le minimax pur (prune: false) départage désormais les ex æquo au
      // hasard : on fige la source pour que la table et le moteur sans table
      // retiennent le même coup (rng ≈ 1 conserve le premier ex æquo visité).
      const rng = () => 0.999999;
      const off = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 5, prune, undefined, false, rng),
        ticTacToe.initialState(),
      );
      const on = recordSearch(
        (recorder) =>
          createTicTacToeAgent(recorder, 5, prune, createTranspositionTable(), false, rng),
        ticTacToe.initialState(),
      );
      expect(finalRootScore(on.values, rootKeyOf(on.visits))).toBe(expected);
      expect(finalRootScore(off.values, rootKeyOf(off.visits))).toBe(expected);
      expect(on.agent.bestMove()).toEqual(off.agent.bestMove());
      expect(on.agent.stats().ttHits).toBeGreaterThan(0);
      expect(on.agent.stats().nodesVisited).toBeLessThan(off.agent.stats().nodesVisited);
    }
  });

  it('misses entries from a shallower search when searching deeper, then hits when searching shallower again', () => {
    const shared = createTranspositionTable();
    recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 2, true, shared),
      ticTacToe.initialState(),
    );
    const deep = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true, shared),
      ticTacToe.initialState(),
    );
    const rootKey = rootKeyOf(deep.visits);
    const deepStats = deep.agent.stats();
    // Les entrées de la recherche à 2 sont strictement moins profondes que le
    // budget des nœuds de la recherche à 4 : impossibles à consommer là-bas.
    // Les touches de la recherche profonde viennent de ses propres
    // transpositions, à profondeur égale et au même tour.
    expect(deepStats.ttMisses).toBeGreaterThan(0);
    expect(deepStats.ttHits).toBeGreaterThan(0);
    expect(finalRootScore(deep.values, rootKey)).toBe(naiveMinimax(ticTacToe.initialState(), 4, 0));
    expect(ticTacToe.legalMoves(ticTacToe.initialState())).toContainEqual(deep.agent.bestMove());
    const again = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 2, true, shared),
      ticTacToe.initialState(),
    );
    const againStats = again.agent.stats();
    // La racine de cette nouvelle recherche à 2 est toujours ré-explorée.
    // En revanche ses nœuds internes consomment les entrées de la recherche à 4
    // strictement plus profondes (même tour) : la valeur retournée est celle,
    // résolue, de la recherche profonde.
    expect(againStats.nodesVisited).toBeGreaterThan(1);
    expect(againStats.ttHits).toBeGreaterThan(0);
    expect(again.agent.bestMove()).toEqual(deep.agent.bestMove());
    // La valeur restituée est celle de la recherche profonde réutilisée ; ici
    // elle coïncide avec la valeur naïve à l'horizon 2 (partie indécise).
    expect(finalRootScore(again.values, rootKeyOf(again.visits))).toBe(
      finalRootScore(deep.values, rootKey),
    );
    expect(finalRootScore(again.values, rootKeyOf(again.visits))).toBe(
      naiveMinimax(ticTacToe.initialState(), 2, 0),
    );
  });

  it('keeps the same score and bestMove as the table-free engine over a random corpus', () => {
    for (const state of randomStates(ticTacToe, 30, {
      rng: createRng(CORPUS_SEED + 4),
      minPly: 0,
      maxPly: 6,
    })) {
      const off = recordSearch((recorder) => createTicTacToeAgent(recorder, 4, true), state);
      const on = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 4, true, createTranspositionTable()),
        state,
      );
      expect(finalRootScore(on.values, rootKeyOf(on.visits))).toBe(
        finalRootScore(off.values, rootKeyOf(off.visits)),
      );
      expect(on.agent.bestMove()).toEqual(off.agent.bestMove());
    }
  });

  it('stays exact in strict mode with a transposition table on corpus positions', () => {
    // L'élagage strict refuse les entrées exactement sur la borne : la table ne
    // doit jamais faire dévier la valeur racine de la référence naïve.
    const corpus = randomStates(ticTacToe, 2, {
      rng: createRng(CORPUS_SEED + 5),
      minPly: 2,
      maxPly: 4,
    });
    for (const state of corpus) {
      const rootPlayer = ticTacToe.currentPlayer(state);
      const expected = naiveMinimax(state, 4, rootPlayer);
      const strict = recordSearch(
        (recorder) => createTicTacToeAgent(recorder, 4, true, createTranspositionTable(), true),
        state,
      );
      expect(finalRootScore(strict.values, rootKeyOf(strict.visits))).toBe(expected);
      const move = strict.agent.bestMove();
      if (move !== null) {
        expect(naiveMinimax(ticTacToe.apply(state, move), 3, rootPlayer)).toBe(expected);
      }
    }
  });

  it('the root of a search is never answered from the table', () => {
    const reference = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true),
      ticTacToe.initialState(),
    );
    const shared = createTranspositionTable();
    recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true, shared),
      ticTacToe.initialState(),
    );
    // Table peuplée par une recherche identique au même tour : la racine doit
    // malgré tout être redéveloppée et choisir son coup via la recherche.
    const replay = recordSearch(
      (recorder) => createTicTacToeAgent(recorder, 4, true, shared),
      ticTacToe.initialState(),
    );
    const replayRootKey = rootKeyOf(replay.visits);
    expect(replay.visits[0]).toMatchObject({
      nodeKey: replayRootKey,
      parentKey: null,
      depth: 0,
      kind: 'expand',
    });
    const replayRootChildren = replay.visits.filter((visit) => visit.parentKey === replayRootKey);
    expect(replayRootChildren).toHaveLength(ticTacToe.legalMoves(ticTacToe.initialState()).length);
    expect(finalRootScore(replay.values, replayRootKey)).toBe(
      finalRootScore(reference.values, rootKeyOf(reference.visits)),
    );
    expect(replay.agent.bestMove()).toEqual(reference.agent.bestMove());
  });
});
