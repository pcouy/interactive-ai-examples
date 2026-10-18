import {
  DRAW_SCORE,
  type EvalContext,
  type Evaluator,
  type Game,
  LOSS_SCORE,
  type SearchAgent,
  type SearchStats,
  type SearchStatus,
  type TreeRecorder,
  type ValueEvent,
  type VisitEvent,
  type VisitKind,
  WIN_SCORE,
} from '../contracts.js';
import { isEntryUsable, type TranspositionTable } from './transposition.js';

export interface MinimaxOptions<S, M> {
  readonly depth: number;
  readonly evaluator: Evaluator<S, M>;
  readonly prune: boolean;
  readonly recorder: TreeRecorder<S>;
  readonly transposition?: TranspositionTable<M>;
}

interface Frame<S, M> {
  key: string;
  state: S;
  move: M | null;
  parentKey: string | null;
  depth: number;
  remaining: number;
  maximizing: boolean;
  alpha: number;
  beta: number;
  moves: readonly M[];
  nextChild: number;
  cut: boolean;
  bestScore: number;
  bestMove: M | null;
}

type NodeResolution =
  | { readonly expanded: true }
  | { readonly expanded: false; readonly score: number; readonly exact: boolean };

export function createMinimaxAgent<S, M>(
  game: Game<S, M>,
  options: MinimaxOptions<S, M>,
): SearchAgent<S, M> {
  const maxDepth = options.depth;
  const evaluator = options.evaluator;
  const prune = options.prune;
  const recorder = options.recorder;
  const transposition = options.transposition ?? null;

  let iterator: Generator<void, void, void> | null = null;
  let finished = false;
  let rootMove: M | null = null;

  let nodesVisited = 0;
  let nodesPruned = 0;
  let ttHits = 0;
  let ttMisses = 0;
  let leavesEvaluated = 0;
  let steps = 0;

  const emitVisit = (event: VisitEvent<S>): void => {
    nodesVisited += 1;
    recorder.onVisit(event);
  };

  const emitValue = (event: ValueEvent): void => {
    recorder.onValue(event);
  };

  function makeIterator(rootState: S): Generator<void, void, void> {
    const rootPlayer = game.currentPlayer(rootState);
    const ctx: EvalContext<S, M> = { rootPlayer, game };
    const frames: Frame<S, M>[] = [];
    let rootResultMove: M | null = null;

    const ttKey = (state: S): string => `${rootPlayer}:${game.stateKey(state)}`;

    const createFrame = (
      state: S,
      move: M | null,
      parentKey: string | null,
      depth: number,
      alpha: number,
      beta: number,
    ): Frame<S, M> => ({
      key: game.stateKey(state),
      state,
      move,
      parentKey,
      depth,
      remaining: maxDepth - depth,
      maximizing: game.currentPlayer(state) === rootPlayer,
      alpha,
      beta,
      moves: [],
      nextChild: 0,
      cut: false,
      bestScore: 0,
      bestMove: null,
    });

    const visitEvent = (frame: Frame<S, M>, kind: VisitKind): VisitEvent<S> => ({
      nodeKey: frame.key,
      parentKey: frame.parentKey,
      move: frame.move,
      state: frame.state,
      depth: frame.depth,
      kind,
    });

    const visitNode = (frame: Frame<S, M>): NodeResolution => {
      const outcome = game.winner(frame.state);
      if (outcome !== null) {
        const score =
          outcome === 'draw' ? DRAW_SCORE : outcome === rootPlayer ? WIN_SCORE : LOSS_SCORE;
        emitVisit(visitEvent(frame, 'terminal'));
        emitValue({ nodeKey: frame.key, score, depth: frame.remaining, exact: true });
        return { expanded: false, score, exact: true };
      }
      frame.moves = game.legalMoves(frame.state);
      if (transposition !== null) {
        const entry = transposition.probe(ttKey(frame.state));
        if (
          entry !== null &&
          isEntryUsable(entry, frame.remaining, frame.alpha, frame.beta, frame.maximizing)
        ) {
          ttHits += 1;
          emitVisit(visitEvent(frame, 'tt-hit'));
          emitValue({
            nodeKey: frame.key,
            score: entry.score,
            depth: frame.remaining,
            exact: entry.exact,
          });
          if (frame.parentKey === null) {
            rootResultMove = entry.bestMove;
          }
          return { expanded: false, score: entry.score, exact: entry.exact };
        }
        ttMisses += 1;
      }
      if (frame.moves.length === 0 || frame.remaining <= 0) {
        const score = evaluator.evaluate(frame.state, ctx);
        leavesEvaluated += 1;
        emitVisit(visitEvent(frame, 'leaf'));
        emitValue({ nodeKey: frame.key, score, depth: frame.remaining, exact: true });
        return { expanded: false, score, exact: true };
      }
      frame.bestScore = frame.maximizing ? -Infinity : Infinity;
      emitVisit(visitEvent(frame, 'expand'));
      return { expanded: true };
    };

    const emitPruned = (parent: Frame<S, M>, move: M, childState: S, childDepth: number): void => {
      emitVisit({
        nodeKey: game.stateKey(childState),
        parentKey: parent.key,
        move,
        state: childState,
        depth: childDepth,
        kind: 'pruned',
      });
      nodesPruned += 1;
    };

    const accumulate = (frame: Frame<S, M>, move: M, childScore: number): void => {
      const improves = frame.maximizing
        ? childScore > frame.bestScore
        : childScore < frame.bestScore;
      if (!improves) {
        return;
      }
      frame.bestScore = childScore;
      frame.bestMove = move;
      if (frame.maximizing) {
        if (childScore > frame.alpha) {
          frame.alpha = childScore;
        }
      } else if (childScore < frame.beta) {
        frame.beta = childScore;
      }
      emitValue({ nodeKey: frame.key, score: childScore, depth: frame.remaining, exact: false });
      if (prune) {
        if (frame.maximizing ? frame.bestScore >= frame.beta : frame.bestScore <= frame.alpha) {
          frame.cut = true;
        }
      }
    };

    const completeFrame = (frame: Frame<S, M>): void => {
      frames.pop();
      const exact = !frame.cut;
      emitValue({ nodeKey: frame.key, score: frame.bestScore, depth: frame.remaining, exact });
      if (transposition !== null) {
        transposition.store(ttKey(frame.state), {
          score: frame.bestScore,
          depth: frame.remaining,
          exact,
          bestMove: frame.bestMove,
        });
      }
      if (frame.parentKey === null) {
        rootResultMove = frame.bestMove;
        return;
      }
      const parent = frames.at(-1);
      if (parent) {
        accumulate(parent, frame.move as M, frame.bestScore);
      }
    };

    const settleUp = (): void => {
      for (;;) {
        const frame = frames.at(-1);
        if (!frame || frame.nextChild < frame.moves.length) {
          return;
        }
        completeFrame(frame);
      }
    };

    function* run(): Generator<void, void, void> {
      const root = createFrame(rootState, null, null, 0, -Infinity, Infinity);
      const rootResolution = visitNode(root);
      if (rootResolution.expanded) {
        frames.push(root);
        yield;
      } else {
        rootMove = rootResultMove;
        return;
      }
      for (;;) {
        const frame = frames.at(-1);
        if (!frame) {
          break;
        }
        if (frame.cut) {
          const move = frame.moves[frame.nextChild];
          const childState = game.apply(frame.state, move);
          emitPruned(frame, move, childState, frame.depth + 1);
          frame.nextChild += 1;
          settleUp();
          if (frames.length === 0) {
            break;
          }
          yield;
          continue;
        }
        const move = frame.moves[frame.nextChild];
        frame.nextChild += 1;
        const childState = game.apply(frame.state, move);
        const child = createFrame(
          childState,
          move,
          frame.key,
          frame.depth + 1,
          frame.alpha,
          frame.beta,
        );
        const resolution = visitNode(child);
        if (resolution.expanded) {
          frames.push(child);
        } else {
          accumulate(frame, move, resolution.score);
          settleUp();
        }
        if (frames.length === 0) {
          break;
        }
        yield;
      }
      rootMove = rootResultMove;
    }

    return run();
  }

  return {
    start(state: S): void {
      iterator?.return();
      iterator = makeIterator(state);
      finished = false;
      rootMove = null;
      nodesVisited = 0;
      nodesPruned = 0;
      ttHits = 0;
      ttMisses = 0;
      leavesEvaluated = 0;
      steps = 0;
    },
    step(): SearchStatus {
      if (iterator === null || finished) {
        return 'done';
      }
      steps += 1;
      const result = iterator.next();
      if (result.done) {
        finished = true;
        return 'done';
      }
      return 'running';
    },
    bestMove(): M | null {
      return finished ? rootMove : null;
    },
    stats(): SearchStats {
      return {
        nodesVisited,
        nodesPruned,
        ttHits,
        ttMisses,
        leavesEvaluated,
        elapsedSteps: steps,
      };
    },
  };
}
