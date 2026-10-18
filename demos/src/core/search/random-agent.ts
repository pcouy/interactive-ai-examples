import type { Game, SearchAgent, SearchStats, SearchStatus } from '../contracts.js';

const DEFAULT_AGENT_SEED = 6203;

function createMulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function createRandomAgent<S, M>(
  game: Game<S, M>,
  options?: { rng?: () => number },
): SearchAgent<S, M> {
  const rng = options?.rng ?? createMulberry32(DEFAULT_AGENT_SEED);
  let pendingMove: M | null = null;
  let stepped = false;

  const pickMove = (state: S): M | null => {
    const moves = game.legalMoves(state);
    if (moves.length === 0) {
      return null;
    }
    return moves[Math.min(moves.length - 1, Math.floor(rng() * moves.length))];
  };

  return {
    start(state: S): void {
      pendingMove = pickMove(state);
      stepped = false;
    },
    step(): SearchStatus {
      stepped = true;
      return 'done';
    },
    bestMove(): M | null {
      return stepped ? pendingMove : null;
    },
    stats(): SearchStats {
      return {
        nodesVisited: 1,
        nodesPruned: 0,
        ttHits: 0,
        ttMisses: 0,
        leavesEvaluated: 0,
        elapsedSteps: 1,
      };
    },
  };
}
