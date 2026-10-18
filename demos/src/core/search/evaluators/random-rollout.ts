import type { Evaluator, GameOutcome, PlayerId } from '../../contracts.js';

const DEFAULT_ROLLOUT_SEED = 4157;
const DEFAULT_SAMPLES = 1;
const DEFAULT_MAX_PLIES = 200;

function createMulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function outcomeValue(outcome: GameOutcome, rootPlayer: PlayerId): number {
  if (outcome === rootPlayer) {
    return 1;
  }
  if (outcome === 'draw' || outcome === null) {
    return 0;
  }
  return -1;
}

export function createRandomRolloutEvaluator<S, M>(options?: {
  rng?: () => number;
  samples?: number;
  maxPlies?: number;
}): Evaluator<S, M> {
  const rng = options?.rng ?? createMulberry32(DEFAULT_ROLLOUT_SEED);
  const sampleCount = Math.max(1, Math.floor(options?.samples ?? DEFAULT_SAMPLES));
  const maxPlies = Math.max(1, Math.floor(options?.maxPlies ?? DEFAULT_MAX_PLIES));
  return {
    id: 'random-rollout',
    // Le moteur exploite ce compte pour fusionner les moyennes de rollout
    // des feuilles d'horizon dans la table de transposition (profondeur 0).
    samplesPerEvaluation: sampleCount,
    evaluate(state, ctx) {
      const game = ctx.game;
      const rootPlayer = ctx.rootPlayer;
      let total = 0;
      for (let sample = 0; sample < sampleCount; sample += 1) {
        let current = state;
        let outcome: GameOutcome = null;
        for (let ply = 0; ply < maxPlies && outcome === null; ply += 1) {
          const terminal = game.winner(current);
          if (terminal !== null) {
            outcome = terminal;
            break;
          }
          const moves = game.legalMoves(current);
          if (moves.length === 0) {
            break;
          }
          const index = Math.min(moves.length - 1, Math.floor(rng() * moves.length));
          current = game.apply(current, moves[index]);
        }
        if (outcome === null) {
          outcome = game.winner(current);
        }
        total += outcomeValue(outcome, rootPlayer);
      }
      return total / sampleCount;
    },
  };
}
