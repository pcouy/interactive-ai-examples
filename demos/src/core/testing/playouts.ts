import type { Game } from '../contracts.js';

export type Rng = () => number;

export function createRng(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomInt(rng: Rng, bound: number): number {
  return Math.min(bound - 1, Math.floor(rng() * bound));
}

export function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[randomInt(rng, items.length)];
}

export function playRandomMove<S, M>(game: Game<S, M>, state: S, rng: Rng): S {
  return game.apply(state, pick(rng, game.legalMoves(state)));
}

export function randomPlayout<S, M>(game: Game<S, M>, state: S, rng: Rng, maxPlies = 1000): S {
  let current = state;
  for (let ply = 0; ply < maxPlies; ply += 1) {
    if (game.winner(current) !== null) return current;
    const moves = game.legalMoves(current);
    if (moves.length === 0) return current;
    current = game.apply(current, pick(rng, moves));
  }
  return current;
}

export interface RandomStateOptions {
  readonly rng: Rng;
  readonly minPly?: number;
  readonly maxPly?: number;
}

export function randomState<S, M>(game: Game<S, M>, options: RandomStateOptions): S {
  const minPly = options.minPly ?? 0;
  const maxPly = options.maxPly ?? minPly;
  const target = minPly + randomInt(options.rng, maxPly - minPly + 1);
  let current = game.initialState();
  for (let ply = 0; ply < target; ply += 1) {
    if (game.winner(current) !== null) break;
    const moves = game.legalMoves(current);
    if (moves.length === 0) break;
    current = game.apply(current, pick(options.rng, moves));
  }
  return current;
}

export function randomStates<S, M>(
  game: Game<S, M>,
  count: number,
  options: RandomStateOptions,
): S[] {
  return Array.from({ length: count }, () => randomState(game, options));
}
