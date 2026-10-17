import { describe, expect, it } from 'vitest';
import type { Game, PlayerId } from '../contracts.js';
import { createRng, pick, type Rng, randomPlayout, randomStates } from './playouts.js';

export interface GameContractOptions {
  readonly seed?: number;
  readonly corpusSize?: number;
  readonly minPly?: number;
  readonly maxPly?: number;
}

function isPlayerId(value: unknown): value is PlayerId {
  return value === 0 || value === 1;
}

export function describeGameContract<S, M>(
  game: Game<S, M>,
  name: string,
  options: GameContractOptions = {},
): void {
  const seed = options.seed ?? 987654321;
  const corpusSize = options.corpusSize ?? 60;
  const minPly = options.minPly ?? 0;
  const maxPly = options.maxPly ?? 12;

  describe(`game contract — ${name}`, () => {
    const rng: Rng = createRng(seed);
    const states = randomStates(game, corpusSize, { rng, minPly, maxPly });

    it('exposes a valid current player that alternates on every move', () => {
      for (const state of states) {
        expect(isPlayerId(game.currentPlayer(state))).toBe(true);
        for (const move of game.legalMoves(state)) {
          expect(game.currentPlayer(game.apply(state, move))).not.toBe(game.currentPlayer(state));
        }
      }
    });

    it('inverts apply with undo', () => {
      for (const state of states) {
        for (const move of game.legalMoves(state)) {
          expect(game.undo(game.apply(state, move), move)).toEqual(state);
        }
      }
    });

    it('never mutates the source state in apply', () => {
      for (const state of states) {
        const snapshot = structuredClone(state);
        for (const move of game.legalMoves(state)) {
          game.apply(state, move);
        }
        expect(state).toEqual(snapshot);
      }
    });

    it('produces distinct successor keys for distinct legal moves', () => {
      for (const state of states) {
        const moves = game.legalMoves(state);
        const keys = new Set(moves.map((move) => game.stateKey(game.apply(state, move))));
        expect(keys.size).toBe(moves.length);
      }
    });

    it('keeps terminal detection consistent', () => {
      for (const state of states) {
        const outcome = game.winner(state);
        if (game.legalMoves(state).length === 0) {
          expect(outcome).not.toBeNull();
        }
        if (outcome !== null && outcome !== 'draw') {
          expect(isPlayerId(outcome)).toBe(true);
          expect(outcome).not.toBe(game.currentPlayer(state));
        }
      }
    });

    it('always terminates random playouts', () => {
      const playoutRng = createRng(seed ^ 0x5f3759df);
      for (let index = 0; index < 25; index += 1) {
        const finalState = randomPlayout(game, game.initialState(), playoutRng);
        expect(game.winner(finalState)).not.toBeNull();
      }
    });

    it('derives a stable stateKey, distinct along a trajectory', () => {
      const keysA = states.map((state) => game.stateKey(state));
      const keysB = states.map((state) => game.stateKey(state));
      expect(keysA).toEqual(keysB);

      const pathRng = createRng(seed ^ 0x9e3779b9);
      const pathKeys: string[] = [];
      let current = game.initialState();
      while (game.winner(current) === null && game.legalMoves(current).length > 0) {
        pathKeys.push(game.stateKey(current));
        current = game.apply(current, pick(pathRng, game.legalMoves(current)));
      }
      expect(new Set(pathKeys).size).toBe(pathKeys.length);
    });

    it('exposes a boardView consistent with the grid', () => {
      for (const state of states) {
        const view = game.boardView(state);
        expect(view.rows).toBeGreaterThan(0);
        expect(view.cols).toBeGreaterThan(0);
        expect(view.cells.length).toBe(view.rows);
        for (const row of view.cells) {
          expect(row.length).toBe(view.cols);
          for (const cell of row) {
            expect(cell === null || isPlayerId(cell)).toBe(true);
          }
        }
      }
    });
  });
}
