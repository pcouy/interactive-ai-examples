import { describe, expect, it } from 'vitest';
import { noopRecorder, type PlayerKind, type SearchAgent } from '../../core/contracts.js';
import {
  type TicTacToeMove,
  type TicTacToeState,
  ticTacToe,
} from '../../core/games/tic-tac-toe.js';
import { createRng, randomStates } from '../../core/testing/playouts.js';
import {
  CONNECT4_GAME_CONFIG,
  createGameAgent,
  parseEvaluatorMode,
  parsePlayersConfig,
  sameMove,
  sanitizeDepth,
  sanitizeSamples,
  TIC_TAC_TOE_GAME_CONFIG,
} from './create-game-component.js';

const MAPPING_SEED = 20261019;
const AGENT_BUDGET = 2_000_000;

function dropConnect4(columns: readonly number[]) {
  let state = CONNECT4_GAME_CONFIG.game.initialState();
  for (const col of columns) {
    state = CONNECT4_GAME_CONFIG.game.apply(state, { col });
  }
  return state;
}

function playTicTacToe(indices: readonly number[]): TicTacToeState {
  let state = ticTacToe.initialState();
  for (const index of indices) {
    state = ticTacToe.apply(state, { index });
  }
  return state;
}

function runAgentToDone<S, M>(agent: SearchAgent<S, M>, state: S): M | null {
  agent.start(state);
  for (let budget = AGENT_BUDGET; budget > 0; budget -= 1) {
    if (agent.step() === 'done') {
      break;
    }
  }
  return agent.bestMove();
}

describe('tic-tac-toe cell mapping', () => {
  const config = TIC_TAC_TOE_GAME_CONFIG;

  it('makes every empty cell clickable, in row-major order, on the initial board', () => {
    const cells = config.clickableCells(config.game.initialState());
    const expected = Array.from({ length: 9 }, (_, index) => ({
      row: Math.floor(index / 3),
      col: index % 3,
    }));
    expect(cells).toEqual(expected);
  });

  it('keeps only the empty cells clickable on a played board', () => {
    const state = playTicTacToe([0, 4, 2]);
    expect(config.clickableCells(state)).toEqual([
      { row: 0, col: 1 },
      { row: 1, col: 0 },
      { row: 1, col: 2 },
      { row: 2, col: 0 },
      { row: 2, col: 1 },
      { row: 2, col: 2 },
    ]);
  });

  it('maps every clickable cell to exactly one legal move, exhaustively', () => {
    const states = randomStates(ticTacToe, 40, {
      rng: createRng(MAPPING_SEED),
      minPly: 0,
      maxPly: 8,
    });
    for (const state of states) {
      const clickable = config.clickableCells(state);
      const moves = ticTacToe.legalMoves(state);
      const seen = new Set<string>();
      for (const cell of clickable) {
        const move = config.moveForCell(cell.row, cell.col, state);
        expect(move).not.toBeNull();
        const moveValue = move as TicTacToeMove;
        expect(moves.some((legal) => sameMove(legal, moveValue))).toBe(true);
        seen.add(String(moveValue.index));
      }
      expect(seen.size).toBe(moves.length);
    }
  });

  it('rejects every non-clickable cell and out-of-range cells', () => {
    const state = playTicTacToe([4]);
    const clickable = config.clickableCells(state);
    for (let row = 0; row < 3; row += 1) {
      for (let col = 0; col < 3; col += 1) {
        const isClickable = clickable.some((cell) => cell.row === row && cell.col === col);
        const move = config.moveForCell(row, col, state);
        if (isClickable) {
          expect(move).not.toBeNull();
        } else {
          expect(move).toBeNull();
        }
      }
    }
    const fresh = ticTacToe.initialState();
    expect(config.moveForCell(-1, 0, fresh)).toBeNull();
    expect(config.moveForCell(0, 3, fresh)).toBeNull();
    expect(config.moveForCell(3, 0, fresh)).toBeNull();
    expect(config.moveForCell(-1, -1, fresh)).toBeNull();
  });
});

describe('connect4 cell mapping', () => {
  const config = CONNECT4_GAME_CONFIG;

  it('makes the whole column clickable for every non-full column', () => {
    const cells = config.clickableCells(config.game.initialState());
    expect(cells).toHaveLength(42);
    expect(cells.every((cell) => cell.row >= 0 && cell.row < 6)).toBe(true);
  });

  it('excludes full columns entirely and keeps the rest clickable', () => {
    const state = dropConnect4([0, 0, 0, 0, 0, 0]);
    const cells = config.clickableCells(state);
    expect(cells).toHaveLength(36);
    expect(cells.every((cell) => cell.col !== 0)).toBe(true);
    const oneBeforeFull = dropConnect4([3, 3, 3, 3, 3]);
    expect(config.clickableCells(oneBeforeFull)).toHaveLength(42);
  });

  it('reduces a click on any row to the column move', () => {
    const state = dropConnect4([2]);
    for (let row = 0; row < 6; row += 1) {
      expect(config.moveForCell(row, 2, state)).toEqual({ col: 2 });
    }
  });

  it('maps every clickable cell to a legal move, exhaustively, over random playouts', () => {
    const states = randomStates(CONNECT4_GAME_CONFIG.game, 24, {
      rng: createRng(MAPPING_SEED + 1),
      minPly: 0,
      maxPly: 20,
    });
    for (const state of states) {
      const clickable = config.clickableCells(state);
      const moves = CONNECT4_GAME_CONFIG.game.legalMoves(state);
      const legalCols = new Set(moves.map((move) => move.col));
      const clickableCols = new Set(clickable.map((cell) => cell.col));
      expect([...clickableCols].sort((a, b) => a - b)).toEqual(
        [...legalCols].sort((a, b) => a - b),
      );
      for (const cell of clickable) {
        const move = config.moveForCell(cell.row, cell.col, state);
        expect(move).toEqual({ col: cell.col });
      }
    }
  });

  it('rejects clicks on full columns and out-of-range columns', () => {
    const state = dropConnect4([5, 5, 5, 5, 5, 5]);
    for (let row = 0; row < 6; row += 1) {
      expect(config.moveForCell(row, 5, state)).toBeNull();
    }
    const fresh = CONNECT4_GAME_CONFIG.game.initialState();
    expect(config.moveForCell(0, -1, fresh)).toBeNull();
    expect(config.moveForCell(0, 7, fresh)).toBeNull();
  });
});

describe('parsePlayersConfig', () => {
  it('parses the short form', () => {
    expect(parsePlayersConfig('["human","agent"]')).toEqual(['human', 'agent']);
    expect(parsePlayersConfig('["agent","human"]')).toEqual(['agent', 'human']);
  });

  it('parses the full PlayerConfig objects, ignoring unknown extra fields', () => {
    const value = '[{ "kind": "agent", "stepBudget": 4 }, { "kind": "human", "agent": null }]';
    expect(parsePlayersConfig(value)).toEqual(['agent', 'human']);
  });

  it('falls back per slot on missing or invalid entries', () => {
    expect(parsePlayersConfig('["agent"]')).toEqual(['agent', 'agent']);
    expect(parsePlayersConfig('["robot","agent"]')).toEqual(['human', 'agent']);
    expect(parsePlayersConfig('["human","robot","human"]')).toEqual(['human', 'agent']);
  });

  it('falls back entirely on invalid JSON, non-arrays and missing values', () => {
    const fallback: readonly PlayerKind[] = ['agent', 'agent'];
    expect(parsePlayersConfig('not json', ['agent', 'agent'])).toEqual(fallback);
    expect(parsePlayersConfig('"human"', ['agent', 'agent'])).toEqual(fallback);
    expect(parsePlayersConfig('42', ['agent', 'agent'])).toEqual(fallback);
    expect(parsePlayersConfig('["human","agent"]')).toEqual(['human', 'agent']);
  });
});

describe('parseEvaluatorMode', () => {
  it('recognizes the three documented modes', () => {
    expect(parseEvaluatorMode('windows')).toBe('windows');
    expect(parseEvaluatorMode('rollout')).toBe('rollout');
    expect(parseEvaluatorMode('random')).toBe('random');
  });

  it('falls back to windows for unknown or empty values', () => {
    expect(parseEvaluatorMode('neural')).toBe('windows');
    expect(parseEvaluatorMode('')).toBe('windows');
    expect(parseEvaluatorMode('WINDOWS')).toBe('windows');
  });
});

describe('sanitizeDepth', () => {
  it('clamps to at least one ply and drops decimals', () => {
    expect(sanitizeDepth(6, 4)).toBe(6);
    expect(sanitizeDepth(0, 4)).toBe(1);
    expect(sanitizeDepth(-2, 4)).toBe(1);
    expect(sanitizeDepth(3.7, 4)).toBe(3);
    expect(sanitizeDepth(Number.NaN, 4)).toBe(4);
  });
});

describe('sanitizeSamples', () => {
  it('clamps to at least one sample and drops decimals', () => {
    expect(sanitizeSamples(8, 1)).toBe(8);
    expect(sanitizeSamples(0, 1)).toBe(1);
    expect(sanitizeSamples(-3, 1)).toBe(1);
    expect(sanitizeSamples(4.9, 1)).toBe(4);
    expect(sanitizeSamples(Number.NaN, 1)).toBe(1);
  });
});

describe('sameMove', () => {
  it('compares structurally identical moves', () => {
    expect(sameMove({ index: 3 }, { index: 3 })).toBe(true);
    expect(sameMove({ col: 5 }, { col: 5 })).toBe(true);
    expect(sameMove({ index: 3 }, { index: 4 })).toBe(false);
    expect(sameMove({ col: 0 }, { col: 1 })).toBe(false);
  });
});

describe('createGameAgent', () => {
  it('returns a random baseline agent for the random mode', () => {
    const agent = createGameAgent(ticTacToe, {
      mode: 'random',
      depth: 6,
      fallbackDepth: 6,
      transposition: false,
      recorder: noopRecorder,
      heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
    });
    const move = runAgentToDone(agent, ticTacToe.initialState());
    expect(move).not.toBeNull();
    expect(
      ticTacToe
        .legalMoves(ticTacToe.initialState())
        .some((legal) => sameMove(legal, move as TicTacToeMove)),
    ).toBe(true);
    expect(agent.stats().nodesVisited).toBe(1);
  });

  it('runs a windows-based search agent that returns a legal best move', () => {
    const state = playTicTacToe([4]);
    const agent = createGameAgent(ticTacToe, {
      mode: 'windows',
      depth: 6,
      fallbackDepth: 6,
      transposition: false,
      recorder: noopRecorder,
      heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
    });
    const move = runAgentToDone(agent, state);
    expect(move).not.toBeNull();
    expect(
      ticTacToe.legalMoves(state).some((legal) => sameMove(legal, move as TicTacToeMove)),
    ).toBe(true);
    expect(agent.stats().nodesVisited).toBeGreaterThan(0);
  });

  it('runs a rollout-based search agent that returns a legal best move', () => {
    const state = playTicTacToe([0, 4]);
    const agent = createGameAgent(ticTacToe, {
      mode: 'rollout',
      depth: 4,
      fallbackDepth: 6,
      transposition: false,
      recorder: noopRecorder,
      heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
    });
    const move = runAgentToDone(agent, state);
    expect(move).not.toBeNull();
    expect(
      ticTacToe.legalMoves(state).some((legal) => sameMove(legal, move as TicTacToeMove)),
    ).toBe(true);
  });

  it('supports the transposition table flag on the engine', () => {
    const state = playTicTacToe([0, 4, 8]);
    const agent = createGameAgent(ticTacToe, {
      mode: 'windows',
      depth: 6,
      fallbackDepth: 6,
      transposition: true,
      recorder: noopRecorder,
      heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
    });
    const move = runAgentToDone(agent, state);
    expect(move).not.toBeNull();
    expect(agent.stats().ttMisses).toBeGreaterThan(0);
  });

  it('plays a full connect4 search agent turn within a bounded step budget', () => {
    const state = dropConnect4([3, 3]);
    const agent = createGameAgent(CONNECT4_GAME_CONFIG.game, {
      mode: 'windows',
      depth: 4,
      fallbackDepth: 4,
      transposition: false,
      recorder: noopRecorder,
      heuristic: CONNECT4_GAME_CONFIG.heuristic,
    });
    const move = runAgentToDone(agent, state);
    expect(move).not.toBeNull();
    expect(
      CONNECT4_GAME_CONFIG.game
        .legalMoves(state)
        .some((legal) => sameMove(legal, move as { col: number })),
    ).toBe(true);
  });
});
