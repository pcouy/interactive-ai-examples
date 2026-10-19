import { describe, expect, it } from 'vitest';
import { type Connect4State, connect4 } from '../games/connect4.js';
import { type TicTacToeState, ticTacToe } from '../games/tic-tac-toe.js';
import { createRng, randomStates } from '../testing/playouts.js';
import { createRandomAgent } from './random-agent.js';

const CORPUS_SEED = 20261017;

function ticTacToeCorpus(rngSeed: number): TicTacToeState[] {
  return randomStates(ticTacToe, 40, { rng: createRng(rngSeed), minPly: 0, maxPly: 8 });
}

function connect4Corpus(rngSeed: number): Connect4State[] {
  return randomStates(connect4, 30, { rng: createRng(rngSeed), minPly: 0, maxPly: 20 });
}

const UNIFORM_SEED = 20261019;
const UNIFORM_DRAWS = 900;
const UNIFORM_MEAN = UNIFORM_DRAWS / 9;
const UNIFORM_SIGMA = Math.sqrt(UNIFORM_DRAWS * (1 / 9) * (8 / 9));
const UNIFORM_MARGIN = 4.5 * UNIFORM_SIGMA;

function requireMove<M>(move: M | null): M {
  if (move === null) {
    throw new Error('expected search to be done with a move');
  }
  return move;
}

describe('createRandomAgent', () => {
  it('always plays a legal move over random tic-tac-toe and connect-four corpora', () => {
    for (const state of ticTacToeCorpus(CORPUS_SEED)) {
      const agent = createRandomAgent(ticTacToe, { rng: createRng(1) });
      agent.start(state);
      expect(agent.step()).toBe('done');
      const move = agent.bestMove();
      expect(move).not.toBeNull();
      expect(ticTacToe.legalMoves(state)).toContainEqual(move);
    }
    for (const state of connect4Corpus(CORPUS_SEED)) {
      const agent = createRandomAgent(connect4, { rng: createRng(2) });
      agent.start(state);
      expect(agent.step()).toBe('done');
      const move = agent.bestMove();
      expect(move).not.toBeNull();
      expect(connect4.legalMoves(state)).toContainEqual(move);
    }
  });

  it('renders bestMove null until step is called, even after start', () => {
    const agent = createRandomAgent(ticTacToe, { rng: createRng(3) });
    agent.start(ticTacToe.initialState());
    expect(agent.bestMove()).toBeNull();
    expect(agent.step()).toBe('done');
    expect(agent.bestMove()).not.toBeNull();
  });

  it('returns done on every step call after the first start', () => {
    const agent = createRandomAgent(ticTacToe, { rng: createRng(4) });
    agent.start(ticTacToe.initialState());
    expect(agent.step()).toBe('done');
    expect(agent.step()).toBe('done');
  });

  it('restarts cleanly when start is called again', () => {
    const agent = createRandomAgent(ticTacToe, { rng: createRng(5) });
    agent.start(ticTacToe.initialState());
    agent.step();
    agent.start(ticTacToe.initialState());
    expect(agent.bestMove()).toBeNull();
    expect(agent.step()).toBe('done');
    expect(ticTacToe.legalMoves(ticTacToe.initialState())).toContainEqual(agent.bestMove());
  });

  it('follows the same move sequence for the same seed on successive starts', () => {
    const states = ticTacToeCorpus(CORPUS_SEED + 1).filter(
      (state) => ticTacToe.legalMoves(state).length > 0,
    );
    const movesOf = (seed: number) => {
      const agent = createRandomAgent(ticTacToe, { rng: createRng(seed) });
      return states.map((state) => {
        agent.start(state);
        agent.step();
        return agent.bestMove();
      });
    };
    expect(movesOf(42)).toEqual(movesOf(42));
    expect(movesOf(42)).not.toEqual(movesOf(43));
  });

  it('draws each empty-board move uniformly (900 draws, loose multinomial bounds mean ±4.5σ)', () => {
    const agent = createRandomAgent(ticTacToe, { rng: createRng(UNIFORM_SEED) });
    const counts = new Array(9).fill(0);
    for (let draw = 0; draw < UNIFORM_DRAWS; draw += 1) {
      agent.start(ticTacToe.initialState());
      agent.step();
      const move = agent.bestMove();
      expect(move).not.toBeNull();
      counts[requireMove(move).index] += 1;
    }
    for (const count of counts) {
      expect(count).toBeGreaterThanOrEqual(Math.floor(UNIFORM_MEAN - UNIFORM_MARGIN));
      expect(count).toBeLessThanOrEqual(Math.ceil(UNIFORM_MEAN + UNIFORM_MARGIN));
    }
  });

  it('follows an injected rng: always-zero rng picks the first legal move', () => {
    const first = createRandomAgent(ticTacToe, { rng: () => 0 });
    first.start(ticTacToe.initialState());
    first.step();
    expect(first.bestMove()).toEqual({ index: 0 });
    const connect4Agent = createRandomAgent(connect4, { rng: () => 0 });
    connect4Agent.start(connect4.initialState());
    connect4Agent.step();
    expect(connect4Agent.bestMove()).toEqual({ col: 0 });
    const last = createRandomAgent(ticTacToe, { rng: () => 0.999999 });
    last.start(ticTacToe.initialState());
    last.step();
    expect(last.bestMove()).toEqual({ index: 8 });
  });

  it('reports constant baseline stats', () => {
    const agent = createRandomAgent(ticTacToe, { rng: createRng(6) });
    const before = agent.stats();
    agent.start(ticTacToe.initialState());
    agent.step();
    expect(agent.stats()).toEqual(before);
    expect(agent.stats()).toEqual({
      nodesVisited: 1,
      nodesPruned: 0,
      ttHits: 0,
      ttMisses: 0,
      leavesEvaluated: 0,
      elapsedSteps: 1,
      effectiveDepth: 0,
    });
  });
});
