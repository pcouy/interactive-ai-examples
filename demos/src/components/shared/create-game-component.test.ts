import { describe, expect, it } from 'vitest';
import { noopRecorder, type PlayerKind, type SearchAgent } from '../../core/contracts.js';
import {
  type TicTacToeMove,
  type TicTacToeState,
  ticTacToe,
} from '../../core/games/tic-tac-toe.js';
import { createTranspositionTable, SOLVED_DEPTH } from '../../core/search/transposition.js';
import { createRng, randomStates } from '../../core/testing/playouts.js';
import {
  CONNECT4_GAME_CONFIG,
  createGameAgent,
  createGameComponent,
  type GameComponentApi,
  parseEvaluatorMode,
  parsePlayersConfig,
  sameMove,
  sanitizeDepth,
  sanitizeDiscount,
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

/**
 * Aplatit un `TemplateResult` Lit en texte. Le banc de test tourne sans DOM :
 * on inspecte le template produit plutôt que le rendu effectif.
 */
function renderTemplateText(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }
  if (Array.isArray(value)) {
    return value.map(renderTemplateText).join('');
  }
  if (typeof value === 'object' && 'strings' in value && 'values' in value) {
    const template = value as { strings: readonly string[]; values: readonly unknown[] };
    let out = '';
    for (let index = 0; index < template.strings.length; index += 1) {
      out += template.strings[index];
      if (index < template.values.length) {
        out += renderTemplateText(template.values[index]);
      }
    }
    return out;
  }
  return String(value);
}

/** Texte du template, espaces (retours de ligne compris) normalisés. */
function renderTemplateTextFlat(value: unknown): string {
  return renderTemplateText(value).replace(/\s+/g, ' ').trim();
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

describe('sanitizeDiscount', () => {
  it('keeps values inside ]0, 1] and rejects the others to the fallback', () => {
    expect(sanitizeDiscount(1, 1)).toBe(1);
    expect(sanitizeDiscount(0.5, 1)).toBe(0.5);
    expect(sanitizeDiscount(0, 1)).toBe(1);
    expect(sanitizeDiscount(1.5, 1)).toBe(1);
    expect(sanitizeDiscount(Number.NaN, 1)).toBe(1);
    expect(sanitizeDiscount(0.01, 1)).toBe(0.01);
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
      prune: true,
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
      prune: true,
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
      prune: true,
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
      prune: true,
      recorder: noopRecorder,
      heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
    });
    const move = runAgentToDone(agent, state);
    expect(move).not.toBeNull();
    expect(agent.stats().ttMisses).toBeGreaterThan(0);
  });

  it('forwards the discount and rng options to the minimax engine', () => {
    // Croix joue 0 et 1, cercles 3 : { index: 2 } gagne immédiatement.
    const state = playTicTacToe([0, 3, 1]);
    const agent = createGameAgent(ticTacToe, {
      mode: 'windows',
      depth: 6,
      fallbackDepth: 6,
      transposition: false,
      prune: true,
      recorder: noopRecorder,
      heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
      discount: 0.5,
      rng: Math.random,
    });
    const move = runAgentToDone(agent, state);
    expect(move).toEqual({ index: 2 });
  });

  it('plays a full connect4 search agent turn within a bounded step budget', () => {
    const state = dropConnect4([3, 3]);
    const agent = createGameAgent(CONNECT4_GAME_CONFIG.game, {
      mode: 'windows',
      depth: 4,
      fallbackDepth: 4,
      transposition: false,
      prune: true,
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

type TicTacToeElement = GameComponentApi<TicTacToeState, TicTacToeMove>;

function createTicTacToeElement(): TicTacToeElement {
  const ctor = createGameComponent(TIC_TAC_TOE_GAME_CONFIG) as unknown as {
    new (): TicTacToeElement;
  };
  const element = new ctor();
  // No DOM in the node test environment: neutralize Lit's async render pass
  // while keeping the reactive property bookkeeping.
  (element as unknown as { performUpdate(): void }).performUpdate = () => {};
  return element;
}

describe('persistent transposition table', () => {
  const baseOptions = {
    mode: 'windows' as const,
    depth: 4,
    fallbackDepth: 6,
    transposition: true,
    recorder: noopRecorder,
    heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
  };

  it('stores entries in an injected table that survive across separate searches', () => {
    const table = createTranspositionTable();
    const state = playTicTacToe([4]);
    const first = runAgentToDone(
      createGameAgent(ticTacToe, { ...baseOptions, transpositionTable: table }),
      state,
    );
    expect(first).not.toBeNull();
    const sizeAfterFirstSearch = table.size();
    expect(sizeAfterFirstSearch).toBeGreaterThan(0);
    const nextState = ticTacToe.apply(state, first as TicTacToeMove);
    runAgentToDone(
      createGameAgent(ticTacToe, { ...baseOptions, transpositionTable: table }),
      nextState,
    );
    // La seconde recherche ajoute de nouveaux nœuds plus profonds par-dessus
    // ceux de la première, qui restent présents : la table grandit.
    expect(table.size()).toBeGreaterThan(sizeAfterFirstSearch);
  });

  it('reuses strictly deeper entries left by a previous search', () => {
    const table = createTranspositionTable();
    // Première recherche profonde : elle certifie les positions à une
    // profondeur strictement supérieure à celle de la seconde recherche.
    runAgentToDone(
      createGameAgent(ticTacToe, { ...baseOptions, depth: 6, transpositionTable: table }),
      ticTacToe.initialState(),
    );
    expect(table.size()).toBeGreaterThan(0);

    const shallowAgent = createGameAgent(ticTacToe, {
      ...baseOptions,
      depth: 2,
      transpositionTable: table,
    });
    runAgentToDone(shallowAgent, ticTacToe.initialState());
    // Les entrées strictement plus profondes sont consommées quel que soit
    // leur tour : la seconde recherche touche donc la table.
    expect(shallowAgent.stats().ttHits).toBeGreaterThan(0);
  });

  it('keeps the component table alive across turns, undos and resets, and clears on demand', () => {
    const element = createTicTacToeElement();
    element.players = '["agent","agent"]';
    element.transposition = true;
    element.depth = 3;
    element.stepTurn();
    const sizeAfterFirstTurn = element.transpositionTable.size();
    expect(sizeAfterFirstTurn).toBeGreaterThan(0);
    expect(element.gameOutcome).toBeNull();
    element.stepTurn();
    const sizeAfterSecondTurn = element.transpositionTable.size();
    expect(sizeAfterSecondTurn).toBeGreaterThan(sizeAfterFirstTurn);
    element.reset();
    expect(element.transpositionTable.size()).toBe(sizeAfterSecondTurn);
    element.clearTranspositionTable();
    expect(element.transpositionTable.size()).toBe(0);
  });

  it('memorizes visited states by key and clears them when the table is cleared', () => {
    const element = createTicTacToeElement();
    element.players = '["agent","agent"]';
    element.transposition = true;
    element.depth = 3;
    element.stepTurn();
    const internals = element as unknown as {
      seenStates: Map<string, TicTacToeState>;
    };
    expect(internals.seenStates.size).toBeGreaterThan(0);
    // Un état nécessite une recherche complète pour des entrées TT fiables.
    element.stepTurn();
    const current = element.currentState as TicTacToeState;
    expect(internals.seenStates.has(ticTacToe.stateKey(current))).toBe(true);
    expect(internals.seenStates.get(ticTacToe.stateKey(current))).toBe(current);
    element.clearTranspositionTable();
    expect(internals.seenStates.size).toBe(0);
    element.reset();
    // Le reset conserve le mémo : la TT reste persistante d'elle-même.
    // (comparé à clearTranspositionTable qui vide les deux)
  });

  it('renders a deterministic inline SVG mini-board for a visited state', () => {
    const element = createTicTacToeElement();
    element.players = '["agent","agent"]';
    element.transposition = true;
    element.depth = 3;
    element.stepTurn();
    const internals = element as unknown as {
      seenStates: Map<string, TicTacToeState>;
      ttStateSvg(state: TicTacToeState): string;
    };
    const entry = internals.seenStates.entries().next();
    expect(entry.done).toBe(false);
    const [key, state] = entry.value as [string, TicTacToeState];
    expect(key).toBe(ticTacToe.stateKey(state));
    const svg = internals.ttStateSvg(state);
    expect(svg.startsWith('<svg xmlns="http://www.w3.org/2000/svg"')).toBe(true);
    expect(svg).toContain('viewBox="0 0 36 36"');
    expect(svg).not.toContain('bestMove');
    expect(internals.ttStateSvg(state)).toBe(svg);
  });

  it('renders a SOLVED_DEPTH entry as ∞ in the TT dialog', () => {
    const element = createTicTacToeElement();
    element.transposition = true;
    const state = ticTacToe.initialState();
    const key = ticTacToe.stateKey(state);
    element.transpositionTable.store(key, { score: 0, depth: SOLVED_DEPTH, exact: true });

    const internals = element as unknown as { renderTtDialog(): unknown };
    const text = renderTemplateTextFlat(internals.renderTtDialog());
    expect(text).toContain('Profondeur eff.');
    expect(text).toContain('∞');
    expect(text).not.toContain('Infinity');
    expect(text).toContain(key);
  });

  it('shows the effective depth on the search status line', () => {
    const element = createTicTacToeElement();
    element.players = '["agent","agent"]';
    element.transposition = true;
    element.depth = 3;
    element.stepTurn();

    const stats = element.searchStats;
    expect(stats).not.toBeNull();
    const expected =
      stats?.effectiveDepth === SOLVED_DEPTH ? '∞' : String(stats?.effectiveDepth ?? '');
    const internals = element as unknown as { renderSearchStatus(): unknown };
    const text = renderTemplateTextFlat(internals.renderSearchStatus());
    expect(text).toContain(`profondeur effective : ${expected}`);
  });
});

describe('alpha-beta vs minimax pruning', () => {
  const baseOptions = {
    mode: 'windows' as const,
    depth: 6,
    fallbackDepth: 6,
    transposition: false,
    recorder: noopRecorder,
    heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
  };

  it('visits fewer nodes with alpha-beta than with plain minimax on a cutting position', () => {
    // Croix joue 1 et 2, cercles 4 et 5 : le premier coup légal (0) gagne,
    // donc la borne alpha monte au maximum et coupe toute la suite.
    const state = playTicTacToe([1, 4, 2, 5]);
    const pruned = createGameAgent(ticTacToe, { ...baseOptions, prune: true });
    const plain = createGameAgent(ticTacToe, { ...baseOptions, prune: false });
    runAgentToDone(pruned, state);
    runAgentToDone(plain, state);
    expect(plain.stats().nodesPruned).toBe(0);
    expect(pruned.stats().nodesPruned).toBeGreaterThan(0);
    expect(pruned.stats().nodesVisited).toBeLessThan(plain.stats().nodesVisited);
  });

  it('forwards strictPruning to the engine: never prunes more than the standard cut', () => {
    // Même position coupante : en strict, l'égalité exacte avec la borne ne
    // coupe pas, donc l'arbre est au moins aussi grand et les coupures au plus
    // aussi nombreuses qu'en alpha-bêta classique.
    const state = playTicTacToe([1, 4, 2, 5]);
    const standard = createGameAgent(ticTacToe, { ...baseOptions, prune: true });
    const strict = createGameAgent(ticTacToe, {
      ...baseOptions,
      prune: true,
      strictPruning: true,
    });
    const standardMove = runAgentToDone(standard, state);
    const strictMove = runAgentToDone(strict, state);
    expect(standardMove).not.toBeNull();
    expect(strictMove).not.toBeNull();
    const legal = ticTacToe.legalMoves(state);
    expect(legal.some((move) => sameMove(move, standardMove as TicTacToeMove))).toBe(true);
    expect(legal.some((move) => sameMove(move, strictMove as TicTacToeMove))).toBe(true);
    expect(strict.stats().nodesVisited).toBeGreaterThanOrEqual(standard.stats().nodesVisited);
    expect(strict.stats().nodesPruned).toBeLessThanOrEqual(standard.stats().nodesPruned);
  });

  it('exposes the strict pruning toggle in the toolbar, except for the random evaluator', () => {
    const element = createTicTacToeElement();
    const internals = element as unknown as { renderStrictPruningField(): unknown };

    element.evaluator = 'windows';
    expect(renderTemplateTextFlat(internals.renderStrictPruningField())).toContain(
      'Élagage strict',
    );

    element.evaluator = 'random';
    // Lit renvoie `nothing` : le champ n'apparaît pas dans le template rendu.
    expect(renderTemplateTextFlat(internals.renderStrictPruningField())).not.toContain(
      'Élagage strict',
    );
  });
});

describe('GameComponent stepTurn', () => {
  it('runs one full agent search and plays exactly one move per call', () => {
    const element = createTicTacToeElement();
    element.players = '["agent","agent"]';
    element.depth = 3;
    expect(element.history).toHaveLength(1);
    element.stepTurn();
    expect(element.history).toHaveLength(2);
    expect(element.gameOutcome).toBeNull();
    expect(element.running).toBe(false);
    expect(element.searchStats).not.toBeNull();
    element.stepTurn();
    expect(element.history).toHaveLength(3);
    expect(element.gameOutcome).toBeNull();
  });

  it('does nothing once the game is over', () => {
    const element = createTicTacToeElement();
    element.players = '["agent","agent"]';
    element.depth = 3;
    while (element.gameOutcome === null && element.history.length < 9) {
      element.stepTurn();
    }
    expect(element.gameOutcome).not.toBeNull();
    const stalledLength = element.history.length;
    element.stepTurn();
    expect(element.history).toHaveLength(stalledLength);
  });
});

describe('GameComponent auto-restart', () => {
  function collectEvents(element: TicTacToeElement, names: string[]): () => string[] {
    const seen: string[] = [];
    for (const name of names) {
      element.addEventListener(name, () => {
        seen.push(name);
      });
    }
    return () => seen;
  }

  it('restarts the game on the ending move when autoRestart is enabled', () => {
    const element = createTicTacToeElement();
    element.players = '["agent","agent"]';
    element.depth = 3;
    element.autoRestart = true;
    const events = collectEvents(element, ['drl-game-over', 'drl-game-reset']);
    element.stepTurn();
    expect(element.history.length).toBeGreaterThan(1);
    // La partie se termine, puis la relance ramène l'historique à 1 coup d'ici
    // la fin de la boucle (les coups rejoués regrandissent l'historique).
    let guard = 100;
    while (element.history.length > 1 && guard > 0) {
      element.stepTurn();
      guard -= 1;
    }
    expect(guard).toBeGreaterThan(0);
    expect(element.history).toHaveLength(1);
    expect(element.gameOutcome).toBeNull();
    const fired = events();
    expect(fired).toContain('drl-game-over');
    expect(fired).toContain('drl-game-reset');
    expect(fired.indexOf('drl-game-over')).toBeLessThan(fired.indexOf('drl-game-reset'));
    // En pas à pas, la nouvelle partie attend le prochain clic : running
    // reste false et aucun tour n'a été démarré automatiquement.
    expect(element.running).toBe(false);
    element.stepTurn();
    expect(element.history).toHaveLength(2);
    expect(element.gameOutcome).toBeNull();
  });

  it('keeps the game over after the ending move when autoRestart is disabled', () => {
    const element = createTicTacToeElement();
    element.players = '["agent","agent"]';
    element.depth = 3;
    const events = collectEvents(element, ['drl-game-reset']);
    while (element.gameOutcome === null && element.history.length < 9) {
      element.stepTurn();
    }
    expect(element.gameOutcome).not.toBeNull();
    const stalledLength = element.history.length;
    element.stepTurn();
    expect(element.history).toHaveLength(stalledLength);
    expect(events()).toHaveLength(0);
  });
});
