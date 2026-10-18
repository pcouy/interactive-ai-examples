import type { LitElement, PropertyValues, TemplateResult } from 'lit';
import { css, html, nothing } from 'lit';
import { property } from 'lit/decorators.js';
import { DemoElement } from '../../base/demo-element.js';
import {
  type BoardView,
  type CellCoord,
  type Evaluator,
  type Game,
  type GameOutcome,
  noopRecorder,
  type PlayerId,
  type PlayerKind,
  type SearchAgent,
  type SearchStats,
  type TreeRecorder,
} from '../../core/contracts.js';
import {
  CONNECT4_COLS,
  CONNECT4_ROWS,
  type Connect4Move,
  type Connect4State,
  connect4,
} from '../../core/games/connect4.js';
import {
  type TicTacToeMove,
  type TicTacToeState,
  ticTacToe,
} from '../../core/games/tic-tac-toe.js';
import { connect4Heuristic, ticTacToeHeuristic } from '../../core/search/evaluators/heuristic.js';
import { createRandomRolloutEvaluator } from '../../core/search/evaluators/random-rollout.js';
import { createMinimaxAgent } from '../../core/search/minimax.js';
import { createRandomAgent } from '../../core/search/random-agent.js';
import { createTranspositionTable } from '../../core/search/transposition.js';
import { drlTokens } from '../../styles/tokens.js';
import type { DrlCellClickEvent, PieceStyle } from '../drl-game-board/drl-game-board.js';

export type EvaluatorMode = 'windows' | 'rollout' | 'random';

const DEFAULT_PLAYER_KINDS: readonly [PlayerKind, PlayerKind] = ['human', 'agent'];
const DEFAULT_EVALUATOR: string = 'windows';
const AGENT_STEPS_PER_SPEED_UNIT = 32;
const AGENT_STEP_CAP = 256;

export interface GameComponentConfig<S, M> {
  readonly game: Game<S, M>;
  readonly title: string;
  readonly playerNames: readonly [string, string];
  readonly defaultDepth: number;
  readonly defaultCellSize: number;
  readonly pieceStyle: PieceStyle;
  clickableCells(state: S): readonly CellCoord[];
  moveForCell(row: number, col: number, state: S): M | null;
  readonly heuristic: Evaluator<S, M>;
}

export interface GameComponentApi<S, M> extends LitElement {
  readonly game: Game<S, M>;
  readonly title: string;
  readonly pieceStyle: PieceStyle;
  readonly playerNames: readonly [string, string];
  readonly playerKinds: readonly [PlayerKind, PlayerKind];
  readonly currentState: S | null;
  readonly history: readonly S[];
  readonly gameOutcome: GameOutcome;
  readonly searchStats: SearchStats | null;
  recorder: TreeRecorder<S> | null;
  players: string;
  depth: number;
  evaluator: string;
  transposition: boolean;
  cellSize: number;
  running: boolean;
  reset(): void;
  applyMove(move: M): boolean;
}

export interface GameComponentConstructor<S, M> {
  new (): GameComponentApi<S, M>;
  readonly prototype: GameComponentApi<S, M>;
}

export interface DrlMovePlayedDetail<S, M> {
  readonly move: M;
  readonly state: S;
  readonly player: PlayerId;
}

export interface DrlSearchStartedDetail<S, M> {
  readonly state: S;
  readonly agent: SearchAgent<S, M>;
  readonly player: PlayerId;
}

export interface DrlGameOverDetail {
  readonly outcome: GameOutcome;
}

export interface DrlMoveUndoneDetail<S> {
  readonly plies: number;
  readonly state: S;
}

export interface DrlInvalidMoveDetail {
  readonly row: number;
  readonly col: number;
}

export interface DrlGameResetDetail<S> {
  readonly state: S;
}

export interface PendingSearch<S, M> {
  readonly agent: SearchAgent<S, M>;
  readonly player: PlayerId;
  readonly state: S;
}

function readPlayerKind(entry: unknown): PlayerKind | null {
  if (entry === 'human' || entry === 'agent') {
    return entry;
  }
  if (typeof entry === 'object' && entry !== null && 'kind' in entry) {
    const kind = (entry as { kind: unknown }).kind;
    if (kind === 'human' || kind === 'agent') {
      return kind;
    }
  }
  return null;
}

export function parsePlayersConfig(
  value: string,
  fallback: readonly [PlayerKind, PlayerKind] = DEFAULT_PLAYER_KINDS,
): readonly [PlayerKind, PlayerKind] {
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(value);
  } catch {
    return [fallback[0], fallback[1]];
  }
  if (!Array.isArray(parsed)) {
    return [fallback[0], fallback[1]];
  }
  const first = readPlayerKind(parsed[0]) ?? fallback[0];
  const second = readPlayerKind(parsed[1]) ?? fallback[1];
  return [first, second];
}

export function parseEvaluatorMode(value: string): EvaluatorMode {
  if (value === 'rollout' || value === 'random') {
    return value;
  }
  return 'windows';
}

export function sanitizeDepth(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(1, Math.floor(value));
}

export function sameMove<M>(a: M, b: M): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export interface GameAgentOptions<S, M> {
  readonly mode: EvaluatorMode;
  readonly depth: number;
  readonly fallbackDepth: number;
  readonly transposition: boolean;
  readonly recorder: TreeRecorder<S>;
  readonly heuristic: Evaluator<S, M>;
}

export function createGameAgent<S, M>(
  game: Game<S, M>,
  options: GameAgentOptions<S, M>,
): SearchAgent<S, M> {
  if (options.mode === 'random') {
    return createRandomAgent(game);
  }
  const evaluator: Evaluator<S, M> =
    options.mode === 'rollout' ? createRandomRolloutEvaluator<S, M>() : options.heuristic;
  const transposition = options.transposition ? createTranspositionTable<M>() : undefined;
  return createMinimaxAgent(game, {
    depth: sanitizeDepth(options.depth, options.fallbackDepth),
    evaluator,
    prune: true,
    recorder: options.recorder,
    transposition,
  });
}

export function createGameComponent<S, M>(
  config: GameComponentConfig<S, M>,
): GameComponentConstructor<S, M> {
  const { game } = config;

  class GameComponent extends DemoElement {
    static override styles = [
      drlTokens,
      css`
        :host {
          display: block;
        }

        .drl-title {
          margin: 0 0 var(--drl-space-2);
          font-size: 1.1rem;
          font-weight: 700;
        }

        .drl-status {
          display: flex;
          flex-wrap: wrap;
          align-items: baseline;
          gap: var(--drl-space-2);
          margin-bottom: var(--drl-space-2);
          min-height: 1.4em;
        }

        .drl-turn {
          font-weight: 700;
        }

        .drl-search {
          color: var(--drl-muted);
          font-variant-numeric: tabular-nums;
        }

        .drl-banner {
          display: inline-block;
          padding: var(--drl-space-1) var(--drl-space-2);
          border: 1px solid var(--drl-accent);
          border-radius: calc(var(--drl-radius) / 2);
          background: var(--drl-surface);
          font-weight: 700;
        }

        .drl-reject {
          color: var(--drl-arrow);
        }

        .drl-toolbar {
          display: flex;
          flex-wrap: wrap;
          gap: var(--drl-space-2);
          margin-bottom: var(--drl-space-2);
        }

        .drl-button {
          font: inherit;
          padding: var(--drl-space-1) var(--drl-space-2);
          border: 1px solid var(--drl-border);
          border-radius: calc(var(--drl-radius) / 2);
          background: var(--drl-bg);
          color: var(--drl-fg);
          cursor: pointer;
        }

        .drl-button:hover:not(:disabled) {
          border-color: var(--drl-accent);
        }

        .drl-button:disabled {
          color: var(--drl-muted);
          cursor: not-allowed;
        }
      `,
    ];

    @property({ type: String, attribute: 'players' })
    players = '["human","agent"]';

    @property({ type: Number, attribute: 'depth' })
    depth = config.defaultDepth;

    @property({ type: String, attribute: 'evaluator' })
    evaluator = DEFAULT_EVALUATOR;

    @property({ type: Boolean, attribute: 'transposition' })
    transposition = false;

    @property({ type: Number, attribute: 'cell-size' })
    cellSize = config.defaultCellSize;

    @property({ attribute: false })
    recorder: TreeRecorder<S> | null = null;

    private states: S[] = [game.initialState()];

    private pending: PendingSearch<S, M> | null = null;

    private outcome: GameOutcome = null;

    private lastStats: SearchStats | null = null;

    private rejected: CellCoord | null = null;

    get game(): Game<S, M> {
      return game;
    }

    get title(): string {
      return config.title;
    }

    get pieceStyle(): PieceStyle {
      return config.pieceStyle;
    }

    get playerNames(): readonly [string, string] {
      return config.playerNames;
    }

    get playerKinds(): readonly [PlayerKind, PlayerKind] {
      return parsePlayersConfig(this.players, DEFAULT_PLAYER_KINDS);
    }

    get currentState(): S | null {
      return this.states.at(-1) ?? null;
    }

    get history(): readonly S[] {
      return [...this.states];
    }

    get gameOutcome(): GameOutcome {
      return this.outcome;
    }

    get searchStats(): SearchStats | null {
      return this.lastStats;
    }

    applyMove(move: M): boolean {
      const state = this.currentState;
      if (state === null || this.outcome !== null || !this.isLegalMove(state, move)) {
        return false;
      }
      this.pending = null;
      this.playMove(move, game.currentPlayer(state));
      return true;
    }

    override step(): void {
      if (this.outcome !== null) {
        return;
      }
      if (this.pending === null) {
        this.startTurn();
        if (this.pending === null) {
          return;
        }
      }
      const pending = this.pending;
      const budget = Math.min(
        AGENT_STEP_CAP,
        Math.max(1, Math.round(this.speed * AGENT_STEPS_PER_SPEED_UNIT)),
      );
      let done = false;
      for (let index = 0; index < budget; index += 1) {
        if (pending.agent.step() === 'done') {
          done = true;
          break;
        }
      }
      this.lastStats = pending.agent.stats();
      if (done) {
        this.pending = null;
        const move = pending.agent.bestMove();
        const state = this.currentState;
        if (move !== null && state !== null && this.isLegalMove(state, move)) {
          this.playMove(move, pending.player);
          return;
        }
      }
      this.requestUpdate();
    }

    protected override onReset(): void {
      this.states = [game.initialState()];
      this.pending = null;
      this.outcome = null;
      this.lastStats = null;
      this.rejected = null;
      this.emitEvent<DrlGameResetDetail<S>>('drl-game-reset', { state: this.currentState as S });
      this.requestUpdate();
    }

    protected override willUpdate(changed: PropertyValues<this>): void {
      super.willUpdate(changed);
      if (
        changed.has('players') ||
        changed.has('depth') ||
        changed.has('evaluator') ||
        changed.has('transposition')
      ) {
        this.pending = null;
      }
    }

    protected override renderContent(): TemplateResult {
      const state = this.currentState;
      const player = state === null ? 0 : game.currentPlayer(state);
      const outcome = this.outcome;
      return html`
        <h2 class="drl-title">${config.title}</h2>
        <div class="drl-status" role="status">
          ${
            outcome === null
              ? html`
                  <span class="drl-turn">Au trait : ${this.playerName(player)}</span>
                  ${this.renderSearchStatus()} ${this.renderRejected()}
                `
              : html`<span class="drl-banner">${this.outcomeLabel(outcome)}</span>`
          }
        </div>
        <div class="drl-toolbar">
          <button
            type="button"
            class="drl-button"
            ?disabled=${this.states.length <= 1}
            @click=${this.onUndoClick}
          >
            Annuler le coup
          </button>
        </div>
        <drl-game-board
          .view=${this.boardView()}
          .pieceStyle=${config.pieceStyle}
          .cellSize=${this.cellSize}
          @drl-cell-click=${this.onCellClick}
        ></drl-game-board>
      `;
    }

    private readonly onCellClick = (event: DrlCellClickEvent): void => {
      const state = this.currentState;
      if (state === null || this.outcome !== null || this.pending !== null) {
        return;
      }
      const player = game.currentPlayer(state);
      if (this.playerKinds[player] !== 'human') {
        return;
      }
      const { row, col } = event.detail;
      const move = config.moveForCell(row, col, state);
      if (move === null || !this.isLegalMove(state, move)) {
        this.rejected = { row, col };
        this.emitEvent<DrlInvalidMoveDetail>('drl-invalid-move', { row, col });
        this.requestUpdate();
        return;
      }
      this.playMove(move, player);
    };

    private readonly onUndoClick = (): void => {
      if (this.states.length <= 1) {
        return;
      }
      this.pending = null;
      const kinds = this.playerKinds;
      const hasHuman = kinds[0] === 'human' || kinds[1] === 'human';
      let plies = 0;
      do {
        this.states.pop();
        plies += 1;
        const state = this.currentState;
        if (state === null || !hasHuman) {
          break;
        }
        if (kinds[game.currentPlayer(state)] === 'human') {
          break;
        }
      } while (this.states.length > 1);
      const state = this.currentState;
      if (state === null) {
        return;
      }
      this.outcome = game.winner(state);
      this.rejected = null;
      this.lastStats = null;
      this.emitEvent<DrlMoveUndoneDetail<S>>('drl-move-undone', { plies, state });
      this.requestUpdate();
    };

    private playMove(move: M, player: PlayerId): void {
      const state = this.currentState;
      if (state === null || this.outcome !== null) {
        return;
      }
      const next = game.apply(state, move);
      this.states.push(next);
      this.rejected = null;
      this.outcome = game.winner(next);
      this.emitEvent<DrlMovePlayedDetail<S, M>>('drl-move-played', { move, state: next, player });
      if (this.outcome === null) {
        this.startTurn();
      } else {
        this.running = false;
        this.emitEvent<DrlGameOverDetail>('drl-game-over', { outcome: this.outcome });
      }
      this.requestUpdate();
    }

    private startTurn(): void {
      const state = this.currentState;
      if (state === null || this.outcome !== null || this.pending !== null) {
        return;
      }
      const player = game.currentPlayer(state);
      if (this.playerKinds[player] !== 'agent') {
        return;
      }
      const agent = createGameAgent(game, {
        mode: parseEvaluatorMode(this.evaluator),
        depth: this.depth,
        fallbackDepth: config.defaultDepth,
        transposition: this.transposition,
        recorder: this.recorder ?? noopRecorder,
        heuristic: config.heuristic,
      });
      agent.start(state);
      this.pending = { agent, player, state };
      this.emitEvent<DrlSearchStartedDetail<S, M>>('drl-search-started', {
        state,
        agent,
        player,
      });
    }

    private isLegalMove(state: S, move: M): boolean {
      return game.legalMoves(state).some((legal) => sameMove(legal, move));
    }

    private boardView(): BoardView {
      const state = this.currentState;
      if (state === null) {
        return game.boardView(game.initialState());
      }
      const view = game.boardView(state);
      if (this.outcome !== null || this.pending !== null) {
        return view;
      }
      const player = game.currentPlayer(state);
      if (this.playerKinds[player] !== 'human') {
        return view;
      }
      return { ...view, clickable: config.clickableCells(state) };
    }

    private outcomeLabel(outcome: GameOutcome): string {
      if (outcome === 'draw' || outcome === null) {
        return 'Partie nulle';
      }
      return `Victoire : ${this.playerName(outcome)}`;
    }

    private playerName(player: PlayerId): string {
      return config.playerNames[player] ?? `Joueur ${player}`;
    }

    private renderSearchStatus(): TemplateResult | typeof nothing {
      const stats = this.lastStats;
      if (stats === null) {
        return nothing;
      }
      const label = this.pending === null ? 'Recherche terminée' : 'Recherche en cours';
      return html`<span class="drl-search">
        ${label} · ${stats.nodesVisited} nœuds · ${stats.nodesPruned} coupures ·
        ${stats.ttHits} touches TT · ${stats.elapsedSteps} étapes
      </span>`;
    }

    private renderRejected(): TemplateResult | typeof nothing {
      const rejected = this.rejected;
      if (rejected === null) {
        return nothing;
      }
      return html`<span class="drl-reject">
        Coup non jouable (ligne ${rejected.row + 1}, colonne ${rejected.col + 1})
      </span>`;
    }

    private emitEvent<D>(name: string, detail: D): void {
      this.dispatchEvent(new CustomEvent<D>(name, { detail, bubbles: true, composed: true }));
    }
  }

  return GameComponent as unknown as GameComponentConstructor<S, M>;
}

export const TIC_TAC_TOE_GAME_CONFIG: GameComponentConfig<TicTacToeState, TicTacToeMove> = {
  game: ticTacToe,
  title: 'Morpion',
  playerNames: ['Croix', 'Cercles'],
  defaultDepth: 6,
  defaultCellSize: 72,
  pieceStyle: 'marks',
  clickableCells(state) {
    const coords: CellCoord[] = [];
    for (let index = 0; index < state.cells.length; index += 1) {
      if (state.cells[index] === null) {
        coords.push({ row: Math.floor(index / 3), col: index % 3 });
      }
    }
    return coords;
  },
  moveForCell(row, col, state) {
    const inGrid = row >= 0 && row < 3 && col >= 0 && col < 3;
    const index = row * 3 + col;
    return inGrid && state.cells[index] === null ? { index } : null;
  },
  heuristic: ticTacToeHeuristic,
};

export const CONNECT4_GAME_CONFIG: GameComponentConfig<Connect4State, Connect4Move> = {
  game: connect4,
  title: 'Puissance 4',
  playerNames: ['Bleus', 'Violets'],
  defaultDepth: 4,
  defaultCellSize: 48,
  pieceStyle: 'discs',
  clickableCells(state) {
    const coords: CellCoord[] = [];
    for (let col = 0; col < CONNECT4_COLS; col += 1) {
      if (state.grid[0][col] !== null) {
        continue;
      }
      for (let row = 0; row < CONNECT4_ROWS; row += 1) {
        coords.push({ row, col });
      }
    }
    return coords;
  },
  moveForCell(_row, col, state) {
    return col >= 0 && col < CONNECT4_COLS && state.grid[0][col] === null ? { col } : null;
  },
  heuristic: connect4Heuristic,
};
