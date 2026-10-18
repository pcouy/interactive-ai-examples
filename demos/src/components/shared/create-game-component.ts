import type { LitElement, PropertyValues, TemplateResult } from 'lit';
import { css, html, nothing } from 'lit';
import { property, state } from 'lit/decorators.js';
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
import {
  createTranspositionTable,
  type TranspositionTable,
} from '../../core/search/transposition.js';
import { drlTokens } from '../../styles/tokens.js';
import type { DrlCellClickEvent, PieceStyle } from '../drl-game-board/drl-game-board.js';

export type EvaluatorMode = 'windows' | 'rollout' | 'random';

const DEFAULT_PLAYER_KINDS: readonly [PlayerKind, PlayerKind] = ['human', 'agent'];
const DEFAULT_EVALUATOR: string = 'windows';
const DEFAULT_ROLLOUT_SAMPLES = 1;
const AGENT_STEPS_PER_SPEED_UNIT = 32;
const AGENT_STEP_CAP = 256;
const TT_BROWSER_ROW_LIMIT = 200;

function formatScoreValue(value: number): string {
  return value.toFixed(2).replace('.', ',');
}
const STEP_TURN_STEP_CAP = 100_000;

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
  rolloutSamples: number;
  discount: number;
  transposition: boolean;
  pruning: boolean;
  cellSize: number;
  running: boolean;
  autoRestart: boolean;
  readonly transpositionTable: TranspositionTable;
  reset(): void;
  clearTranspositionTable(): void;
  applyMove(move: M): boolean;
  stepTurn(): void;
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

export function sanitizeSamples(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(1, Math.floor(value));
}

/** Actualisation γ ∈ ]0, 1] : hors plage ou non fini, retour au défaut. */
export function sanitizeDiscount(value: number, fallback: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return fallback;
  }
  return Math.min(1, Math.max(0.01, value));
}

export function sameMove<M>(a: M, b: M): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export interface GameAgentOptions<S, M> {
  readonly mode: EvaluatorMode;
  readonly depth: number;
  readonly fallbackDepth: number;
  readonly transposition: boolean;
  readonly transpositionTable?: TranspositionTable;
  readonly prune?: boolean;
  readonly recorder: TreeRecorder<S>;
  readonly heuristic: Evaluator<S, M>;
  readonly samples?: number;
  readonly discount?: number;
  readonly rng?: () => number;
  /** Politique de consommation de la table par l'agent minimax. */
  readonly probeTurns?: 'all' | 'previous';
}

export function createGameAgent<S, M>(
  game: Game<S, M>,
  options: GameAgentOptions<S, M>,
): SearchAgent<S, M> {
  if (options.mode === 'random') {
    return createRandomAgent(game);
  }
  const evaluator: Evaluator<S, M> =
    options.mode === 'rollout'
      ? createRandomRolloutEvaluator<S, M>({ samples: options.samples })
      : options.heuristic;
  const transposition = options.transposition
    ? (options.transpositionTable ?? createTranspositionTable())
    : undefined;
  return createMinimaxAgent(game, {
    depth: sanitizeDepth(options.depth, options.fallbackDepth),
    evaluator,
    prune: options.prune ?? true,
    recorder: options.recorder,
    transposition,
    discount: options.discount,
    rng: options.rng,
    probeTurns: options.probeTurns,
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
          align-items: center;
          gap: var(--drl-space-2);
          margin-bottom: var(--drl-space-2);
          line-height: 1.5;
          /* Hauteur constante entre les états « recherche en cours » et
           * « bannière de fin de partie » : la rangée est dimensionnée pour
           * la statistique de recherche qui peut s'étaler sur plusieurs lignes
           * (5 × 1,5em) plus les espacements et les bordures de bannière,
           * ce qui évite tout décalage de mise en page entre les parties. */
          min-height: calc(7.5em + 4 * var(--drl-space-1) + 2px);
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
          line-height: 1.5;
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

        .drl-field {
          display: inline-flex;
          align-items: center;
          gap: var(--drl-space-1);
          color: var(--drl-muted);
        }

        .drl-number {
          width: 4.5rem;
          font: inherit;
          padding: 2px 4px;
          border: 1px solid var(--drl-border);
          border-radius: calc(var(--drl-radius) / 2);
          background: var(--drl-surface);
          color: var(--drl-fg);
        }

        .drl-tt-dialog {
          padding: 0;
          border: 1px solid var(--drl-border);
          border-radius: var(--drl-radius);
          background: var(--drl-surface);
          color: var(--drl-fg);
          font: inherit;
          width: min(92vw, 720px);
          max-height: 85vh;
        }

        .drl-tt-dialog::backdrop {
          background: rgba(15, 23, 42, 0.45);
        }

        .drl-tt-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: var(--drl-space-2);
          padding: var(--drl-space-2) var(--drl-space-3);
          border-bottom: 1px solid var(--drl-border);
        }

        .drl-tt-title {
          margin: 0;
          font-size: 1.05rem;
          font-weight: 700;
        }

        .drl-tt-summary {
          margin: 0;
          padding: var(--drl-space-2) var(--drl-space-3) 0;
          color: var(--drl-muted);
          font-size: 0.9em;
        }

        .drl-tt-filter {
          display: inline-flex;
          align-items: center;
          gap: var(--drl-space-1);
          padding: var(--drl-space-2) var(--drl-space-3) 0;
          color: var(--drl-muted);
        }

        .drl-tt-filter input {
          font: inherit;
          padding: var(--drl-space-1) var(--drl-space-2);
          border: 1px solid var(--drl-border);
          border-radius: calc(var(--drl-radius) / 2);
          background: var(--drl-bg);
          color: var(--drl-fg);
          min-width: 14rem;
        }

        .drl-tt-scroll {
          margin: var(--drl-space-2) var(--drl-space-3);
          border: 1px solid var(--drl-border);
          border-radius: calc(var(--drl-radius) / 2);
          overflow: auto;
          max-height: 50vh;
        }

        .drl-tt-table {
          border-collapse: collapse;
          width: 100%;
          font-variant-numeric: tabular-nums;
          font-size: 0.92em;
        }

        .drl-tt-table th,
        .drl-tt-table td {
          padding: var(--drl-space-1) var(--drl-space-2);
          border-bottom: 1px solid var(--drl-border);
          text-align: left;
          white-space: nowrap;
        }

        .drl-tt-table th {
          position: sticky;
          top: 0;
          background: var(--drl-bg);
          font-weight: 700;
        }

        .drl-tt-svg {
        vertical-align: middle;
      }

      .drl-tt-svg svg {
        display: block;
      }

      .drl-tt-svg-empty {
        color: var(--drl-muted);
      }

      .drl-tt-key {
          font-family: monospace;
          font-size: 0.9em;
        }

        .drl-tt-note {
          margin: 0 var(--drl-space-3) var(--drl-space-3);
          color: var(--drl-muted);
          font-size: 0.9em;
        }
      `,
    ];

    @property({ type: String, attribute: 'players' })
    players = '["human","agent"]';

    @property({ type: Number, attribute: 'depth' })
    depth = config.defaultDepth;

    @property({ type: String, attribute: 'evaluator' })
    evaluator = DEFAULT_EVALUATOR;

    @property({ type: Number, attribute: 'rollout-samples' })
    rolloutSamples = DEFAULT_ROLLOUT_SAMPLES;

    @property({ type: Number, attribute: 'discount' })
    discount = 1;

    @property({ type: Boolean, attribute: 'transposition' })
    transposition = false;

    @property({ type: Boolean, attribute: 'pruning' })
    pruning = true;

    @property({ type: Number, attribute: 'cell-size' })
    cellSize = config.defaultCellSize;

    @property({ type: Boolean, attribute: 'auto-restart' })
    autoRestart = false;

    @property({ attribute: false })
    recorder: TreeRecorder<S> | null = null;

    @state()
    private ttSize = 0;

    @state()
    private ttKeyFilter = '';

    private readonly tt: TranspositionTable = createTranspositionTable();

    // Mémo clé d'état → état, uniquement pour l'affichage des minis-plateaux
    // de la TT (les clefs ne permettent pas de reconstruire un état).
    private readonly seenStates = new Map<string, S>();

    private ttSvgObserver: IntersectionObserver | null = null;

    private restartScheduled = false;

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

    get transpositionTable(): TranspositionTable {
      return this.tt;
    }

    clearTranspositionTable(): void {
      this.tt.clear();
      // Les minis-plateaux ne servent qu'au contenu courant de la table.
      this.seenStates.clear();
      this.ttSize = this.tt.size();
      this.requestUpdate();
    }

    stepTurn(): void {
      if (this.outcome !== null) {
        return;
      }
      this.running = false;
      if (this.pending === null) {
        this.startTurn();
      }
      const pending = this.pending;
      if (pending === null) {
        return;
      }
      let done = false;
      for (let budget = STEP_TURN_STEP_CAP; budget > 0; budget -= 1) {
        if (pending.agent.step() === 'done') {
          done = true;
          break;
        }
      }
      this.finishTurn(pending, done);
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
      this.finishTurn(pending, done);
    }

    private finishTurn(pending: PendingSearch<S, M>, done: boolean): void {
      this.lastStats = pending.agent.stats();
      this.ttSize = this.tt.size();
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
      this.pending = null;
      this.resetGameState();
      this.emitEvent<DrlGameResetDetail<S>>('drl-game-reset', { state: this.currentState as S });
      this.requestUpdate();
    }

    private resetGameState(): void {
      this.states = [game.initialState()];
      this.outcome = null;
      this.rejected = null;
      this.lastStats = null;
    }

    protected override willUpdate(changed: PropertyValues<this>): void {
      super.willUpdate(changed);
      if (
        changed.has('players') ||
        changed.has('depth') ||
        changed.has('evaluator') ||
        changed.has('rolloutSamples') ||
        changed.has('discount') ||
        changed.has('transposition') ||
        changed.has('pruning')
      ) {
        this.pending = null;
      }
      if (changed.has('evaluator')) {
        this.tt.clear();
        this.ttSize = this.tt.size();
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
                  ${this.renderSearchStatus()} ${this.renderTtCount()} ${this.renderRejected()}
                `
              : html`<span class="drl-banner">${this.outcomeLabel(outcome)}</span>${this.renderTtCount()}`
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
          ${
            this.transposition
              ? html`
                  <button
                    type="button"
                    class="drl-button"
                    ?disabled=${this.ttSize === 0}
                    @click=${this.onClearTtClick}
                  >
                    Vider la TT
                  </button>
                  <button
                    type="button"
                    class="drl-button"
                    ?disabled=${this.ttSize === 0}
                    @click=${this.onBrowseTtClick}
                  >
                    Parcourir la TT
                  </button>
                `
              : nothing
          }
          ${this.renderStepTurnButton()}
          ${this.renderAutoRestartField()}
          ${this.renderSamplesField()}
          ${this.renderDiscountField()}
        </div>
        <drl-game-board
          .view=${this.boardView()}
          .pieceStyle=${config.pieceStyle}
          .cellSize=${this.cellSize}
          @drl-cell-click=${this.onCellClick}
        ></drl-game-board>
        ${this.renderTtDialog()}
      `;
    }

    override disconnectedCallback(): void {
      this.ttSvgObserver?.disconnect();
      this.ttSvgObserver = null;
      super.disconnectedCallback();
    }

    protected override updated(changed: PropertyValues<this>): void {
      super.updated(changed);
      this.syncTtSvgObserver();
    }

    private syncTtSvgObserver(): void {
      if (typeof IntersectionObserver === 'undefined') {
        return;
      }
      const dialog = this.renderRoot.querySelector<HTMLDialogElement>('.drl-tt-dialog');
      if (dialog === null || !dialog.open) {
        this.ttSvgObserver?.disconnect();
        this.ttSvgObserver = null;
        return;
      }
      // Recréé à chaque mise à jour : les lignes filtrées sont re-rendues et
      // l'ancien observateur pointerait sur des cellules détachées.
      this.ttSvgObserver?.disconnect();
      const observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (!entry.isIntersecting) {
              continue;
            }
            const td = entry.target as HTMLTableCellElement;
            const key = td.getAttribute('data-entry-key');
            if (key === null || td.childElementCount > 0) {
              continue;
            }
            const state = this.seenStates.get(key);
            td.innerHTML =
              state === undefined
                ? '<span class="drl-tt-svg-empty" title="État non visité par cette partie">–</span>'
                : this.ttStateSvg(state);
            observer.unobserve(td);
          }
        },
        { rootMargin: '100px' },
      );
      for (const td of Array.from(
        this.renderRoot.querySelectorAll<HTMLTableCellElement>('.drl-tt-svg[data-entry-key]'),
      )) {
        if (td.childElementCount === 0) {
          observer.observe(td);
        }
      }
      this.ttSvgObserver = observer;
    }

    private ttStateSvg(state: S): string {
      const view = game.boardView(state);
      const cell = 12;
      const width = view.cols * cell;
      const height = view.rows * cell;
      const fills = ['#2563eb', '#7c3aed'];
      const parts: string[] = [
        `<rect x="0" y="0" width="${width}" height="${height}" style="fill:var(--drl-surface,#ffffff)"/>`,
      ];
      for (let row = 0; row < view.rows; row += 1) {
        const line = view.cells[row] ?? [];
        for (let col = 0; col < view.cols; col += 1) {
          const owner = line[col] ?? null;
          const fill = owner === null ? 'none' : (fills[owner] ?? 'none');
          parts.push(
            `<rect x="${col * cell}" y="${row * cell}" width="${cell}" height="${cell}"` +
              ` style="fill:${fill};stroke:var(--drl-border,#cbd5e1)"/>`,
          );
        }
      }
      return (
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}"` +
        ` width="${width}" height="${height}" role="img">${parts.join('')}</svg>`
      );
    }

    private renderTtDialog(): TemplateResult | typeof nothing {
      if (!this.transposition) {
        return nothing;
      }
      const all = this.tt.snapshot();
      const filter = this.ttKeyFilter.trim().toLowerCase();
      const rows = (
        filter === '' ? all : all.filter((row) => row.key.toLowerCase().includes(filter))
      ).slice(0, TT_BROWSER_ROW_LIMIT);
      const totalRows = filter === '' ? all.length : undefined;
      return html`
        <dialog class="drl-tt-dialog" @close=${this.onCloseTtClick}>
          <header class="drl-tt-head">
            <h3 class="drl-tt-title">Table de transposition</h3>
            <button
              type="button"
              class="drl-button drl-tt-close"
              @click=${this.onCloseTtClick}
            >
              Fermer
            </button>
          </header>
          <p class="drl-tt-summary">
            ${all.length} entrée${all.length > 1 ? 's' : ''} · tour courant :
            ${this.tt.currentTurn()} · les entrées à profondeur égale d'un tour
            antérieur ne sont plus consommées.
          </p>
          <label class="drl-tt-filter">
            <span>Filtrer par clé</span>
            <input
              type="search"
              .value=${this.ttKeyFilter}
              @input=${this.onTtFilterInput}
              placeholder="ex. : 001200…"
            />
          </label>
          <div class="drl-tt-scroll" role="region" aria-label="Entrées de la table">
            <table class="drl-tt-table">
              <thead>
                <tr>
                  <th scope="col">Position</th>
                  <th scope="col">Clé</th>
                  <th scope="col">Score</th>
                  <th scope="col">Profondeur</th>
                  <th scope="col">Éch.</th>
                  <th scope="col">Exact</th>
                  <th scope="col">Tour</th>
                </tr>
              </thead>
              <tbody>
                ${rows.map(
                  (row) => html`
                    <tr>
                      <td class="drl-tt-svg" data-entry-key="${row.key}"></td>
                      <td class="drl-tt-key">${row.key}</td>
                      <td>${formatScoreValue(row.score)}</td>
                      <td>${row.depth}</td>
                      <td>${row.samples ?? '–'}</td>
                      <td>${row.exact ? 'oui' : 'non'}</td>
                      <td>${row.turn}</td>
                    </tr>
                  `,
                )}
              </tbody>
            </table>
          </div>
          ${
            rows.length === 0
              ? html`<p class="drl-tt-note">Aucune entrée correspondante.</p>`
              : totalRows !== undefined && totalRows > rows.length
                ? html`
                    <p class="drl-tt-note">
                      ${rows.length} sur ${totalRows} entrées affichées (limite
                      ${TT_BROWSER_ROW_LIMIT}).
                    </p>
                  `
                : nothing
          }
        </dialog>
      `;
    }

    private readonly onBrowseTtClick = (): void => {
      const dialog = this.renderRoot.querySelector<HTMLDialogElement>('.drl-tt-dialog');
      if (dialog !== null && !dialog.open) {
        this.ttKeyFilter = '';
        dialog.showModal();
        // Les lignes étaient hors-champ tant que le dialogue était fermé :
        // l'observateur d'intersection est (re)créé maintenant qu'il est ouvert.
        this.syncTtSvgObserver();
      }
    };

    private readonly onCloseTtClick = (): void => {
      const dialog = this.renderRoot.querySelector<HTMLDialogElement>('.drl-tt-dialog');
      if (dialog?.open) {
        dialog.close();
      }
    };

    private readonly onTtFilterInput = (event: Event): void => {
      const input = event.target;
      if (input instanceof HTMLInputElement) {
        this.ttKeyFilter = input.value;
      }
    };

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

    private renderDiscountField(): TemplateResult | typeof nothing {
      if (parseEvaluatorMode(this.evaluator) === 'random') {
        return nothing;
      }
      return html`
        <label class="drl-field">
          <span>Actualisation (γ)</span>
          <input
            class="drl-number"
            type="number"
            min="0.05"
            max="1"
            step="0.05"
            .value=${this.discount.toFixed(2)}
            @change=${this.onDiscountInput}
          />
        </label>
      `;
    }

    private readonly onSamplesInput = (event: Event): void => {
      const input = event.target;
      if (!(input instanceof HTMLInputElement)) {
        return;
      }
      this.rolloutSamples = sanitizeSamples(Number(input.value), DEFAULT_ROLLOUT_SAMPLES);
    };

    private readonly onDiscountInput = (event: Event): void => {
      const input = event.target;
      if (!(input instanceof HTMLInputElement)) {
        return;
      }
      this.discount = sanitizeDiscount(Number(input.value), 1);
    };

    private readonly onClearTtClick = (): void => {
      this.clearTranspositionTable();
    };

    private readonly onStepTurnClick = (): void => {
      this.stepTurn();
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
      // L'annulation ouvre un nouveau contexte de tour : les entrées à
      // profondeur égale accumulées avant l'annulation ne sont plus
      // consommables (les plus profondes restent valables).
      this.tt.beginTurn();
      this.ttSize = this.tt.size();
      this.emitEvent<DrlMoveUndoneDetail<S>>('drl-move-undone', { plies, state });
      this.requestUpdate();
    };

    private playMove(move: M, player: PlayerId): void {
      const state = this.currentState;
      if (state === null || this.outcome !== null) {
        return;
      }
      const next = game.apply(state, move);
      this.seenStates.set(game.stateKey(state), state);
      this.seenStates.set(game.stateKey(next), next);
      this.states.push(next);
      this.rejected = null;
      this.outcome = game.winner(next);
      this.emitEvent<DrlMovePlayedDetail<S, M>>('drl-move-played', { move, state: next, player });
      if (this.outcome === null) {
        this.startTurn();
      } else {
        this.emitEvent<DrlGameOverDetail>('drl-game-over', { outcome: this.outcome });
        this.maybeAutoRestart();
      }
      this.requestUpdate();
    }

    /**
     * En agent contre agent, relance une nouvelle partie dès qu'une partie
     * se termine. On ne passe pas par reset() (il couperait la boucle de
     * jeu) : l'état de partie seul est réinitialisé, `running` est conservé.
     * Boucle active ⇒ le prochain step() démarre le tour suivant ; en pas à
     * pas (running false) la nouvelle position attend le prochain clic.
     *
     * La relance est différée de deux frames d'animation : au moins un
     * rendu peint l'état final de la partie terminée avant que le plateau
     * ne revienne à la position initiale. Si requestAnimationFrame n'existe
     * pas (environnement de test), la relance reste synchrone.
     */
    private maybeAutoRestart(): void {
      if (!this.autoRestart || !this.playerKinds.every((kind) => kind === 'agent')) {
        this.running = false;
        return;
      }
      if (this.restartScheduled) {
        return;
      }
      this.restartScheduled = true;
      let deferred = false;
      const schedule =
        typeof requestAnimationFrame === 'function'
          ? (callback: () => void) => {
              deferred = true;
              requestAnimationFrame(() => {
                requestAnimationFrame(callback);
              });
            }
          : (callback: () => void) => callback();
      schedule(() => {
        this.restartScheduled = false;
        // Entre-temps la configuration a pu changer, l'élément a pu être
        // déconnecté ou un reset manuel être arrivé : ne relance que si la
        // partie est toujours finie (la déconnexion n'a de sens que pour
        // la relance réellement différée).
        if (this.outcome === null || !this.autoRestart || (deferred && !this.isConnected)) {
          return;
        }
        if (!this.playerKinds.every((kind) => kind === 'agent')) {
          return;
        }
        this.pending = null;
        this.resetGameState();
        this.emitEvent<DrlGameResetDetail<S>>('drl-game-reset', {
          state: this.currentState as S,
        });
        this.requestUpdate();
      });
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
      // Nouveau tour : les entrées à profondeur égale des tours précédents
      // ne sont plus consommables par la recherche qui démarre.
      this.tt.beginTurn();
      // Mémo des états visités par la recherche elle-même, pour les
      // minis-plateaux de la TT : on enveloppe le recorder existant.
      const baseRecorder: TreeRecorder<S> = this.recorder ?? noopRecorder;
      const recorder: TreeRecorder<S> = this.transposition
        ? {
            onVisit: (event) => {
              this.seenStates.set(game.stateKey(event.state), event.state);
              baseRecorder.onVisit(event);
            },
            onValue: (event) => baseRecorder.onValue(event),
          }
        : baseRecorder;
      const agent = createGameAgent(game, {
        mode: parseEvaluatorMode(this.evaluator),
        depth: this.depth,
        fallbackDepth: config.defaultDepth,
        transposition: this.transposition,
        transpositionTable: this.tt,
        prune: this.pruning,
        recorder,
        heuristic: config.heuristic,
        samples: this.rolloutSamples,
        discount: sanitizeDiscount(this.discount, 1),
        rng: Math.random,
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

    private renderTtCount(): TemplateResult | typeof nothing {
      if (!this.transposition) {
        return nothing;
      }
      const count = this.ttSize;
      const plural = count > 1 ? 's' : '';
      return html`<span class="drl-search">TT : ${count} entrée${plural}</span>`;
    }

    private renderAutoRestartField(): TemplateResult | typeof nothing {
      if (!this.playerKinds.every((kind) => kind === 'agent')) {
        return nothing;
      }
      return html`
        <label class="drl-field">
          <input
            type="checkbox"
            .checked=${this.autoRestart}
            @change=${this.onAutoRestartChange}
          />
          Relancer automatiquement
        </label>
      `;
    }

    private readonly onAutoRestartChange = (event: Event): void => {
      const input = event.target;
      if (input instanceof HTMLInputElement) {
        this.autoRestart = input.checked;
        this.requestUpdate();
      }
    };

    private renderStepTurnButton(): TemplateResult | typeof nothing {
      const kinds = this.playerKinds;
      if (kinds[0] !== 'agent' || kinds[1] !== 'agent' || this.outcome !== null) {
        return nothing;
      }
      return html`
        <button
          type="button"
          class="drl-button"
          title="Jouer exactement un coup de l'agent, recherche complète incluse"
          @click=${this.onStepTurnClick}
        >
          Pas à pas par coup
        </button>
      `;
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

    private renderSamplesField(): TemplateResult | typeof nothing {
      if (parseEvaluatorMode(this.evaluator) !== 'rollout') {
        return nothing;
      }
      return html`
        <label class="drl-field">
          <span>Échantillons de rollout</span>
          <input
            class="drl-number"
            type="number"
            min="1"
            step="1"
            .value=${String(this.rolloutSamples)}
            @change=${this.onSamplesInput}
          />
        </label>
      `;
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
