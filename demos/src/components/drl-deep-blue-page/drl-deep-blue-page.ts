import type { PropertyValues, TemplateResult } from 'lit';
import { css, html, nothing } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { LocalizedElement } from '../../base/localized-element.js';
import type { Evaluator } from '../../core/contracts.js';
import type { Connect4Move, Connect4State } from '../../core/games/connect4.js';
import type { TicTacToeMove, TicTacToeState } from '../../core/games/tic-tac-toe.js';
import {
  compileCustomEvaluator,
  defaultCustomSource,
} from '../../core/search/evaluators/custom.js';
import { createReadOnlyOverlay } from '../../core/search/transposition.js';
import type { TreeModel } from '../../core/search/tree-model.js';
import type { Locale } from '../../i18n/locale.js';
import '../drl-connect4/drl-connect4.js';
import '../drl-tic-tac-toe/drl-tic-tac-toe.js';
import { drlTokens } from '../../styles/tokens.js';
import { DrlEvalTree, type DrlRootChangedEvent } from '../drl-eval-tree/drl-eval-tree.js';
import type { PieceStyle } from '../drl-game-board/drl-game-board.js';
import {
  CONNECT4_GAME_CONFIG,
  createGameAgent,
  type GameComponentApi,
  parseEvaluatorMode,
  parsePlayersConfig,
  sanitizeDepth,
  sanitizeDiscount,
  sanitizeSamples,
  TIC_TAC_TOE_GAME_CONFIG,
} from '../shared/create-game-component.js';
import { type DeepBlueMessageKey, deepBlueMessage } from './messages.js';

export type DeepBlueGameId = 'tic-tac-toe' | 'connect4';

const DEFAULT_PLAYERS = '["human","agent"]';
/** Nombre d'échantillons par défaut quand l'évaluateur rollout est choisi. */
const PAGE_ROLLOUT_SAMPLES = 10;

interface VsMode {
  readonly labelKey: DeepBlueMessageKey;
  readonly players: string;
}

const VS_MODES: readonly VsMode[] = [
  { labelKey: 'modeHumanAgent', players: '["human","agent"]' },
  { labelKey: 'modeAgentHuman', players: '["agent","human"]' },
  { labelKey: 'modeAgentAgent', players: '["agent","agent"]' },
];

/** Compare deux configurations de joueurs indépendamment des espaces JSON. */
function samePlayers(left: string, right: string): boolean {
  const [leftFirst, leftSecond] = parsePlayersConfig(left);
  const [rightFirst, rightSecond] = parsePlayersConfig(right);
  return leftFirst === rightFirst && leftSecond === rightSecond;
}

interface GameSettings {
  readonly pieceStyle: PieceStyle;
  readonly nodeSize: string;
  readonly defaultDepth: number;
}

const GAME_SETTINGS: Record<DeepBlueGameId, GameSettings> = {
  'tic-tac-toe': {
    pieceStyle: TIC_TAC_TOE_GAME_CONFIG.pieceStyle,
    nodeSize: '4rem',
    defaultDepth: TIC_TAC_TOE_GAME_CONFIG.defaultDepth,
  },
  connect4: {
    pieceStyle: CONNECT4_GAME_CONFIG.pieceStyle,
    nodeSize: '3rem',
    defaultDepth: CONNECT4_GAME_CONFIG.defaultDepth,
  },
};

function readGameId(value: unknown): DeepBlueGameId {
  return value === 'connect4' ? 'connect4' : 'tic-tac-toe';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isTicTacToeMove(value: unknown): value is TicTacToeMove {
  return isRecord(value) && typeof value.index === 'number';
}

function isConnect4Move(value: unknown): value is Connect4Move {
  return isRecord(value) && typeof value.col === 'number';
}

export function firstMoveOnPath<S>(model: TreeModel<S>, nodeKey: string | null): unknown {
  if (nodeKey === null || !model.has(nodeKey)) {
    return null;
  }
  const root = model.root();
  if (root === null || root.nodeKey === nodeKey) {
    return null;
  }
  let cursor = model.node(nodeKey);
  while (cursor !== null && cursor.parentKey !== null) {
    const parent = model.node(cursor.parentKey);
    if (parent === null) {
      return null;
    }
    if (parent.nodeKey === root.nodeKey) {
      return cursor.move ?? null;
    }
    cursor = parent;
  }
  return null;
}

function moveLabel(move: unknown, id: DeepBlueGameId, locale: Locale = 'fr'): string {
  if (id === 'tic-tac-toe' && isTicTacToeMove(move)) {
    return deepBlueMessage(locale, 'moveCase', { number: move.index + 1 });
  }
  if (id === 'connect4' && isConnect4Move(move)) {
    return deepBlueMessage(locale, 'moveColumn', { number: move.col + 1 });
  }
  return '';
}

@customElement('drl-deep-blue-page')
export class DrlDeepBluePage extends LocalizedElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: block;
        color: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: var(--drl-font-size);
      }

      .drl-page-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-3);
      }

      .drl-page-title {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 700;
      }

      .drl-page-picker {
        display: inline-flex;
        align-items: center;
        gap: var(--drl-space-1);
        color: var(--drl-muted);
      }

      .drl-page-picker select {
        font: inherit;
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-surface);
        color: var(--drl-fg);
        cursor: pointer;
      }

      .drl-page-picker select:hover {
        border-color: var(--drl-accent);
      }

      .drl-page-modes {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-3);
      }

      .drl-page-mode {
        font: inherit;
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-bg);
        color: var(--drl-fg);
        cursor: pointer;
      }

      .drl-page-mode:hover {
        border-color: var(--drl-accent);
      }

      .drl-page-mode[aria-pressed='true'] {
        border-color: var(--drl-accent);
        color: var(--drl-accent);
      }

      .drl-page-search {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-3);
      }

      .drl-page-field {
        display: inline-flex;
        align-items: center;
        gap: var(--drl-space-1);
        color: var(--drl-muted);
      }

      .drl-page-field select {
        font: inherit;
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-surface);
        color: var(--drl-fg);
        cursor: pointer;
      }

      .drl-page-field select:hover {
        border-color: var(--drl-accent);
      }

      .drl-page-number {
        width: calc(var(--drl-space-3) * 5);
        max-width: 100%;
        box-sizing: border-box;
        font: inherit;
        padding: 2px 4px;
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-surface);
        color: var(--drl-fg);
      }

      .drl-page-custom {
        width: 100%;
        margin-bottom: var(--drl-space-3);
      }

      .drl-page-custom-label {
        display: block;
        font-weight: 700;
        margin-bottom: var(--drl-space-1);
      }

      .drl-page-custom-source {
        display: block;
        width: 100%;
        min-height: 12rem;
        box-sizing: border-box;
        font-family: monospace;
        font-size: 0.9em;
        line-height: 1.5;
        tab-size: 2;
        resize: vertical;
        padding: var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
        background: var(--drl-surface);
        color: var(--drl-fg);
      }

      .drl-page-custom-actions {
        display: flex;
        flex-wrap: wrap;
        gap: var(--drl-space-2);
        margin-top: var(--drl-space-2);
      }

      .drl-page-custom-actions button {
        font: inherit;
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-bg);
        color: var(--drl-fg);
        cursor: pointer;
      }

      .drl-page-custom-actions button:hover {
        border-color: var(--drl-accent);
      }

      .drl-page-custom-warning {
        margin: var(--drl-space-1) 0 0;
        padding: var(--drl-space-1) var(--drl-space-2);
        border-left: 3px solid var(--drl-arrow);
        color: var(--drl-arrow);
      }

      .drl-page-error {
        margin: var(--drl-space-2) 0 0;
        color: var(--drl-arrow);
        white-space: pre-wrap;
      }

      .drl-page-layout {
        display: grid;
        grid-template-columns: minmax(0, 2fr) minmax(0, 3fr);
        gap: var(--drl-space-3);
        align-items: start;
      }

      @media (max-width: 900px) {
        .drl-page-layout {
          grid-template-columns: minmax(0, 1fr);
        }
      }

      .drl-page-panel {
        min-width: 0;
        padding: var(--drl-space-3);
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
      }

      .drl-page-tree-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-2);
        min-height: 2rem;
      }

      .drl-page-hint {
        color: var(--drl-muted);
      }

      .drl-page-replay {
        font: inherit;
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-accent);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-bg);
        color: var(--drl-fg);
        cursor: pointer;
      }

      .drl-page-replay:hover {
        border-color: var(--drl-accent-2);
      }
    `,
  ];

  @property({ attribute: 'game' })
  gameKey = 'tic-tac-toe';

  @property({ attribute: 'players' })
  players = DEFAULT_PLAYERS;

  @property({ type: Number, attribute: 'depth' })
  depth: number | null = null;

  @property({ attribute: 'evaluator' })
  evaluator = 'windows';

  @property({ type: Number, attribute: 'rollout-samples' })
  rolloutSamples = PAGE_ROLLOUT_SAMPLES;

  @property({ type: Boolean, attribute: 'auto-fit' })
  autoFit = true;

  @property({ type: Boolean, attribute: 'transposition' })
  transposition = false;

  @property({ type: Boolean, attribute: 'pruning' })
  pruning = false;

  @property({ type: Boolean, attribute: 'strict-pruning' })
  strictPruning = false;

  @property({ type: Number, attribute: 'discount' })
  discount = 1;

  @state()
  replayMove: unknown = null;

  // Brouillon par jeu du code source personnalisé. Volontairement hors de
  // l'état réactif : chaque frappe relancerait un rendu et replacerait le
  // curseur du champ.
  private customSources: Record<DeepBlueGameId, string> = {
    'tic-tac-toe': defaultCustomSource('tic-tac-toe'),
    connect4: defaultCustomSource('connect4'),
  };

  @state()
  private customEvaluators: Partial<Record<DeepBlueGameId, Evaluator<unknown, unknown>>> = {};

  @state()
  private customError: string | null = null;

  private treeGameId: DeepBlueGameId | null = null;

  private get activeGameId(): DeepBlueGameId {
    return readGameId(this.gameKey);
  }

  private get activeSettings(): GameSettings {
    return GAME_SETTINGS[this.activeGameId];
  }

  /** Évaluateur compilé actif, uniquement quand l'évaluateur choisi est `custom`. */
  private get activeCustomEvaluator(): Evaluator<unknown, unknown> | null {
    if (parseEvaluatorMode(this.evaluator) !== 'custom') {
      return null;
    }
    return this.customEvaluators[this.activeGameId] ?? null;
  }

  /** Compile le brouillon du jeu donné si aucun évaluateur n'y est encore stocké. */
  private ensureCustomEvaluator(gameId: DeepBlueGameId): void {
    if (this.customEvaluators[gameId] !== undefined) {
      return;
    }
    const game = gameId === 'connect4' ? CONNECT4_GAME_CONFIG.game : TIC_TAC_TOE_GAME_CONFIG.game;
    const result = compileCustomEvaluator(this.customSources[gameId], game, gameId);
    if (result.ok) {
      this.customEvaluators = { ...this.customEvaluators, [gameId]: result.evaluator };
    }
  }

  private get gameElement(): GameComponentApi<unknown, unknown> | null {
    return this.renderRoot.querySelector<GameComponentApi<unknown, unknown>>('.drl-page-game');
  }

  private get treeElement(): DrlEvalTree<unknown> | null {
    const element = this.renderRoot.querySelector('.drl-page-tree');
    if (!(element instanceof DrlEvalTree)) {
      return null;
    }
    return element as DrlEvalTree<unknown>;
  }

  protected override updated(changed: PropertyValues<this>): void {
    super.updated(changed);
    if (this.treeGameId !== this.activeGameId) {
      this.syncTreeGame();
    }
    // Une page créée directement en `custom` (attribut ou propriété avant le
    // premier rendu) n'a pas encore d'évaluateur compilé : on prépare ici le
    // défaut du jeu actif. L'appel est idempotent, donc pas de boucle de rendu.
    if (parseEvaluatorMode(this.evaluator) === 'custom') {
      this.ensureCustomEvaluator(this.activeGameId);
    }
  }

  protected override render(): TemplateResult {
    const locale = this.locale;
    const settings = this.activeSettings;
    return html`
      <header class="drl-page-header">
        <h1 class="drl-page-title">${deepBlueMessage(locale, 'title')}</h1>
        <label class="drl-page-picker">
          <span>${deepBlueMessage(locale, 'game')}</span>
          <select aria-label=${deepBlueMessage(locale, 'gameAriaLabel')} @change=${this.onGamePick}>
            <option value="tic-tac-toe" ?selected=${this.activeGameId === 'tic-tac-toe'}>
              ${deepBlueMessage(locale, 'ticTacToe')}
            </option>
            <option value="connect4" ?selected=${this.activeGameId === 'connect4'}>
              ${deepBlueMessage(locale, 'connect4')}
            </option>
          </select>
        </label>
      </header>
      <div class="drl-page-modes">${this.renderVsModeRow()}</div>
      <div class="drl-page-search">${this.renderSearchFields()}</div>
      ${this.renderCustomBlock()}
      <div class="drl-page-layout">
        <section class="drl-page-panel">${this.renderGame()}</section>
        <section class="drl-page-panel">
          <div class="drl-page-tree-bar">
            <span class="drl-page-hint">
              ${deepBlueMessage(locale, 'treeHint')}
            </span>
            ${
              this.replayMove === null
                ? nothing
                : html`
                  <button
                    type="button"
                    class="drl-page-replay"
                    title=${deepBlueMessage(locale, 'replayTitle')}
                    @click=${this.onReplayClick}
                  >
                    ${deepBlueMessage(locale, 'replay', {
                      move: moveLabel(this.replayMove, this.activeGameId, locale),
                    })}
                  </button>
                `
            }
          </div>
          <drl-eval-tree
            class="drl-page-tree"
            lang=${locale}
            .pieceStyle=${settings.pieceStyle}
            .nodeSize=${settings.nodeSize}
            .autoFit=${this.autoFit}
            @drl-root-changed=${this.onRootChanged}
          ></drl-eval-tree>
        </section>
      </div>
    `;
  }

  private renderVsModeRow(): TemplateResult {
    const locale = this.locale;
    return html`
      ${VS_MODES.map(
        (mode) => html`
          <button
            type="button"
            class="drl-page-mode"
            data-players=${mode.players}
            aria-pressed=${samePlayers(this.players, mode.players)}
            @click=${this.onModePick}
          >
            ${deepBlueMessage(locale, mode.labelKey)}
          </button>
        `,
      )}
    `;
  }

  private renderSearchFields(): TemplateResult {
    const locale = this.locale;
    const settings = this.activeSettings;
    const evaluator = parseEvaluatorMode(this.evaluator);
    return html`
      <label class="drl-page-field">
        <span>${deepBlueMessage(locale, 'depth')}</span>
        <input
          class="drl-page-number"
          type="number"
          min="1"
          step="1"
          placeholder=${settings.defaultDepth}
          .value=${this.depth === null ? '' : String(this.depth)}
          @change=${this.onDepthChange}
        />
      </label>
      <label class="drl-page-field">
        <span>${deepBlueMessage(locale, 'evaluator')}</span>
        <select .value=${this.evaluator} @change=${this.onEvaluatorChange}>
          <option value="windows">windows</option>
          <option value="rollout">rollout</option>
          <option value="random">random</option>
          <option value="custom">custom</option>
        </select>
      </label>
      ${evaluator === 'rollout' ? this.renderSamplesField() : nothing}
      ${evaluator === 'random' ? nothing : this.renderDiscountField()}
      <label class="drl-page-field">
        <span>${deepBlueMessage(locale, 'pruning')}</span>
        <select .value=${this.pruning ? 'alpha-beta' : 'minimax'} @change=${this.onPruningChange}>
          <option value="alpha-beta">alpha-beta</option>
          <option value="minimax">minimax</option>
        </select>
      </label>
      ${evaluator === 'random' ? nothing : this.renderStrictPruningField()}
      <label class="drl-page-field">
        <input
          type="checkbox"
          .checked=${this.transposition}
          @change=${this.onTranspositionChange}
        />
        ${deepBlueMessage(locale, 'transposition')}
      </label>
    `;
  }

  private renderCustomBlock(): TemplateResult | typeof nothing {
    if (parseEvaluatorMode(this.evaluator) !== 'custom') {
      return nothing;
    }
    const locale = this.locale;
    return html`
      <section class="drl-page-custom">
        <label class="drl-page-custom-label" for="drl-custom-source">
          ${deepBlueMessage(locale, 'customEvaluator')}
        </label>
        <p class="drl-page-hint">
          ${deepBlueMessage(locale, 'customHelpReceives')}<code>state</code>${deepBlueMessage(
            locale,
            'customHelpAnd',
          )}<code>ctx</code>${deepBlueMessage(locale, 'customHelpFinite')}<code>ctx</code>${deepBlueMessage(
            locale,
            'customHelpExposes',
          )}<code>rootPlayer</code>, <code>opponent</code>, <code>rows</code>,
          <code>cols</code>, <code>game</code>${deepBlueMessage(
            locale,
            'customHelpThen',
          )}<code>lines</code>${deepBlueMessage(
            locale,
            'customHelpTicTacToe',
          )}<code>windows</code>${deepBlueMessage(locale, 'customHelpConnect4')}
        </p>
        <p class="drl-page-custom-warning" id="drl-custom-warning">
          ${deepBlueMessage(locale, 'customWarning')}
        </p>
        <textarea
          id="drl-custom-source"
          class="drl-page-custom-source"
          spellcheck="false"
          autocomplete="off"
          aria-describedby="drl-custom-warning"
          .value=${this.customSources[this.activeGameId]}
          @input=${this.onCustomInput}
        ></textarea>
        <div class="drl-page-custom-actions">
          <button type="button" @click=${this.onCustomApply}>
            ${deepBlueMessage(locale, 'apply')}
          </button>
          <button type="button" @click=${this.onCustomReset}>
            ${deepBlueMessage(locale, 'reset')}
          </button>
        </div>
        ${
          this.customError === null
            ? nothing
            : html`<p class="drl-page-error" role="alert">${this.customError}</p>`
        }
      </section>
    `;
  }

  private renderSamplesField(): TemplateResult {
    const locale = this.locale;
    return html`
      <label class="drl-page-field">
        <span>${deepBlueMessage(locale, 'rolloutSamples')}</span>
        <input
          class="drl-page-number"
          type="number"
          min="1"
          step="1"
          .value=${String(this.rolloutSamples)}
          @change=${this.onSamplesChange}
        />
      </label>
    `;
  }

  private renderDiscountField(): TemplateResult {
    const locale = this.locale;
    return html`
      <label class="drl-page-field">
        <span>${deepBlueMessage(locale, 'discount')}</span>
        <input
          class="drl-page-number"
          type="number"
          min="0.05"
          max="1"
          step="0.01"
          .value=${this.discount.toFixed(2)}
          @change=${this.onDiscountChange}
        />
      </label>
    `;
  }

  private renderStrictPruningField(): TemplateResult {
    const locale = this.locale;
    return html`
      <label
        class="drl-page-field"
        title=${deepBlueMessage(locale, 'strictPruningTitle')}
      >
        <input
          type="checkbox"
          .checked=${this.strictPruning}
          @change=${this.onStrictPruningChange}
        />
        ${deepBlueMessage(locale, 'strictPruning')}
      </label>
    `;
  }

  private renderGame(): TemplateResult {
    if (this.activeGameId === 'connect4') {
      return html`
        <drl-connect4
          class="drl-page-game"
          lang=${this.locale}
          .players=${this.players}
          .depth=${this.depth ?? CONNECT4_GAME_CONFIG.defaultDepth}
          .evaluator=${this.evaluator}
          .rolloutSamples=${this.rolloutSamples}
          .showSearchFields=${false}
          .transposition=${this.transposition}
          .pruning=${this.pruning}
          .strictPruning=${this.strictPruning}
          .discount=${this.discount}
          .customEvaluator=${this.activeCustomEvaluator}
          @drl-search-started=${this.onSearchStarted}
          @drl-step=${this.onGameStep}
          @drl-game-reset=${this.onTreeReset}
          @drl-move-undone=${this.onMoveUndone}
        ></drl-connect4>
      `;
    }
    return html`
      <drl-tic-tac-toe
        class="drl-page-game"
        lang=${this.locale}
        .players=${this.players}
        .depth=${this.depth ?? TIC_TAC_TOE_GAME_CONFIG.defaultDepth}
        .evaluator=${this.evaluator}
        .rolloutSamples=${this.rolloutSamples}
        .showSearchFields=${false}
        .transposition=${this.transposition}
        .pruning=${this.pruning}
        .strictPruning=${this.strictPruning}
        .discount=${this.discount}
        .customEvaluator=${this.activeCustomEvaluator}
        @drl-search-started=${this.onSearchStarted}
        @drl-step=${this.onGameStep}
        @drl-game-reset=${this.onTreeReset}
        @drl-move-undone=${this.onMoveUndone}
      ></drl-tic-tac-toe>
    `;
  }

  private syncTreeGame(): void {
    const tree = this.treeElement;
    const game = this.gameElement;
    if (tree === null || game === null) {
      return;
    }
    this.treeGameId = this.activeGameId;
    tree.clearTree();
    tree.attach(game.game);
    this.replayMove = null;
  }

  private startParallelSearch(state: unknown): void {
    const tree = this.treeElement;
    const game = this.gameElement;
    if (tree === null) {
      return;
    }
    const samples = this.rolloutSamples;
    const discount = sanitizeDiscount(this.discount, 1);
    // Le rejoué lit la table partagée en lecture seule et écrit dans sa propre
    // table temporaire : la table réelle n'est jamais polluée par le rejoué.
    const realTable = game?.transpositionTable;
    // L'overlay n'est construit que si la transposition est active : sinon on
    // passe `undefined` et le rejoué n'utilise aucune table.
    const replayTable =
      this.transposition && realTable !== undefined ? createReadOnlyOverlay(realTable) : undefined;
    if (this.activeGameId === 'connect4') {
      const agent = createGameAgent(CONNECT4_GAME_CONFIG.game, {
        mode: parseEvaluatorMode(this.evaluator),
        depth: this.depth ?? CONNECT4_GAME_CONFIG.defaultDepth,
        fallbackDepth: CONNECT4_GAME_CONFIG.defaultDepth,
        transposition: this.transposition,
        transpositionTable: replayTable,
        prune: this.pruning,
        strictPruning: this.strictPruning,
        recorder: tree.recorder,
        heuristic: CONNECT4_GAME_CONFIG.heuristic,
        // Le champ n'a de sens qu'en mode `custom` : ailleurs l'agent l'ignore,
        // mais on l'omet pour que l'intention reste lisible.
        customEvaluator:
          parseEvaluatorMode(this.evaluator) === 'custom'
            ? (this.customEvaluators[this.activeGameId] as
                | Evaluator<Connect4State, Connect4Move>
                | undefined)
            : undefined,
        samples,
        discount,
        rng: Math.random,
      });
      tree.startSearch<Connect4Move>(agent, state);
      return;
    }
    const agent = createGameAgent(TIC_TAC_TOE_GAME_CONFIG.game, {
      mode: parseEvaluatorMode(this.evaluator),
      depth: this.depth ?? TIC_TAC_TOE_GAME_CONFIG.defaultDepth,
      fallbackDepth: TIC_TAC_TOE_GAME_CONFIG.defaultDepth,
      transposition: this.transposition,
      transpositionTable: replayTable,
      prune: this.pruning,
      strictPruning: this.strictPruning,
      recorder: tree.recorder,
      heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
      // Le champ n'a de sens qu'en mode `custom` : ailleurs l'agent l'ignore,
      // mais on l'omet pour que l'intention reste lisible.
      customEvaluator:
        parseEvaluatorMode(this.evaluator) === 'custom'
          ? (this.customEvaluators[this.activeGameId] as
              | Evaluator<TicTacToeState, TicTacToeMove>
              | undefined)
          : undefined,
      samples,
      discount,
      rng: Math.random,
    });
    tree.startSearch<TicTacToeMove>(agent, state);
  }

  private readonly onGamePick = (event: Event): void => {
    const select = event.target as HTMLSelectElement | null;
    if (select === null || typeof select.value !== 'string') {
      return;
    }
    this.gameKey = readGameId(select.value);
    this.customError = null;
    // Le jeu peut n'avoir jamais été compilé : on prépare son défaut pour que
    // le sélecteur `custom` et le rejoué disposent tout de suite d'un évaluateur.
    if (parseEvaluatorMode(this.evaluator) === 'custom') {
      this.ensureCustomEvaluator(this.activeGameId);
    }
  };

  private readonly onModePick = (event: Event): void => {
    const button = event.target as HTMLElement | null;
    const players = button?.dataset.players;
    if (players !== undefined) {
      this.players = players;
    }
  };

  private readonly onDepthChange = (event: Event): void => {
    const input = event.target as HTMLInputElement | null;
    if (input === null || typeof input.value !== 'string') {
      return;
    }
    this.depth =
      input.value === ''
        ? null
        : sanitizeDepth(Number(input.value), this.activeSettings.defaultDepth);
  };

  private readonly onEvaluatorChange = (event: Event): void => {
    const select = event.target as HTMLSelectElement | null;
    if (select === null || typeof select.value !== 'string') {
      return;
    }
    const mode = parseEvaluatorMode(select.value);
    this.evaluator = mode;
    // Passer en `custom` compile le défaut du jeu actif s'il ne l'est pas déjà.
    // Revenir à un autre évaluateur laisse brouillons et évaluateurs en place :
    // un retour en `custom` retrouve alors l'état précédent.
    if (mode === 'custom') {
      this.ensureCustomEvaluator(this.activeGameId);
    }
  };

  private readonly onCustomInput = (event: Event): void => {
    const textarea = event.target as HTMLTextAreaElement | null;
    if (textarea === null || typeof textarea.value !== 'string') {
      return;
    }
    // Brouillon hors état réactif : aucune mise à jour déclenchée, le curseur
    // du champ ne bouge pas pendant la frappe.
    this.customSources[this.activeGameId] = textarea.value;
  };

  private readonly onCustomApply = (): void => {
    const gameId = this.activeGameId;
    const game = gameId === 'connect4' ? CONNECT4_GAME_CONFIG.game : TIC_TAC_TOE_GAME_CONFIG.game;
    const result = compileCustomEvaluator(this.customSources[gameId], game, gameId);
    if (!result.ok) {
      // La source est invalide : l'ancien évaluateur reste en place.
      this.customError = result.error;
      return;
    }
    this.customEvaluators = { ...this.customEvaluators, [gameId]: result.evaluator };
    this.customError = null;
    // Le rejoué doit repartir de zéro avec le nouvel évaluateur.
    this.treeElement?.clearTree();
    // Le champ et les liaisons enfants sont rafraîchis même si l'identité de
    // l'état n'a pas changé (brouillon appliqué à l'identique).
    this.requestUpdate();
  };

  private readonly onCustomReset = (): void => {
    const gameId = this.activeGameId;
    const source = defaultCustomSource(gameId);
    this.customSources[gameId] = source;
    const game = gameId === 'connect4' ? CONNECT4_GAME_CONFIG.game : TIC_TAC_TOE_GAME_CONFIG.game;
    const result = compileCustomEvaluator(source, game, gameId);
    this.customError = null;
    if (result.ok) {
      this.customEvaluators = { ...this.customEvaluators, [gameId]: result.evaluator };
    }
    this.treeElement?.clearTree();
    this.requestUpdate();
  };

  private readonly onSamplesChange = (event: Event): void => {
    const input = event.target as HTMLInputElement | null;
    if (input === null || typeof input.value !== 'string') {
      return;
    }
    this.rolloutSamples = sanitizeSamples(Number(input.value), PAGE_ROLLOUT_SAMPLES);
  };

  private readonly onDiscountChange = (event: Event): void => {
    const input = event.target as HTMLInputElement | null;
    if (input === null || typeof input.value !== 'string') {
      return;
    }
    this.discount = sanitizeDiscount(Number(input.value), 1);
  };

  private readonly onPruningChange = (event: Event): void => {
    const select = event.target as HTMLSelectElement | null;
    if (select === null || typeof select.value !== 'string') {
      return;
    }
    this.pruning = select.value === 'alpha-beta';
  };

  private readonly onStrictPruningChange = (event: Event): void => {
    const input = event.target as HTMLInputElement | null;
    if (input === null || typeof input.checked !== 'boolean') {
      return;
    }
    this.strictPruning = input.checked;
  };

  private readonly onTranspositionChange = (event: Event): void => {
    const input = event.target as HTMLInputElement | null;
    if (input === null || typeof input.checked !== 'boolean') {
      return;
    }
    this.transposition = input.checked;
  };

  private readonly onSearchStarted = (event: Event): void => {
    const { state } = (event as CustomEvent<{ state: unknown }>).detail;
    this.replayMove = null;
    this.startParallelSearch(state);
  };

  // Le bouton Step de la partie est le même geste pour l'arbre : on avance la
  // recherche rejouée d'un pas. On respecte la pause propre à l'arbre, car son
  // bouton Step gère déjà ce cas de son côté.
  private readonly onGameStep = (): void => {
    const tree = this.treeElement;
    if (tree !== null && !tree.running) {
      tree.step();
    }
  };

  private readonly onTreeReset = (): void => {
    this.replayMove = null;
    this.treeElement?.clearTree();
  };

  private readonly onMoveUndone = (event: Event): void => {
    const { state } = (event as CustomEvent<{ state: unknown }>).detail;
    this.replayMove = null;
    this.startParallelSearch(state);
  };

  private readonly onRootChanged = (event: Event): void => {
    const tree = this.treeElement;
    if (tree === null) {
      return;
    }
    const { nodeKey } = (event as DrlRootChangedEvent).detail;
    this.replayMove = firstMoveOnPath(tree.model, nodeKey);
  };

  private readonly onReplayClick = (): void => {
    const move = this.replayMove;
    const game = this.gameElement;
    if (move === null || game === null) {
      return;
    }
    if (this.activeGameId === 'connect4') {
      if (isConnect4Move(move) && game.applyMove(move)) {
        this.replayMove = null;
      }
      return;
    }
    if (isTicTacToeMove(move) && game.applyMove(move)) {
      this.replayMove = null;
    }
  };
}
