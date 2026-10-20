import type { PropertyValues, TemplateResult } from 'lit';
import { css, html, LitElement, nothing } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { Connect4Move } from '../../core/games/connect4.js';
import type { TicTacToeMove } from '../../core/games/tic-tac-toe.js';
import type { TreeModel } from '../../core/search/tree-model.js';
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

export type DeepBlueGameId = 'tic-tac-toe' | 'connect4';

const DEFAULT_PLAYERS = '["human","agent"]';
/** Nombre d'échantillons par défaut quand l'évaluateur rollout est choisi. */
const PAGE_ROLLOUT_SAMPLES = 10;

interface VsMode {
  readonly label: string;
  readonly players: string;
}

const VS_MODES: readonly VsMode[] = [
  { label: 'Humain vs Agent', players: '["human","agent"]' },
  { label: 'Agent vs Humain', players: '["agent","human"]' },
  { label: 'Agent vs Agent', players: '["agent","agent"]' },
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

function moveLabel(move: unknown, id: DeepBlueGameId): string {
  if (id === 'tic-tac-toe' && isTicTacToeMove(move)) {
    return `case ${move.index + 1}`;
  }
  if (id === 'connect4' && isConnect4Move(move)) {
    return `colonne ${move.col + 1}`;
  }
  return '';
}

@customElement('drl-deep-blue-page')
export class DrlDeepBluePage extends LitElement {
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

  private treeGameId: DeepBlueGameId | null = null;

  private get activeGameId(): DeepBlueGameId {
    return readGameId(this.gameKey);
  }

  private get activeSettings(): GameSettings {
    return GAME_SETTINGS[this.activeGameId];
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
  }

  protected override render(): TemplateResult {
    const settings = this.activeSettings;
    return html`
      <header class="drl-page-header">
        <h1 class="drl-page-title">Deep Blue · recherche adversariale</h1>
        <label class="drl-page-picker">
          <span>Jeu</span>
          <select aria-label="Choix du jeu" @change=${this.onGamePick}>
            <option value="tic-tac-toe" ?selected=${this.activeGameId === 'tic-tac-toe'}>
              Morpion
            </option>
            <option value="connect4" ?selected=${this.activeGameId === 'connect4'}>
              Puissance 4
            </option>
          </select>
        </label>
      </header>
      <div class="drl-page-modes">${this.renderVsModeRow()}</div>
      <div class="drl-page-search">${this.renderSearchFields()}</div>
      <div class="drl-page-layout">
        <section class="drl-page-panel">${this.renderGame()}</section>
        <section class="drl-page-panel">
          <div class="drl-page-tree-bar">
            <span class="drl-page-hint">
              Arbre d'évaluation · la recherche de l'agent se rejoue pas à pas
            </span>
            ${
              this.replayMove === null
                ? nothing
                : html`
                  <button
                    type="button"
                    class="drl-page-replay"
                    title="Applique dans la partie le premier coup du chemin menant au nœud sélectionné"
                    @click=${this.onReplayClick}
                  >
                    Rejouer ce coup (${moveLabel(this.replayMove, this.activeGameId)})
                  </button>
                `
            }
          </div>
          <drl-eval-tree
            class="drl-page-tree"
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
            ${mode.label}
          </button>
        `,
      )}
    `;
  }

  private renderSearchFields(): TemplateResult {
    const settings = this.activeSettings;
    const evaluator = parseEvaluatorMode(this.evaluator);
    return html`
      <label class="drl-page-field">
        <span>Profondeur</span>
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
        <span>Évaluateur</span>
        <select .value=${this.evaluator} @change=${this.onEvaluatorChange}>
          <option value="windows">windows</option>
          <option value="rollout">rollout</option>
          <option value="random">random</option>
        </select>
      </label>
      ${evaluator === 'rollout' ? this.renderSamplesField() : nothing}
      ${evaluator === 'random' ? nothing : this.renderDiscountField()}
      <label class="drl-page-field">
        <span>Élagage</span>
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
        Transposition
      </label>
    `;
  }

  private renderSamplesField(): TemplateResult {
    return html`
      <label class="drl-page-field">
        <span>Échantillons de rollout</span>
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
    return html`
      <label class="drl-page-field">
        <span>Actualisation (γ)</span>
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
    return html`
      <label
        class="drl-page-field"
        title="Ne coupe que sur un dépassement strict de la borne (max : > β, min : < α) et départage au hasard les coups à valeur égale. Décochez pour l'alpha-bêta classique (>= / <=, premier coup retenu à égalité)."
      >
        <input
          type="checkbox"
          .checked=${this.strictPruning}
          @change=${this.onStrictPruningChange}
        />
        Élagage strict
      </label>
    `;
  }

  private renderGame(): TemplateResult {
    if (this.activeGameId === 'connect4') {
      return html`
        <drl-connect4
          class="drl-page-game"
          .players=${this.players}
          .depth=${this.depth ?? CONNECT4_GAME_CONFIG.defaultDepth}
          .evaluator=${this.evaluator}
          .rolloutSamples=${this.rolloutSamples}
          .showSearchFields=${false}
          .transposition=${this.transposition}
          .pruning=${this.pruning}
          .strictPruning=${this.strictPruning}
          .discount=${this.discount}
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
        .players=${this.players}
        .depth=${this.depth ?? TIC_TAC_TOE_GAME_CONFIG.defaultDepth}
        .evaluator=${this.evaluator}
        .rolloutSamples=${this.rolloutSamples}
        .showSearchFields=${false}
        .transposition=${this.transposition}
        .pruning=${this.pruning}
        .strictPruning=${this.strictPruning}
        .discount=${this.discount}
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
    // L'agent rejoué ignore les entrées du tour courant : ce qui vient d'être
    // déployé par la recherche jouée apparaît donc déplié dans l'arbre.
    if (this.activeGameId === 'connect4') {
      const agent = createGameAgent(CONNECT4_GAME_CONFIG.game, {
        mode: parseEvaluatorMode(this.evaluator),
        depth: this.depth ?? CONNECT4_GAME_CONFIG.defaultDepth,
        fallbackDepth: CONNECT4_GAME_CONFIG.defaultDepth,
        transposition: this.transposition,
        transpositionTable: game?.transpositionTable,
        prune: this.pruning,
        strictPruning: this.strictPruning,
        recorder: tree.recorder,
        heuristic: CONNECT4_GAME_CONFIG.heuristic,
        samples,
        discount,
        rng: Math.random,
        probeTurns: 'previous',
      });
      tree.startSearch<Connect4Move>(agent, state);
      return;
    }
    const agent = createGameAgent(TIC_TAC_TOE_GAME_CONFIG.game, {
      mode: parseEvaluatorMode(this.evaluator),
      depth: this.depth ?? TIC_TAC_TOE_GAME_CONFIG.defaultDepth,
      fallbackDepth: TIC_TAC_TOE_GAME_CONFIG.defaultDepth,
      transposition: this.transposition,
      transpositionTable: game?.transpositionTable,
      prune: this.pruning,
      strictPruning: this.strictPruning,
      recorder: tree.recorder,
      heuristic: TIC_TAC_TOE_GAME_CONFIG.heuristic,
      samples,
      discount,
      rng: Math.random,
      probeTurns: 'previous',
    });
    tree.startSearch<TicTacToeMove>(agent, state);
  }

  private readonly onGamePick = (event: Event): void => {
    const select = event.target;
    if (!(select instanceof HTMLSelectElement)) {
      return;
    }
    this.gameKey = readGameId(select.value);
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
    this.evaluator = parseEvaluatorMode(select.value);
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
