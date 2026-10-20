import { type Selection, select } from 'd3';
import { type HierarchyNode, hierarchy, tree } from 'd3-hierarchy';
import {
  type D3ZoomEvent,
  zoom as d3zoom,
  type ZoomBehavior,
  type ZoomTransform,
  zoomIdentity,
} from 'd3-zoom';
import type { PropertyValues, TemplateResult } from 'lit';
import { css, html, nothing } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { DemoElement } from '../../base/demo-element.js';
import type {
  Game,
  SearchAgent,
  TreeRecorder,
  ValueEvent,
  VisitEvent,
} from '../../core/contracts.js';
import { createStepper, type Stepper } from '../../core/search/stepper.js';
import { createModelRecorder, TreeModel, type TreeNode } from '../../core/search/tree-model.js';
import { drlTokens } from '../../styles/tokens.js';
import { DrlGameBoard, type PieceStyle } from '../drl-game-board/drl-game-board.js';
import {
  boundsSummary,
  cutDescription,
  cutLabel,
  formatBound,
  formatEntryDepth,
  formatMove,
  formatScore,
  keepFirstPruned,
  NO_CUT_LABEL,
} from './display.js';
import '../../ui/drl-controls.js';
import '../../ui/drl-slider.js';

export interface DrlRootChangedDetail {
  readonly nodeKey: string | null;
}

export type DrlRootChangedEvent = CustomEvent<DrlRootChangedDetail>;

declare global {
  interface HTMLElementEventMap {
    'drl-root-changed': DrlRootChangedEvent;
  }
}

type StepPrimitive = 'in' | 'over' | 'out';

interface LayoutItem<S> {
  readonly key: string;
  readonly node: TreeNode<S>;
  readonly compact: boolean;
  readonly collapsedCount: number;
  readonly children?: LayoutItem<S>[];
}

interface PositionedItem<S> {
  readonly item: LayoutItem<S>;
  readonly x: number;
  readonly y: number;
  readonly halfHeight: number;
}

interface EdgeDatum {
  readonly key: string;
  readonly sx: number;
  readonly sy: number;
  readonly tx: number;
  readonly ty: number;
  readonly best: boolean;
  readonly cut: boolean;
  readonly label: string;
}

interface Crumb {
  readonly key: string;
  readonly label: string;
  readonly current: boolean;
}

/** Panneau α/β : chaînes déjà formatées pour éviter tout calcul au rendu. */
interface BoundsView {
  readonly key: string;
  readonly title: string;
  readonly alpha: string;
  readonly beta: string;
  readonly score: string;
  readonly exact: boolean;
  readonly cut: string | null;
}

const MAX_RENDER_NODES = 4000;
const LAYOUT_MARGIN = 24;
const BOARD_CELL_SIZE = 24;
const NODE_GAP_X = 20;
const NODE_GAP_Y = 46;
const CARD_EXTRA = 22;
const COMPACT_HALF_HEIGHT = 12;
const ZOOM_MIN = 0.15;
const ZOOM_MAX = 4;
const DEFAULT_BOX = 48;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function scoreClass(score: number | null, exact: boolean): string {
  let classes = 'drl-node-score';
  if (score !== null) {
    if (score > 0) {
      classes += ' drl-node-score--pos';
    } else if (score < 0) {
      classes += ' drl-node-score--neg';
    }
    classes += exact ? ' drl-node-score--exact' : ' drl-node-score--provisional';
  }
  return classes;
}

function autoPieceStyle(view: { rows: number; cols: number }): PieceStyle {
  return view.rows <= 4 && view.cols <= 4 ? 'marks' : 'discs';
}

@customElement('drl-eval-tree')
export class DrlEvalTree<S> extends DemoElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: block;
        color: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: var(--drl-font-size);
      }

      .drl-toolbar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-2);
      }

      .drl-button {
        font: inherit;
        display: inline-flex;
        align-items: center;
        gap: var(--drl-space-1);
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

      .drl-zoom-slider {
        flex: 0 1 180px;
        min-width: 140px;
      }

      .drl-depth {
        display: inline-flex;
        align-items: center;
        gap: var(--drl-space-1);
        color: var(--drl-muted);
      }

      .drl-autofit {
        display: inline-flex;
        align-items: center;
        gap: var(--drl-space-1);
        color: var(--drl-muted);
        cursor: pointer;
      }

      .drl-autofit input {
        cursor: pointer;
      }

      .drl-depth-input {
        width: 4.5rem;
        font: inherit;
        padding: 2px 4px;
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-surface);
        color: var(--drl-fg);
      }

      .drl-breadcrumbs {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-1);
        min-height: 1.6em;
        margin-bottom: var(--drl-space-1);
        font-size: 0.9em;
      }

      .drl-breadcrumbs span {
        color: var(--drl-accent);
        cursor: pointer;
        padding: 0 var(--drl-space-1);
      }

      .drl-breadcrumbs span:hover {
        text-decoration: underline;
      }

      .drl-breadcrumbs span + span::before {
        content: '›';
        color: var(--drl-muted);
        margin-right: var(--drl-space-1);
      }

      .drl-breadcrumbs span[aria-current='true'] {
        color: var(--drl-fg);
        font-weight: 700;
        cursor: default;
        text-decoration: none;
      }

      .drl-bounds {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: var(--drl-space-2);
        min-height: 1.6em;
        margin-bottom: var(--drl-space-1);
        color: var(--drl-muted);
        font-size: 0.9em;
        font-variant-numeric: tabular-nums;
      }

      /* nowrap : le libellé neutre ou la description de coupure tiennent sur une
         seule ligne, donc la hauteur du panneau reste constante entre les deux. */
      .drl-bounds-item {
        white-space: nowrap;
      }

      .drl-bounds-title {
        color: var(--drl-fg);
        font-weight: 700;
      }

      .drl-bounds-cut {
        color: var(--drl-arrow);
      }

      .drl-canvas {
        position: relative;
        overflow: hidden;
        height: 460px;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
        touch-action: none;
        cursor: grab;
      }

      .drl-canvas:active {
        cursor: grabbing;
      }

      .drl-viewport {
        position: absolute;
        top: 0;
        left: 0;
        transform-origin: 0 0;
        will-change: transform;
      }

      .drl-edges {
        position: absolute;
        top: 0;
        left: 0;
        overflow: visible;
        pointer-events: none;
      }

      .drl-edge {
        fill: none;
        stroke: var(--drl-border);
        stroke-width: 1.5;
      }

      .drl-edge--best {
        stroke: var(--drl-accent);
        stroke-width: 2.5;
      }

      /* Après --best : une arête coupée doit primer sur l'arête « meilleure ». */
      .drl-edge--cut {
        stroke: var(--drl-arrow);
        stroke-width: 2;
        stroke-dasharray: 6 4;
      }

      .drl-edge-label {
        fill: var(--drl-muted);
        font-family: var(--drl-font);
        font-size: 10px;
      }

      .drl-node-layer {
        position: absolute;
        top: 0;
        left: 0;
      }

      .drl-node {
        position: absolute;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        align-items: stretch;
        gap: 1px;
        padding: 2px;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: 6px;
        cursor: pointer;
        user-select: none;
      }

      .drl-node:hover {
        border-color: var(--drl-accent);
      }

      .drl-node-board {
        position: relative;
        line-height: 0;
        overflow: hidden;
        border-radius: 4px;
      }

      .drl-node--terminal {
        border-color: var(--drl-fg);
      }

      .drl-node--viewroot {
        box-shadow: 0 0 0 2px var(--drl-accent);
      }

      .drl-node--cursor {
        box-shadow: 0 0 0 2px var(--drl-accent-2);
      }

      .drl-node--pruned .drl-node-board {
        filter: grayscale(1);
        opacity: 0.55;
      }

      .drl-node--pruned .drl-node-board::after {
        content: '';
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(
          45deg,
          rgba(100, 116, 139, 0.35) 0 3px,
          transparent 3px 7px
        );
      }

      .drl-node-badge {
        position: absolute;
        top: -7px;
        right: -7px;
        display: none;
        min-width: 18px;
        height: 16px;
        padding: 0 3px;
        box-sizing: border-box;
        border-radius: 9px;
        background: var(--drl-accent-2);
        color: var(--drl-surface);
        font-size: 11px;
        line-height: 16px;
        text-align: center;
        font-variant-numeric: tabular-nums;
      }

      .drl-node--tt-hit .drl-node-badge {
        display: block;
      }

      .drl-node-score {
        min-height: 14px;
        font-size: 11px;
        line-height: 14px;
        text-align: center;
        font-variant-numeric: tabular-nums;
      }

      .drl-node-score--pos {
        color: var(--drl-accent);
      }

      .drl-node-score--neg {
        color: var(--drl-muted);
      }

      .drl-node-score--exact {
        font-weight: 700;
      }

      .drl-node-score--provisional {
        font-style: italic;
      }

      .drl-node-score--cut {
        color: var(--drl-arrow);
        font-weight: 700;
      }

      .drl-node--compact {
        justify-content: center;
        background: var(--drl-bg);
        border-style: dashed;
      }

      .drl-node--compact .drl-node-board {
        display: none;
      }

      .drl-node--compact .drl-node-score {
        color: var(--drl-muted);
        font-style: normal;
        font-weight: 400;
      }

      .drl-status {
        margin-top: var(--drl-space-2);
        color: var(--drl-muted);
        font-size: 0.9em;
        min-height: 1.3em;
      }

      :host(:fullscreen) {
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100%;
        background: var(--drl-bg);
      }

      :host(:fullscreen) drl-controls {
        flex: none;
      }

      :host(:fullscreen) .drl-content {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
      }

      :host(:fullscreen) .drl-toolbar,
      :host(:fullscreen) .drl-breadcrumbs,
      :host(:fullscreen) .drl-bounds,
      :host(:fullscreen) .drl-status {
        flex: none;
      }

      :host(:fullscreen) .drl-canvas {
        flex: 1;
        height: auto;
        min-height: 320px;
      }
    `,
  ];

  @property({ type: Number, attribute: 'max-depth' })
  maxDepth = Number.POSITIVE_INFINITY;

  @property({ type: String, attribute: 'node-size' })
  nodeSize = '3rem';

  @property({ type: Boolean })
  stepper = true;

  @property({ type: Boolean, attribute: 'auto-fit' })
  autoFit = false;

  @property({ type: String, attribute: 'piece-style' })
  pieceStyle: PieceStyle | null = null;

  @state()
  private isFullscreen = false;

  @state()
  private boundsText: BoundsView | null = null;

  readonly model = new TreeModel<S>();

  readonly recorder: TreeRecorder<S> = {
    onVisit: (event: VisitEvent<S>) => {
      this.modelRecorder.onVisit(event);
      this.invalidate();
    },
    onValue: (event: ValueEvent) => {
      this.modelRecorder.onValue(event);
      this.invalidate();
    },
  };

  private readonly modelRecorder = createModelRecorder(this.model);

  private game: Game<S, unknown> | null = null;

  private gameVersion = 0;

  private driver: Stepper<S> | null = null;

  private boundAgent: SearchAgent<S, unknown> | null = null;

  private boundState: S | null = null;

  private primitive: StepPrimitive = 'in';

  private viewRootKey: string | null = null;

  private pendingFrame = 0;

  private hiddenCount = 0;

  private lastItems: PositionedItem<S>[] = [];

  private boxCache: { size: string; width: number } | null = null;

  private transform: ZoomTransform = zoomIdentity;

  private canvas: HTMLDivElement | null = null;

  private viewport: HTMLDivElement | null = null;

  private nodeLayer: HTMLDivElement | null = null;

  private breadcrumbs: HTMLDivElement | null = null;

  private statusText: HTMLElement | null = null;

  private zoomSlider: import('../../ui/drl-slider.js').DrlSlider | null = null;

  private zoomSelection: Selection<HTMLDivElement, undefined, null, undefined> | null = null;

  private zoomBehavior: ZoomBehavior<HTMLDivElement, undefined> | null = null;

  protected override firstUpdated(): void {
    document.addEventListener('fullscreenchange', this.handleFullscreenChange);
    const canvas = this.renderRoot.querySelector<HTMLDivElement>('.drl-canvas');
    const viewport = this.renderRoot.querySelector<HTMLDivElement>('.drl-viewport');
    const nodeLayer = this.renderRoot.querySelector<HTMLDivElement>('.drl-node-layer');
    const breadcrumbs = this.renderRoot.querySelector<HTMLDivElement>('.drl-breadcrumbs');
    const statusText = this.renderRoot.querySelector<HTMLElement>('.drl-status-text');
    const zoomSlider =
      this.renderRoot.querySelector<import('../../ui/drl-slider.js').DrlSlider>('.drl-zoom-slider');
    if (!canvas || !viewport || !nodeLayer || !breadcrumbs || !statusText || !zoomSlider) {
      return;
    }
    this.canvas = canvas;
    this.viewport = viewport;
    this.nodeLayer = nodeLayer;
    this.breadcrumbs = breadcrumbs;
    this.statusText = statusText;
    this.zoomSlider = zoomSlider;
    this.zoomBehavior = d3zoom<HTMLDivElement, undefined>()
      .scaleExtent([ZOOM_MIN, ZOOM_MAX])
      .on('zoom', (event: D3ZoomEvent<HTMLDivElement, undefined>) => {
        this.transform = event.transform;
        this.applyTransform();
        this.syncZoomSlider();
      });
    this.zoomSelection = select<HTMLDivElement, undefined>(canvas);
    this.zoomSelection.call(this.zoomBehavior).on('dblclick.zoom', null);
    canvas.addEventListener('dblclick', this.handleDblClick);
    nodeLayer.addEventListener('click', this.handleNodeClick);
    this.draw();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    document.removeEventListener('fullscreenchange', this.handleFullscreenChange);
    if (this.pendingFrame !== 0) {
      cancelAnimationFrame(this.pendingFrame);
      this.pendingFrame = 0;
    }
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    super.willUpdate(changed);
    if (changed.has('nodeSize')) {
      this.boxCache = null;
    }
    if (changed.has('maxDepth') || changed.has('nodeSize') || changed.has('pieceStyle')) {
      this.invalidate();
    }
  }

  attach<M>(game: Game<S, M>): void {
    this.game = game;
    this.gameVersion += 1;
    this.invalidate();
  }

  startSearch<M>(agent: SearchAgent<S, M>, state: S): void {
    this.model.clear();
    this.viewRootKey = null;
    this.boundAgent = agent;
    this.boundState = state;
    this.driver = createStepper(agent, this.model);
    this.driver.start(state);
    this.running = false;
    this.invalidate();
  }

  clearTree(): void {
    this.model.clear();
    this.viewRootKey = null;
    this.driver = null;
    this.boundAgent = null;
    this.boundState = null;
    this.running = false;
    this.invalidate();
  }

  setViewRoot(key: string | null): void {
    if (key !== null && !this.model.has(key)) {
      return;
    }
    this.viewRootKey = key;
    this.dispatchEvent(
      new CustomEvent<DrlRootChangedDetail>('drl-root-changed', {
        detail: { nodeKey: key },
        bubbles: true,
        composed: true,
      }),
    );
    this.invalidate();
  }

  override step(): void {
    const driver = this.driver;
    if (!driver || driver.isDone()) {
      if (this.running) {
        this.running = false;
      }
      return;
    }
    this.applyPrimitive(driver);
    if (driver.isDone()) {
      this.running = false;
    }
    this.invalidate();
  }

  protected override onReset(): void {
    this.model.clear();
    this.viewRootKey = null;
    if (this.boundAgent && this.boundState !== null) {
      this.driver = createStepper(this.boundAgent, this.model);
      this.driver.start(this.boundState);
    } else {
      this.driver = null;
    }
    this.invalidate();
  }

  private applyPrimitive(driver: Stepper<S>): void {
    if (this.primitive === 'in') {
      driver.stepIn();
    } else if (this.primitive === 'out') {
      driver.stepOut();
    } else {
      driver.stepOver();
    }
  }

  private invalidate(): void {
    if (this.pendingFrame !== 0) {
      return;
    }
    this.pendingFrame = requestAnimationFrame(() => {
      this.pendingFrame = 0;
      this.draw();
    });
  }

  private readonly handleDblClick = (event: MouseEvent): void => {
    if (event.target instanceof Element && event.target.closest('.drl-node')) {
      return;
    }
    this.setViewRoot(this.model.root()?.nodeKey ?? null);
  };

  private readonly handleNodeClick = (event: MouseEvent): void => {
    const target = event.target instanceof Element ? event.target.closest('.drl-node') : null;
    if (!(target instanceof HTMLDivElement)) {
      return;
    }
    const key = target.dataset.key;
    if (key) {
      this.setViewRoot(key);
    }
  };

  private readonly handlePrimitive = (event: MouseEvent): void => {
    const button = event.currentTarget as HTMLButtonElement;
    const kind = button.dataset.primitive;
    if (kind !== 'in' && kind !== 'over' && kind !== 'out') {
      return;
    }
    this.primitive = kind;
    if (!this.running) {
      this.step();
    }
  };

  private readonly handleRootNow = (): void => {
    this.setViewRoot(this.model.root()?.nodeKey ?? null);
  };

  private readonly handleFit = (): void => {
    this.fit();
  };

  private readonly handleAutoFitChange = (event: Event): void => {
    const input = event.target;
    if (!(input instanceof HTMLInputElement)) {
      return;
    }
    this.autoFit = input.checked;
    if (this.autoFit) {
      this.fit();
    }
  };

  private readonly handleFull = (): void => {
    if (this.zoomSelection && this.zoomBehavior) {
      this.zoomSelection.call(this.zoomBehavior.scaleTo, 1);
    }
  };

  private readonly handleFullscreen = (): void => {
    if (document.fullscreenElement === this) {
      void document.exitFullscreen();
      return;
    }
    if (document.fullscreenEnabled) {
      void this.requestFullscreen();
    }
  };

  private readonly handleFullscreenChange = (): void => {
    const fullscreen = document.fullscreenElement === this;
    if (fullscreen === this.isFullscreen) {
      return;
    }
    this.isFullscreen = fullscreen;
    requestAnimationFrame(() => {
      this.fit();
    });
  };

  private readonly handleZoomInput = (event: Event): void => {
    const value = (event as CustomEvent<number>).detail;
    if (this.zoomSelection && this.zoomBehavior) {
      this.zoomSelection.call(this.zoomBehavior.scaleTo, value);
    }
  };

  private readonly handleDepthInput = (event: Event): void => {
    const input = event.target as HTMLInputElement;
    const raw = input.value.trim();
    if (raw === '') {
      this.maxDepth = Number.POSITIVE_INFINITY;
      return;
    }
    const parsed = Number(raw);
    if (Number.isFinite(parsed) && parsed >= 1) {
      this.maxDepth = Math.floor(parsed);
    }
  };

  private applyTransform(): void {
    if (this.viewport) {
      const { x, y, k } = this.transform;
      this.viewport.style.transform = `translate(${x}px, ${y}px) scale(${k})`;
    }
  }

  private syncZoomSlider(): void {
    if (this.zoomSlider) {
      this.zoomSlider.value = Math.round(this.transform.k * 100) / 100;
    }
  }

  private boxWidth(): number {
    const size = this.nodeSize;
    if (this.boxCache && this.boxCache.size === size) {
      return this.boxCache.width;
    }
    const probe = document.createElement('div');
    probe.style.width = size;
    probe.style.position = 'absolute';
    probe.style.visibility = 'hidden';
    probe.style.pointerEvents = 'none';
    this.renderRoot.append(probe);
    const width = probe.getBoundingClientRect().width;
    probe.remove();
    const box = Number.isFinite(width) && width > 16 ? width : DEFAULT_BOX;
    this.boxCache = { size, width: box };
    return box;
  }

  private countDescendants(key: string, memo: Map<string, number>): number {
    const cached = memo.get(key);
    if (cached !== undefined) {
      return cached;
    }
    let total = 0;
    for (const child of this.model.children(key)) {
      total += 1 + this.countDescendants(child.nodeKey, memo);
    }
    memo.set(key, total);
    return total;
  }

  private buildLayout(start: TreeNode<S>): HierarchyNode<LayoutItem<S>> {
    const memo = new Map<string, number>();
    const maxDepth = this.maxDepth;
    let rendered = 0;
    const walk = (node: TreeNode<S>, relDepth: number): LayoutItem<S> => {
      const children = this.model.children(node.nodeKey);
      if (children.length > 0 && (relDepth >= maxDepth || rendered >= MAX_RENDER_NODES)) {
        // Le nœud reste affiché normalement (état + score) ; la taille du
        // sous-arbre masqué migre sur un enfant factice unique.
        rendered += 1;
        return {
          key: node.nodeKey,
          node,
          compact: false,
          collapsedCount: 0,
          children: [
            {
              key: `${node.nodeKey}:collapsed`,
              node,
              compact: true,
              collapsedCount: this.countDescendants(node.nodeKey, memo),
            },
          ],
        };
      }
      rendered += 1;
      if (children.length === 0) {
        return { key: node.nodeKey, node, compact: false, collapsedCount: 0 };
      }
      return {
        key: node.nodeKey,
        node,
        compact: false,
        collapsedCount: 0,
        // Un seul frère élagué suffit : les autres portent la même coupure.
        children: keepFirstPruned(children).map((child) => walk(child, relDepth + 1)),
      };
    };
    return hierarchy(walk(start, 0));
  }

  private itemHalfHeight(item: LayoutItem<S>, box: number): number {
    return item.compact ? COMPACT_HALF_HEIGHT : box / 2 + CARD_EXTRA / 2;
  }

  private fit(): void {
    if (!this.canvas || !this.zoomSelection || !this.zoomBehavior || this.lastItems.length === 0) {
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return;
    }
    const box = this.boxWidth();
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const item of this.lastItems) {
      minX = Math.min(minX, item.x - box / 2);
      maxX = Math.max(maxX, item.x + box / 2);
      minY = Math.min(minY, item.y - item.halfHeight);
      maxY = Math.max(maxY, item.y + item.halfHeight + 8);
    }
    const k = clamp(
      Math.min((rect.width - 24) / (maxX - minX), (rect.height - 24) / (maxY - minY)),
      ZOOM_MIN,
      ZOOM_MAX,
    );
    const tx = rect.width / 2 - (k * (minX + maxX)) / 2;
    const ty = rect.height / 2 - (k * (minY + maxY)) / 2;
    this.zoomSelection.call(this.zoomBehavior.transform, zoomIdentity.translate(tx, ty).scale(k));
  }

  private draw(): void {
    this.updateToolbarState();
    this.updateBreadcrumbs();
    this.updateBounds();
    this.updateTree();
    this.updateStatus();
    if (this.autoFit) {
      this.fit();
    }
  }

  /**
   * Recalcule le panneau α/β. `draw()` tourne dans une frame d'animation :
   * réassigner un objet équivalent relancerait un rendu, d'où la comparaison
   * champ par champ avant affectation.
   */
  private updateBounds(): void {
    const key =
      this.viewRootKey ??
      this.driver?.current() ??
      this.model.lastVisited()?.nodeKey ??
      this.model.root()?.nodeKey ??
      null;
    const summary = boundsSummary(this.model, key);
    if (summary === null) {
      if (this.boundsText !== null) {
        this.boundsText = null;
      }
      return;
    }
    const next: BoundsView = {
      key: summary.nodeKey,
      title: summary.move === null ? 'Racine' : `Coup ${formatMove(summary.move)}`,
      alpha: formatBound(summary.alpha),
      beta: formatBound(summary.beta),
      score: formatScore(summary.score),
      exact: summary.exact,
      cut: summary.cut === null ? null : cutDescription(summary.cut, summary.triggerMove),
    };
    const current = this.boundsText;
    if (
      current !== null &&
      current.key === next.key &&
      current.title === next.title &&
      current.alpha === next.alpha &&
      current.beta === next.beta &&
      current.score === next.score &&
      current.exact === next.exact &&
      current.cut === next.cut
    ) {
      return;
    }
    this.boundsText = next;
  }

  private updateToolbarState(): void {
    const driver = this.driver;
    const active = driver !== null && !driver.isDone() && !this.running;
    this.renderRoot
      .querySelectorAll<HTMLButtonElement>('button[data-primitive]')
      .forEach((button) => {
        button.disabled = !active;
      });
    const rootButton = this.renderRoot.querySelector<HTMLButtonElement>('.drl-root-now');
    if (rootButton) {
      rootButton.disabled = this.model.root() === null;
    }
  }

  private updateBreadcrumbs(): void {
    const container = this.breadcrumbs;
    if (!container) {
      return;
    }
    const chain: Crumb[] = [];
    let cursor = this.viewRootKey ?? this.model.root()?.nodeKey ?? null;
    while (cursor !== null) {
      const node = this.model.node(cursor);
      if (!node) {
        break;
      }
      const label = node.move === null ? 'racine' : formatMove(node.move) || 'racine';
      chain.push({ key: node.nodeKey, label, current: false });
      cursor = node.parentKey;
    }
    chain.reverse();
    const last = chain.at(-1);
    if (last) {
      chain[chain.length - 1] = { ...last, current: true };
    }
    select(container)
      .selectAll<HTMLSpanElement, Crumb>('span')
      .data(chain, (crumb) => crumb.key)
      .join('span')
      .text((crumb) => crumb.label)
      .attr('aria-current', (crumb) => (crumb.current ? 'true' : null))
      .on('click', (_event, crumb) => {
        this.setViewRoot(crumb.key);
      });
  }

  private updateTree(): void {
    const nodeLayer = this.nodeLayer;
    const edges = this.renderRoot.querySelector<SVGSVGElement>('.drl-edges');
    if (!nodeLayer || !edges) {
      return;
    }
    const start = this.viewRootKey ? this.model.node(this.viewRootKey) : this.model.root();
    if (!start) {
      select(nodeLayer).selectAll('*').remove();
      select(edges).selectAll('*').remove();
      this.lastItems = [];
      this.hiddenCount = 0;
      return;
    }
    const box = this.boxWidth();
    const positioned = tree<LayoutItem<S>>()
      .nodeSize([box + NODE_GAP_X, box + NODE_GAP_Y])
      .separation(() => 1)(this.buildLayout(start));
    const nodes = positioned.descendants();
    let minX = Number.POSITIVE_INFINITY;
    for (const node of nodes) {
      minX = Math.min(minX, node.x - box / 2);
    }
    const shiftX = LAYOUT_MARGIN - minX;
    const positions = new Map<LayoutItem<S>, { x: number; y: number; half: number }>();
    this.lastItems = nodes.map((node) => {
      const item: PositionedItem<S> = {
        item: node.data,
        x: node.x + shiftX,
        y: node.y + LAYOUT_MARGIN,
        halfHeight: this.itemHalfHeight(node.data, box),
      };
      positions.set(node.data, { x: item.x, y: item.y, half: item.halfHeight });
      return item;
    });
    const rendered = new Set(this.lastItems.map((entry) => entry.item.key));
    const rootNode = this.model.root();
    const rootPlayer =
      this.game !== null && rootNode !== null && rootNode.state !== undefined
        ? this.game.currentPlayer(rootNode.state)
        : null;
    const bestEdges = new Map<string, string>();
    for (const entry of this.lastItems) {
      if (entry.item.compact) {
        continue;
      }
      // Nœud maximisant pour le joueur racine, minimisant pour l'adversaire :
      // le meilleur enfant est celui au score le plus bas dans le second cas.
      const maximizing =
        rootPlayer === null ||
        this.game === null ||
        entry.item.node.state === undefined ||
        this.game.currentPlayer(entry.item.node.state) === rootPlayer;
      let bestKey: string | null = null;
      let bestScore = maximizing ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
      for (const child of this.model.children(entry.item.node.nodeKey)) {
        if (child.score === null || !rendered.has(child.nodeKey)) {
          continue;
        }
        const improves = maximizing ? child.score > bestScore : child.score < bestScore;
        if (improves) {
          bestScore = child.score;
          bestKey = child.nodeKey;
        }
      }
      if (bestKey) {
        bestEdges.set(entry.item.key, bestKey);
      }
    }
    // Arête déclencheuse de coupure : parent → fils qui a franchi la borne.
    const cutEdges = new Map<string, string>();
    for (const entry of this.lastItems) {
      if (entry.item.compact) {
        continue;
      }
      const triggerKey = entry.item.node.cut?.triggerKey ?? null;
      if (triggerKey !== null && rendered.has(triggerKey)) {
        cutEdges.set(entry.item.key, triggerKey);
      }
    }
    const edgeData: EdgeDatum[] = [];
    for (const link of positioned.links()) {
      const source = positions.get(link.source.data);
      const target = positions.get(link.target.data);
      if (!source || !target) {
        continue;
      }
      edgeData.push({
        key: link.target.data.key,
        sx: source.x,
        sy: source.y + source.half + 1,
        tx: target.x,
        ty: target.y - target.half - 1,
        best: bestEdges.get(link.source.data.key) === link.target.data.key,
        cut: cutEdges.get(link.source.data.key) === link.target.data.key,
        // L'arête vers l'enfant factice porte le coup du parent : on la
        // laisse sans étiquette pour éviter la redite.
        label: link.target.data.compact ? '' : formatMove(link.target.data.node.move),
      });
    }
    this.hiddenCount = this.lastItems.reduce(
      (sum, entry) => sum + (entry.item.compact ? entry.item.collapsedCount : 0),
      0,
    );
    const edgeSelection = select(edges);
    edgeSelection
      .selectAll<SVGPathElement, EdgeDatum>('path')
      .data(edgeData, (edge) => edge.key)
      .join('path')
      .attr(
        'class',
        (edge) =>
          `drl-edge${edge.best ? ' drl-edge--best' : ''}${edge.cut ? ' drl-edge--cut' : ''}`,
      )
      .attr('d', (edge) => {
        const mid = (edge.sy + edge.ty) / 2;
        return `M ${edge.sx} ${edge.sy} C ${edge.sx} ${mid} ${edge.tx} ${mid} ${edge.tx} ${edge.ty}`;
      });
    edgeSelection
      .selectAll<SVGTextElement, EdgeDatum>('text')
      .data(
        edgeData.filter((edge) => edge.label !== ''),
        (edge) => edge.key,
      )
      .join('text')
      .attr('class', 'drl-edge-label')
      .attr('x', (edge) => (edge.sx + edge.tx) / 2 + 4)
      .attr('y', (edge) => (edge.sy + edge.ty) / 2 + 3)
      .text((edge) => edge.label);
    select(nodeLayer)
      .selectAll<HTMLDivElement, PositionedItem<S>>('div.drl-node')
      .data(this.lastItems, (entry) => entry.item.key)
      .join(
        (enter) =>
          enter
            .append('div')
            .attr('class', 'drl-node')
            .attr('data-key', (entry) => entry.item.key)
            .each(function () {
              const board = document.createElement('div');
              board.className = 'drl-node-board';
              const score = document.createElement('div');
              score.className = 'drl-node-score';
              const badge = document.createElement('span');
              badge.className = 'drl-node-badge';
              badge.setAttribute('aria-hidden', 'true');
              badge.textContent = '↪';
              this.append(board, score, badge);
            }),
        (update) => update,
        (exit) => exit.remove(),
      )
      .each((datum, index, group) => {
        const card = group[index];
        this.syncCard(card, datum, box);
      });
  }

  private syncCard(card: HTMLDivElement, entry: PositionedItem<S>, box: number): void {
    const node = entry.item.node;
    const compact = entry.item.compact;
    card.style.width = `${box}px`;
    card.style.left = `${entry.x - box / 2}px`;
    card.style.top = `${entry.y - entry.halfHeight}px`;
    card.classList.toggle('drl-node--compact', compact);
    card.classList.toggle('drl-node--terminal', node.kind === 'terminal');
    card.classList.toggle('drl-node--pruned', node.kind === 'pruned');
    card.classList.toggle('drl-node--tt-hit', node.kind === 'tt-hit');
    card.classList.toggle('drl-node--cursor', this.driver?.current() === node.nodeKey);
    card.classList.toggle(
      'drl-node--viewroot',
      (this.viewRootKey ?? this.model.root()?.nodeKey) === node.nodeKey,
    );
    const boardWrap = card.children[0] as HTMLDivElement;
    const score = card.children[1] as HTMLDivElement;
    // Badge de raccourci TT : la flèche seule, ou flèche + profondeur certifiée
    // de l'entrée consommée (ex. « ↪3 », « ↪∞ »).
    const badge = card.children[2] as HTMLSpanElement | undefined;
    if (badge !== undefined) {
      badge.textContent =
        node.kind === 'tt-hit' && node.entryDepth !== undefined
          ? `↪${formatEntryDepth(node.entryDepth)}`
          : '↪';
    }
    const cut = node.cut ?? null;
    const pruned = cut !== null && node.kind === 'pruned';
    card.classList.toggle('drl-node--cut', pruned);
    if (compact) {
      score.textContent = `+${entry.item.collapsedCount}`;
      score.className = 'drl-node-score';
      card.title = `${entry.item.collapsedCount} descendants masqués (profondeur maximale atteinte)`;
      return;
    }
    card.title = this.nodeTitle(node);
    score.className = pruned
      ? 'drl-node-score drl-node-score--cut'
      : scoreClass(node.score, node.exact);
    score.textContent = pruned && cut !== null ? cutLabel(cut) : formatScore(node.score);
    const game = this.game;
    if (!game || node.state === undefined) {
      return;
    }
    const signature = `${this.gameVersion}:${game.stateKey(node.state)}`;
    if (card.dataset.sig === signature) {
      return;
    }
    card.dataset.sig = signature;
    let boardEl =
      boardWrap.firstElementChild instanceof DrlGameBoard ? boardWrap.firstElementChild : null;
    if (!boardEl) {
      boardWrap.replaceChildren();
      boardEl = new DrlGameBoard();
      boardWrap.append(boardEl);
    }
    const view = game.boardView(node.state);
    boardEl.cellSize = BOARD_CELL_SIZE;
    boardEl.pieceStyle = this.pieceStyle ?? autoPieceStyle(view);
    boardEl.view = view;
  }

  /** Infobulle d'un nœud : score, fenêtre α/β et coupure éventuelle. */
  private nodeTitle(node: TreeNode<S>): string {
    const parts: string[] = [];
    parts.push(
      node.score === null
        ? 'Score inconnu'
        : `Score ${formatScore(node.score)}${node.exact ? ' (exact)' : ' (provisoire)'}`,
    );
    const alpha = node.window?.alpha ?? Number.NEGATIVE_INFINITY;
    const beta = node.window?.beta ?? Number.POSITIVE_INFINITY;
    parts.push(`α ${formatBound(alpha)} · β ${formatBound(beta)}`);
    const cut = node.cut ?? null;
    if (cut !== null) {
      const triggerMove =
        cut.triggerKey === null ? null : (this.model.node(cut.triggerKey)?.move ?? null);
      parts.push(cutDescription(cut, triggerMove));
    }
    if (node.kind === 'tt-hit' && node.entryDepth !== undefined) {
      parts.push(`Profondeur table : ${formatEntryDepth(node.entryDepth)}`);
    }
    return parts.join(' · ');
  }

  private updateStatus(): void {
    const statusText = this.statusText;
    if (!statusText) {
      return;
    }
    const driver = this.driver;
    const parts = [`Nœuds : ${this.model.size()}`, `Masqués : ${this.hiddenCount}`];
    if (driver) {
      parts.push(`Curseur : ${driver.current() ?? '–'}`);
      parts.push(driver.isDone() ? 'Recherche terminée' : 'Recherche en cours');
    } else {
      parts.push('Aucune recherche liée');
    }
    statusText.textContent = parts.join(' · ');
  }

  protected override renderContent(): TemplateResult {
    const depthValue = this.maxDepth === Number.POSITIVE_INFINITY ? '' : String(this.maxDepth);
    return html`
      <div class="drl-toolbar" role="toolbar" aria-label="Débogueur d'arbre de recherche">
        ${
          this.stepper
            ? html`
              <button
                type="button"
                class="drl-button"
                data-primitive="in"
                title="Pas à pas entrant : descendre d'un nœud visité"
                @click=${this.handlePrimitive}
              >
                Entrer
              </button>
              <button
                type="button"
                class="drl-button"
                data-primitive="over"
                title="Pas à pas principal : terminer le sous-arbre courant"
                @click=${this.handlePrimitive}
              >
                Passer
              </button>
              <button
                type="button"
                class="drl-button"
                data-primitive="out"
                title="Pas à pas sortant : remonter au parent"
                @click=${this.handlePrimitive}
              >
                Sortir
              </button>
            `
            : nothing
        }
        <button
          type="button"
          class="drl-button drl-root-now"
          title="Revenir à la racine de la recherche"
          @click=${this.handleRootNow}
        >
          Racine = état courant
        </button>
        <drl-slider
          class="drl-zoom-slider"
          label="Zoom"
          min="${ZOOM_MIN}"
          max="${ZOOM_MAX}"
          step="0.05"
          value="1"
          @drl-input=${this.handleZoomInput}
        ></drl-slider>
        <button
          type="button"
          class="drl-button"
          title="Ajuster le zoom à l'arbre affiché"
          @click=${this.handleFit}
        >
          Ajuster
        </button>
        <label class="drl-autofit" title="Recentrer et ajuster le zoom à chaque mise à jour de l'arbre ou déplacement du point de vue">
          <input
            type="checkbox"
            .checked=${this.autoFit}
            @change=${this.handleAutoFitChange}
          />
          <span>Ajuster auto</span>
        </label>
        <button type="button" class="drl-button" title="Zoom 100 %" @click=${this.handleFull}>
          100 %
        </button>
        <button
          type="button"
          class="drl-button"
          title=${
            this.isFullscreen
              ? "Quitter le mode plein écran de l'arbre"
              : "Afficher l'arbre et ses commandes en plein écran"
          }
          @click=${this.handleFullscreen}
        >
          ${this.isFullscreen ? 'Quitter le plein écran' : 'Plein écran'}
        </button>
        <label class="drl-depth">
          <span>Prof. max</span>
          <input
            class="drl-depth-input"
            type="number"
            min="1"
            step="1"
            placeholder="∞"
            .value=${depthValue}
            @input=${this.handleDepthInput}
          />
        </label>
      </div>
      <div class="drl-breadcrumbs" aria-label="Fil d'Ariane"></div>
      <div class="drl-bounds" aria-live="polite">
        ${
          this.boundsText === null
            ? nothing
            : html`
                <span class="drl-bounds-item drl-bounds-title">${this.boundsText.title}</span>
                <span class="drl-bounds-item">α ${this.boundsText.alpha}</span>
                <span class="drl-bounds-item">β ${this.boundsText.beta}</span>
                <span class="drl-bounds-item">
                  Score ${this.boundsText.score === '' ? '—' : this.boundsText.score}
                  ${this.boundsText.exact ? '(exact)' : '(provisoire)'}
                </span>
                ${
                  this.boundsText.cut === null
                    ? html`<span class="drl-bounds-item">${NO_CUT_LABEL}</span>`
                    : html`<span class="drl-bounds-item drl-bounds-cut">${this.boundsText.cut}</span>`
                }
              `
        }
      </div>
      <div class="drl-canvas">
        <div class="drl-viewport">
          <svg class="drl-edges" width="1" height="1"></svg>
          <div class="drl-node-layer"></div>
        </div>
      </div>
      <div class="drl-status"><span class="drl-status-text"></span></div>
    `;
  }
}
