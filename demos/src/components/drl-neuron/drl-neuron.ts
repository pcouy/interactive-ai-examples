import type { BaseType, ScaleLinear, Selection } from 'd3';
import { scaleLinear, select } from 'd3';
import type { PropertyValues, TemplateResult } from 'lit';
import { css, html } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { DemoElement } from '../../base/demo-element.js';
import type {
  LossKind,
  NeuronEvaluation,
  NeuronExample,
  NeuronPhase,
  WeightUpdate,
} from '../../core/neuron.js';
import { NeuronCore } from '../../core/neuron.js';
import { drlTokens } from '../../styles/tokens.js';
import {
  createLandscapeScene,
  type LandscapeScene,
  sanitizeHeightScale,
} from '../shared/landscape-scene.js';

/**
 * One 2-input neuron trained by mini-batch gradient descent.
 *
 * The left view is a 2D SVG scatter of the examples over the input square
 * [-d, d]², where d is the configurable input domain. A 16×16 grid painted
 * behind the points shows the neuron's current prediction ŷ = w₁·x₁ + w₂·x₂ + b
 * at each cell centre. The right view is the 3D loss landscape over the weight
 * plane (w₁, w₂); a ball descends the valley batch after batch. The vertical
 * exaggeration of that landscape is configurable.
 *
 * Color scale: targets span [t min, t max]. A low target is blue
 * (--drl-accent), a high target is violet (--drl-accent-2). Predictions use the
 * same range, so a point and the cell below it are directly comparable: a cell
 * that matches its point shows the same color.
 */

const PLANE_X = 40;
const PLANE_Y = 10;
const PLANE_SIZE = 340;
const PLANE_BOTTOM = PLANE_Y + PLANE_SIZE;
const GRID_RES = 16;
const CELL_SIZE = PLANE_SIZE / GRID_RES;

const CANVAS_WIDTH = 1920;
const CANVAS_HEIGHT = 1080;

const MIN_BATCH = 1;
const MAX_BATCH = 12;

/** Minimum delay between two bias-only height-field rebuilds during play. */
const FIELD_REBUILD_INTERVAL_MS = 200;
/** Minimum bias delta that triggers a rebuild outside the time window. */
const FIELD_B_STEP = 0.1;

const COLOR_LOW_FALLBACK = '#2563eb';
const COLOR_HIGH_FALLBACK = '#7c3aed';

const PHASE_LABELS: Record<NeuronPhase, string> = {
  idle: 'Prêt',
  present: 'Exemple',
  gradient: 'Gradient',
  fade: 'Archivage',
  mean: 'Moyenne',
  move: 'Descente',
};

interface GridCell {
  i: number;
  j: number;
}

type SingleSelection<ElementType extends BaseType> = Selection<
  ElementType,
  unknown,
  null,
  undefined
>;

type GridSelection = Selection<SVGRectElement, GridCell, SVGGElement, unknown>;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formatNumber(value: number): string {
  return value.toFixed(2).replace('.', ',');
}

/**
 * Maps an input coordinate to the SVG plane. `domain` must be > 0; the core
 * clamps the input domain to at least 0.1.
 */
export function dataX(x: number, domain: number): number {
  return PLANE_X + ((x + domain) / (2 * domain)) * PLANE_SIZE;
}

/**
 * Maps an input coordinate to the SVG plane. `domain` must be > 0; the core
 * clamps the input domain to at least 0.1.
 */
export function dataY(y: number, domain: number): number {
  return PLANE_BOTTOM - ((y + domain) / (2 * domain)) * PLANE_SIZE;
}

/**
 * Returns the input coordinate at the centre of grid cell `index` along one
 * axis. The grid covers [-domain, domain] with GRID_RES cells.
 */
export function gridCenter(index: number, domain: number): number {
  return -domain + ((2 * index + 1) / GRID_RES) * domain;
}

@customElement('drl-neuron')
export class Neuron extends DemoElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: block;
      }

      .drl-status {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-2);
        min-height: 1.5rem;
      }

      .drl-badge {
        background: var(--drl-accent-2);
        color: var(--drl-surface);
        border-radius: calc(var(--drl-radius) / 2);
        padding: 0 var(--drl-space-2);
        font-size: 0.85em;
      }

      .drl-status-text {
        color: var(--drl-muted);
      }

      .drl-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-2);
      }

      .drl-stat {
        background: var(--drl-bg);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        padding: var(--drl-space-2);
      }

      .drl-stat-label {
        margin: 0 0 2px;
        color: var(--drl-muted);
        font-size: 0.8em;
      }

      .drl-stat-value {
        margin: 0;
        font-variant-numeric: tabular-nums;
      }

      .drl-views {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: var(--drl-space-3);
        align-items: start;
      }

      @media (max-width: 980px) {
        .drl-views {
          grid-template-columns: minmax(0, 1fr);
        }
      }

      svg {
        display: block;
        width: 100%;
        height: auto;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
      }

      canvas {
        display: block;
        width: 100%;
        max-width: ${CANVAS_WIDTH}px;
        height: auto;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
        cursor: grab;
        touch-action: none;
      }

      canvas:active {
        cursor: grabbing;
      }

      .drl-webgl-error {
        color: var(--drl-muted);
      }

      .drl-axis {
        fill: var(--drl-muted);
        font-family: var(--drl-font);
        font-size: 13px;
      }

      /* CSS custom properties do not resolve inside the stop-color attribute. */
      .drl-legend-low {
        stop-color: var(--drl-accent);
      }

      .drl-legend-high {
        stop-color: var(--drl-accent-2);
      }
    `,
  ];

  @property({ type: Number, attribute: 'learning-rate' })
  learningRate = 0.15;

  @property({ type: String, attribute: 'expression' })
  expression = NeuronCore.DEFAULT_EXPRESSION;

  @property({ type: String, attribute: 'loss-kind' })
  lossKind: LossKind = 'l2';

  @property({ type: Boolean, attribute: 'bias' })
  bias = false;

  @property({ type: Number, attribute: 'count' })
  count = 40;

  @property({ type: Number, attribute: 'input-domain' })
  inputDomain = 1;

  @property({ type: Number, attribute: 'batch-size' })
  batchSize = 4;

  @property({ type: Number, attribute: 'height-scale' })
  heightScale = 1.5;

  @state()
  private webglError = false;

  @state()
  private phase: NeuronPhase = 'idle';

  @state()
  private epoch = 0;

  @state()
  private w1 = 0;

  @state()
  private w2 = 0;

  @state()
  private b = 0;

  @state()
  private lossValue = 0;

  @state()
  private gradMag = 0;

  private diverged = false;

  @state()
  private currentExample: NeuronExample | null = null;

  @state()
  private exampleNumber = 0;

  private readonly core = new NeuronCore();

  private scene: LandscapeScene | null = null;

  private setupPromise: Promise<void> | null = null;

  private tweening = false;

  private animFrame: number | null = null;

  private prevFrame = 0;

  private gridGroup: SingleSelection<SVGGElement> | null = null;

  private gridCells: GridSelection | null = null;

  private pointsGroup: SingleSelection<SVGGElement> | null = null;

  private halo: SingleSelection<SVGCircleElement> | null = null;

  private fieldDomain = Number.NaN;

  private fieldDataset: readonly NeuronExample[] | null = null;

  private fieldLoss: LossKind | null = null;

  private fieldBias: boolean | null = null;

  private fieldB = Number.NaN;

  private fieldRebuiltAt = 0;

  private colorScale: ScaleLinear<string, string> | null = null;

  private colorDataset: readonly NeuronExample[] | null = null;

  /** Last weights broadcast to listeners, so drl-weights fires only on change. */
  private lastWeights: { w1: number; w2: number; b: number } | null = null;

  protected override async firstUpdated(): Promise<void> {
    const svgElement = this.renderRoot.querySelector<SVGSVGElement>('svg');
    if (svgElement) {
      const root = select(svgElement);
      this.gridGroup = root.select<SVGGElement>('.drl-grid');
      this.pointsGroup = root.select<SVGGElement>('.drl-points');
      this.halo = root.select<SVGCircleElement>('.drl-halo');
      const gridGroup = this.gridGroup;
      const cells: GridCell[] = [];
      for (let j = 0; j < GRID_RES; j += 1) {
        for (let i = 0; i < GRID_RES; i += 1) {
          cells.push({ i, j });
        }
      }
      this.gridCells = gridGroup
        .selectAll<SVGRectElement, GridCell>('rect')
        .data(cells, (cell) => `${cell.i}-${cell.j}`)
        .join('rect')
        .attr('x', (cell) => PLANE_X + cell.i * CELL_SIZE)
        .attr('y', (cell) => PLANE_Y + cell.j * CELL_SIZE)
        .attr('width', CELL_SIZE)
        .attr('height', CELL_SIZE);
    }

    this.core.setDataset({
      expression: this.expression,
      count: Math.max(1, Math.round(this.count)),
      inputDomain: this.inputDomain,
    });
    this.core.setLoss(this.lossKind);
    this.core.setBiasEnabled(this.bias);
    this.core.setBatchSize(clamp(Math.round(this.batchSize), MIN_BATCH, MAX_BATCH));
    this.dispatchTargetError();
    this.sync();
    this.draw();

    await this.setupScene();
  }

  /**
   * Creates the 3D scene on the current canvas. Called once from firstUpdated
   * and again on reconnect, because Lit runs firstUpdated only once.
   *
   * The guard returns the in-flight promise so a reconnect during an async
   * scene creation does not start a second WebGL context.
   */
  private setupScene(): Promise<void> {
    const pending = this.setupPromise;
    if (pending !== null) {
      return pending;
    }
    this.setupPromise = this.createScene().finally(() => {
      this.setupPromise = null;
    });
    return this.setupPromise;
  }

  private async createScene(): Promise<void> {
    const canvas = this.renderRoot.querySelector('canvas');
    if (!(canvas instanceof HTMLCanvasElement)) {
      return;
    }
    try {
      const scene = await createLandscapeScene(canvas);
      if (!this.isConnected) {
        scene.dispose();
        return;
      }
      this.scene = scene;
      this.rebuildHeightField();
      scene.setHeightScale(sanitizeHeightScale(this.heightScale));
      scene.moveBall(this.core.w1, this.core.w2);
      scene.render();
    } catch {
      this.webglError = true;
    }
  }

  override step(): void {
    const scene = this.scene;
    if (this.running) {
      const move = this.core.advanceBatch(this.learningRate);
      if (move !== null && scene !== null) {
        if (this.core.diverged) {
          // Cancel any in-flight tween: the snap-back must not be overwritten by
          // a tween that still targets the diverged weights.
          scene.clearTween();
          this.tweening = false;
          scene.moveBall(this.core.w1, this.core.w2);
          scene.pushTrail(this.core.w1, this.core.w2);
        } else {
          // Push the trail at move time: at high speeds the tween is reset
          // before its 600 ms elapse, so onDone would never fire.
          this.tweening = true;
          scene.pushTrail(move.toA, move.toB);
          scene.tweenBall({ w1: move.toA, w2: move.toB }, () => {
            this.tweening = false;
          });
        }
      }
    } else {
      this.core.advance(this.learningRate, { detailed: true });
      scene?.moveBall(this.core.w1, this.core.w2);
    }
    this.rebuildHeightField(!this.running);
    this.sync();
    this.draw();
  }

  regenerateDataset(): void {
    // Both callers (the Nouveau jeu de données button and the count slider via
    // willUpdate) pass a fresh seed; only an expression change keeps the
    // current dataset seed.
    this.tweening = false;
    this.scene?.clearTween();
    this.core.setDataset({
      expression: this.expression,
      count: Math.max(1, Math.round(this.count)),
      inputDomain: this.inputDomain,
      seed: (Math.random() * 0xffffffff) >>> 0,
    });
    this.dispatchTargetError();
    this.scene?.clearTrail();
    this.rebuildHeightField(true);
    this.scene?.moveBall(this.core.w1, this.core.w2);
    this.sync();
    this.draw();
  }

  protected override onReset(): void {
    this.tweening = false;
    this.scene?.clearTween();
    this.core.reset();
    this.scene?.clearTrail();
    this.rebuildHeightField(true);
    this.scene?.moveBall(this.core.w1, this.core.w2);
    this.sync();
    this.draw();
  }

  /**
   * Places the neuron weights by hand and restarts training from there. Stops
   * play and any in-flight tween so the manual placement is not overwritten.
   */
  setWeights(weights: WeightUpdate): void {
    this.running = false;
    this.tweening = false;
    this.scene?.clearTween();
    this.scene?.clearTrail();
    this.core.setWeights(weights);
    this.rebuildHeightField(true);
    this.scene?.moveBall(this.core.w1, this.core.w2);
    this.sync();
    this.draw();
  }

  /**
   * Evaluates the neuron on one input point. Delegates to the core, so the
   * result follows the current expression, weights, bias and loss kind. The
   * component emits no event for this: the page re-calls it when it needs a
   * fresh value (on drl-weights, control changes and probe edits).
   */
  evaluate(x1: number, x2: number): NeuronEvaluation {
    return this.core.evaluate(x1, x2);
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    super.willUpdate(changed);
    if (changed.has('running') && !this.running) {
      this.rebuildHeightField(true);
    }
    if (!this.hasUpdated) {
      return;
    }
    // The vertical scale is a pure presentation change: it re-normalizes the
    // cached field samples, so it must not set `reconfigured` and trigger a
    // full height-field rebuild.
    if (changed.has('heightScale')) {
      this.scene?.setHeightScale(sanitizeHeightScale(this.heightScale));
    }
    let reconfigured = false;
    if (changed.has('expression') || changed.has('inputDomain')) {
      this.core.setDataset({
        expression: this.expression,
        count: Math.max(1, Math.round(this.count)),
        inputDomain: this.inputDomain,
      });
      this.dispatchTargetError();
      reconfigured = true;
    }
    if (changed.has('lossKind')) {
      this.core.setLoss(this.lossKind);
      this.core.reset();
      reconfigured = true;
    }
    if (changed.has('bias')) {
      this.core.setBiasEnabled(this.bias);
      this.core.reset();
      reconfigured = true;
    }
    if (changed.has('batchSize')) {
      this.core.setBatchSize(clamp(Math.round(this.batchSize), MIN_BATCH, MAX_BATCH));
      this.core.reset();
      reconfigured = true;
    }
    // regenerateDataset already rebuilds the height field, clears the trail,
    // syncs and draws, so the shared sequence below must not run for it.
    if (changed.has('count')) {
      this.regenerateDataset();
      return;
    }
    if (!reconfigured) {
      return;
    }
    this.rebuildHeightField(true);
    this.scene?.clearTrail();
    this.sync();
    this.scene?.moveBall(this.core.w1, this.core.w2);
    this.draw();
  }

  private sync(): void {
    const core = this.core;
    this.phase = core.phase;
    this.epoch = core.epoch;
    this.w1 = core.w1;
    this.w2 = core.w2;
    this.b = core.b;
    this.lossValue = core.loss;
    this.gradMag = core.gradientMagnitude;
    this.currentExample = core.currentExample;
    this.exampleNumber = core.exampleIndex + 1;
    const last = this.lastWeights;
    if (last === null || last.w1 !== this.w1 || last.w2 !== this.w2 || last.b !== this.b) {
      this.lastWeights = { w1: this.w1, w2: this.w2, b: this.b };
      this.dispatchEvent(
        new CustomEvent<{ w1: number; w2: number; b: number }>('drl-weights', {
          detail: { w1: this.w1, w2: this.w2, b: this.b },
          bubbles: true,
          composed: true,
        }),
      );
    }
    const diverged = core.diverged;
    if (diverged !== this.diverged) {
      this.diverged = diverged;
      this.dispatchEvent(
        new CustomEvent<boolean>('drl-diverged', {
          detail: diverged,
          bubbles: true,
          composed: true,
        }),
      );
    }
    this.requestRender();
  }

  private draw(): void {
    const gridCells = this.gridCells;
    const pointsGroup = this.pointsGroup;
    const halo = this.halo;
    if (!gridCells || !pointsGroup || !halo) {
      return;
    }

    const w1 = this.w1;
    const w2 = this.w2;
    const bias = this.b;
    // The input square spans [-d, d] on both axes.
    const d = this.core.inputDomain;

    gridCells.attr('fill', (cell) => {
      const cx = gridCenter(cell.i, d);
      const cy = gridCenter(GRID_RES - 1 - cell.j, d);
      return this.targetColor(w1 * cx + w2 * cy + bias);
    });

    const dataset = this.core.dataset;
    pointsGroup
      .selectAll<SVGCircleElement, number>('circle')
      .data(
        dataset.map((_, index) => index),
        (index) => String(index),
      )
      .join('circle')
      .attr('r', 7)
      .attr('cx', (index) => dataX(dataset[index].x1, d))
      .attr('cy', (index) => dataY(dataset[index].x2, d))
      .attr('fill', (index) => this.targetColor(dataset[index].t))
      .attr('stroke', 'none');

    const example = this.currentExample;
    const showHalo =
      example !== null &&
      (this.phase === 'present' || this.phase === 'gradient' || this.phase === 'fade');
    if (example && showHalo) {
      halo
        .attr('cx', dataX(example.x1, d))
        .attr('cy', dataY(example.x2, d))
        .attr('visibility', 'visible');
    } else {
      halo.attr('visibility', 'hidden');
    }
  }

  /**
   * Maps a target (or a prediction) to a color. The scale covers the target
   * range [t min, t max]; out-of-range predictions clamp to the end colors.
   */
  private targetColor(value: number): string {
    if (this.colorScale === null || this.core.dataset !== this.colorDataset) {
      this.refreshColorScale();
    }
    const scale = this.colorScale;
    return scale === null ? COLOR_LOW_FALLBACK : scale(value);
  }

  private refreshColorScale(): void {
    const styles = getComputedStyle(this);
    const low = styles.getPropertyValue('--drl-accent').trim() || COLOR_LOW_FALLBACK;
    const high = styles.getPropertyValue('--drl-accent-2').trim() || COLOR_HIGH_FALLBACK;
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const example of this.core.dataset) {
      min = Math.min(min, example.t);
      max = Math.max(max, example.t);
    }
    if (!Number.isFinite(min) || !Number.isFinite(max) || max - min <= 1e-6) {
      const center = Number.isFinite(min) ? min : 0;
      min = center - 0.5;
      max = center + 0.5;
    }
    this.colorScale = scaleLinear<string, string>()
      .domain([min, max])
      .range([low, high])
      .clamp(true);
    this.colorDataset = this.core.dataset;
  }

  private rebuildHeightField(force = false): void {
    const scene = this.scene;
    if (!scene) {
      return;
    }
    const { domain, dataset, lossKind, biasEnabled, b } = this.core;
    const structural =
      domain !== this.fieldDomain ||
      dataset !== this.fieldDataset ||
      lossKind !== this.fieldLoss ||
      biasEnabled !== this.fieldBias;
    const bChanged = b !== this.fieldB;
    if (!force && !structural && !bChanged) {
      return;
    }
    // During play the bias moves on every batch. Rebuilding the whole surface
    // each frame is wasteful, so a bias-only change is throttled. The surface
    // may then lag the bias by up to FIELD_REBUILD_INTERVAL_MS; a forced rebuild
    // (pause, reset, reconfigure) restores exactness.
    if (!force && !structural && bChanged) {
      const stepped = Math.abs(b - this.fieldB) >= FIELD_B_STEP;
      const due = performance.now() - this.fieldRebuiltAt >= FIELD_REBUILD_INTERVAL_MS;
      if (!stepped && !due) {
        return;
      }
    }
    this.fieldDomain = domain;
    this.fieldDataset = dataset;
    this.fieldLoss = lossKind;
    this.fieldBias = biasEnabled;
    this.fieldB = b;
    this.fieldRebuiltAt = performance.now();
    scene.setHeightField((w1, w2) => this.core.landscapeLoss(w1, w2), domain);
  }

  private dispatchTargetError(): void {
    const error = this.core.targetError;
    const detail =
      error === null
        ? null
        : `Expression invalide : « ${this.expression} » — « x1 + x2 » utilisée par défaut.`;
    this.dispatchEvent(
      new CustomEvent<string | null>('drl-target-error', {
        detail,
        bubbles: true,
        composed: true,
      }),
    );
  }

  private readonly requestRender = (): void => {
    if (this.animFrame !== null) {
      return;
    }
    this.animFrame = requestAnimationFrame(this.onAnimFrame);
  };

  private readonly onAnimFrame = (time: number): void => {
    this.animFrame = null;
    const delta = this.prevFrame === 0 ? 0.016 : Math.min(0.1, (time - this.prevFrame) / 1000);
    this.prevFrame = time;
    const scene = this.scene;
    scene?.update(delta);
    scene?.render();
    if (this.tweening) {
      this.requestRender();
    }
  };

  private statusLine(): string {
    switch (this.phase) {
      case 'idle':
        return 'Prêt — lancez la descente ou avancez pas à pas';
      case 'present': {
        const example = this.currentExample;
        if (!example) {
          return '';
        }
        return (
          `Exemple ${this.exampleNumber}/${this.core.batchSize} : ` +
          `(x₁, x₂) = (${formatNumber(example.x1)} ; ${formatNumber(example.x2)}) ` +
          `→ t = ${formatNumber(example.t)}`
        );
      }
      case 'gradient':
        return 'Calcul du gradient de cet exemple';
      case 'fade':
        return 'Exemple archivé : son gradient entre dans la moyenne de la batch';
      case 'mean':
        return 'Moyenne des gradients de la batch';
      case 'move':
        return 'Mise à jour des poids : w ← w − learning rate × gradient moyen';
    }
  }

  /** Axis caption with the current input range, e.g. `x₁ (−1,00 … 1,00)`. */
  private axisLabel(axis: 'x₁' | 'x₂'): string {
    const d = this.core.inputDomain;
    return `${axis} (−${formatNumber(d)} … ${formatNumber(d)})`;
  }

  protected override renderHeader(): TemplateResult {
    return html`
      <div class="drl-status">
        <span class="drl-badge">${PHASE_LABELS[this.phase]}</span>
        <span class="drl-status-text">${this.statusLine()}</span>
      </div>
      <div class="drl-stats">
        <div class="drl-stat">
          <p class="drl-stat-label">w₁</p>
          <p class="drl-stat-value">${formatNumber(this.w1)}</p>
        </div>
        <div class="drl-stat">
          <p class="drl-stat-label">w₂</p>
          <p class="drl-stat-value">${formatNumber(this.w2)}</p>
        </div>
        <div class="drl-stat">
          <p class="drl-stat-label">b</p>
          <p class="drl-stat-value">${formatNumber(this.b)}</p>
        </div>
        <div class="drl-stat">
          <p class="drl-stat-label">Loss (moyenne)</p>
          <p class="drl-stat-value">${formatNumber(this.lossValue)}</p>
        </div>
        <div class="drl-stat">
          <p class="drl-stat-label">|gradient|</p>
          <p class="drl-stat-value">${formatNumber(this.gradMag)}</p>
        </div>
        <div class="drl-stat">
          <p class="drl-stat-label">Itérations</p>
          <p class="drl-stat-value">${this.epoch}</p>
        </div>
      </div>
    `;
  }

  protected override renderContent(): TemplateResult {
    return html`
      <div class="drl-views">
        <svg
          viewBox="0 0 640 360"
          role="img"
          aria-label="Nuage d'exemples et grille de prédictions du neurone"
        >
          <defs>
            <clipPath id="drl-neuron-plane">
              <rect x="40" y="10" width="340" height="340"></rect>
            </clipPath>
            <linearGradient id="drl-neuron-legend" x1="0" y1="1" x2="0" y2="0">
              <stop class="drl-legend-low" offset="0"></stop>
              <stop class="drl-legend-high" offset="1"></stop>
            </linearGradient>
          </defs>
          <rect
            class="drl-plane"
            x="40"
            y="10"
            width="340"
            height="340"
            fill="var(--drl-bg)"
          ></rect>
          <g clip-path="url(#drl-neuron-plane)">
            <g class="drl-grid"></g>
          </g>
          <rect
            class="drl-plane-outline"
            x="40"
            y="10"
            width="340"
            height="340"
            fill="none"
            stroke="var(--drl-border)"
          ></rect>
          <g class="drl-points"></g>
          <circle
            class="drl-halo"
            r="12"
            fill="none"
            stroke="var(--drl-fg)"
            stroke-opacity="0.7"
            visibility="hidden"
          ></circle>
          <text class="drl-axis" x="40" y="356" text-anchor="start">${this.axisLabel('x₁')}</text>
          <text class="drl-axis" x="34" y="14" text-anchor="end">x₂</text>
          <rect
            class="drl-legend"
            x="420"
            y="10"
            width="24"
            height="200"
            fill="url(#drl-neuron-legend)"
          ></rect>
          <text class="drl-axis" x="452" y="18">t max</text>
          <text class="drl-axis" x="452" y="212">t min</text>
        </svg>
        ${
          this.webglError
            ? html`<p class="drl-webgl-error">
              WebGL n'est pas disponible dans cet environnement.
            </p>`
            : html`<canvas
              width=${CANVAS_WIDTH}
              height=${CANVAS_HEIGHT}
              role="img"
              aria-label="Surface de loss 3D et descente de la bille par batch"
            ></canvas>`
        }
      </div>
    `;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    if (this.hasUpdated && this.scene === null && this.setupPromise === null) {
      void this.setupScene();
    }
  }

  override disconnectedCallback(): void {
    if (this.animFrame !== null) {
      cancelAnimationFrame(this.animFrame);
      this.animFrame = null;
    }
    this.tweening = false;
    this.scene?.dispose();
    this.scene = null;
    // The disposed scene owned the height field: drop the cache so the next
    // setupScene pushes the field onto the new scene instead of skipping it.
    this.fieldDomain = Number.NaN;
    this.fieldDataset = null;
    this.fieldLoss = null;
    this.fieldBias = null;
    this.fieldB = Number.NaN;
    // The next sync() must re-broadcast the weights, so drop the change cache.
    this.lastWeights = null;
    super.disconnectedCallback();
  }
}
