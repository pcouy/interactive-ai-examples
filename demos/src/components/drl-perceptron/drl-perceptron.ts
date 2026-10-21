import type { BaseType, Selection } from 'd3';
import { select } from 'd3';
import type { PropertyValues, TemplateResult } from 'lit';
import { css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { DemoElement } from '../../base/demo-element.js';
import type { DatasetName, LabeledPoint } from '../../core/perceptron.js';
import { PerceptronCore, PRESET_DATASETS } from '../../core/perceptron.js';
import { drlTokens } from '../../styles/tokens.js';

const PLANE_X = 40;
const PLANE_Y = 10;
const PLANE_SIZE = 340;
const PLANE_RIGHT = PLANE_X + PLANE_SIZE;
const PLANE_BOTTOM = PLANE_Y + PLANE_SIZE;
const EPSILON = 1e-6;
const MISSING_VALUE = '–';

type Point = [number, number];

interface PendingPrediction {
  index: number;
  point: LabeledPoint;
  prediction: 0 | 1;
}

interface AppliedUpdate {
  point: LabeledPoint;
  prediction: 0 | 1;
  expected: 0 | 1;
  learningRate: number;
  before: { w1: number; w2: number; bias: number };
  after: { w1: number; w2: number; bias: number };
}

export interface PerceptronState {
  /** Poids de x₁. */
  w1: number;
  /** Poids de x₂. */
  w2: number;
  /** Biais ajouté à la somme pondérée. */
  bias: number;
  /** Seuil de décision : seuil = −bias. */
  threshold: number;
  /** Coordonnée x₁ du dernier point présenté, ou null si aucun point. */
  x1: number | null;
  /** Coordonnée x₂ du dernier point présenté, ou null si aucun point. */
  x2: number | null;
  /** Somme pondérée w₁·x₁ + w₂·x₂, ou null si aucun point. */
  y: number | null;
  /** Classe prédite (0 ou 1), ou null si aucun point. */
  decision: 0 | 1 | null;
  /** Classe attendue du point présenté (0 ou 1), ou null si aucun point. */
  expected: 0 | 1 | null;
}

type ReactiveDataset = DatasetName & DOMStringMap;

type SingleSelection<ElementType extends BaseType> = Selection<
  ElementType,
  unknown,
  null,
  undefined
>;

const PLANE_CORNERS: Point[] = [
  [PLANE_X, PLANE_Y],
  [PLANE_RIGHT, PLANE_Y],
  [PLANE_RIGHT, PLANE_BOTTOM],
  [PLANE_X, PLANE_BOTTOM],
];

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function dataX(x: number): number {
  return PLANE_X + x * PLANE_SIZE;
}

function dataY(y: number): number {
  return PLANE_BOTTOM - y * PLANE_SIZE;
}

function formatNumber(value: number): string {
  return value.toFixed(2).replace('.', ',');
}

function formatCoord(value: number | undefined): string {
  return value === undefined ? MISSING_VALUE : formatNumber(value);
}

function classLabel(label: 0 | 1): string {
  return label === 1 ? 'classe bleue' : 'classe violette';
}

function weightWidth(weight: number): number {
  return clamp(1 + 6 * Math.abs(weight), 1, 8);
}

function weightColor(weight: number): string {
  return weight >= 0 ? 'var(--drl-accent)' : 'var(--drl-accent-2)';
}

function isDatasetName(value: string): value is DatasetName {
  return Object.hasOwn(PRESET_DATASETS, value);
}

function linePlaneSegment(w1: number, w2: number, bias: number): [Point, Point] | null {
  const points: Point[] = [];
  const addPoint = (x: number, y: number): void => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return;
    }
    if (x < PLANE_X - EPSILON || x > PLANE_RIGHT + EPSILON) {
      return;
    }
    if (y < PLANE_Y - EPSILON || y > PLANE_BOTTOM + EPSILON) {
      return;
    }
    if (points.some(([px, py]) => Math.abs(px - x) < EPSILON && Math.abs(py - y) < EPSILON)) {
      return;
    }
    points.push([x, y]);
  };
  if (Math.abs(w2) > EPSILON) {
    addPoint(PLANE_X, PLANE_BOTTOM + (PLANE_SIZE * bias) / w2);
    addPoint(PLANE_RIGHT, PLANE_BOTTOM + (PLANE_SIZE * (w1 + bias)) / w2);
  }
  if (Math.abs(w1) > EPSILON) {
    addPoint(PLANE_X - (PLANE_SIZE * (w2 + bias)) / w1, PLANE_Y);
    addPoint(PLANE_X - (PLANE_SIZE * bias) / w1, PLANE_BOTTOM);
  }
  if (points.length < 2) {
    return null;
  }
  return [points[0], points[1]];
}

function clipHalfPlane(
  polygon: Point[],
  valueAt: (x: number, y: number) => number,
  keepPositive: boolean,
): Point[] {
  const isInside = (x: number, y: number): boolean => {
    const value = valueAt(x, y);
    return keepPositive ? value >= 0 : value <= 0;
  };
  const result: Point[] = [];
  for (let index = 0; index < polygon.length; index += 1) {
    const current = polygon[index];
    const next = polygon[(index + 1) % polygon.length];
    const currentInside = isInside(current[0], current[1]);
    const nextInside = isInside(next[0], next[1]);
    if (currentInside) {
      result.push(current);
    }
    if (currentInside !== nextInside) {
      const currentValue = valueAt(current[0], current[1]);
      const nextValue = valueAt(next[0], next[1]);
      const ratio = currentValue / (currentValue - nextValue);
      result.push([
        current[0] + ratio * (next[0] - current[0]),
        current[1] + ratio * (next[1] - current[1]),
      ]);
    }
  }
  return result;
}

@customElement('drl-perceptron')
export class Perceptron extends DemoElement {
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
        display: none;
        background: var(--drl-accent);
        color: var(--drl-surface);
        border-radius: calc(var(--drl-radius) / 2);
        padding: 0 var(--drl-space-2);
        font-size: 0.85em;
      }

      .drl-explanation {
        margin-bottom: var(--drl-space-2);
        padding: var(--drl-space-2);
        background: var(--drl-bg);
        border: 1px solid var(--drl-border);
        border-left: 3px solid var(--drl-accent-2);
        border-radius: calc(var(--drl-radius) / 2);
      }

      .drl-explanation[hidden],
      .drl-rule[hidden] {
        display: none;
      }

      .drl-explanation[data-phase='update'] {
        border-left-color: var(--drl-accent);
      }

      .drl-explanation-headline {
        margin: 0 0 var(--drl-space-1);
        font-weight: 600;
      }

      .drl-explanation[data-phase='prediction'] .drl-explanation-headline {
        color: var(--drl-accent-2);
      }

      .drl-explanation[data-phase='update'] .drl-explanation-headline {
        color: var(--drl-accent);
      }

      .drl-explanation-detail,
      .drl-explanation-hint {
        margin: 0;
      }

      .drl-explanation-hint {
        margin-top: var(--drl-space-1);
        color: var(--drl-muted);
        font-size: 0.9em;
      }

      .drl-rule {
        margin-top: var(--drl-space-2);
        padding-top: var(--drl-space-2);
        border-top: 1px dashed var(--drl-border);
      }

      .drl-rule-title {
        margin: 0 0 var(--drl-space-1);
        color: var(--drl-muted);
        font-size: 0.85em;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }

      .drl-rule-line {
        margin: 0.15rem 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.9em;
        overflow-wrap: anywhere;
      }

      .drl-toolbar {
        display: flex;
        flex-wrap: wrap;
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

      .drl-button:hover {
        border-color: var(--drl-accent);
      }

      .drl-button[aria-pressed='true'] {
        border-color: var(--drl-accent);
        color: var(--drl-accent);
      }

      .drl-dot {
        display: inline-block;
        width: 0.6em;
        height: 0.6em;
        border-radius: 50%;
      }

      .drl-dot--1 {
        background: var(--drl-accent);
      }

      .drl-dot--0 {
        background: var(--drl-accent-2);
      }

      svg {
        display: block;
        width: 100%;
        max-width: 640px;
        height: auto;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
      }

      .drl-plane-fill,
      .drl-plane-outline {
        cursor: pointer;
      }

      .drl-boundary {
        fill: none;
      }

      .drl-neuron-label {
        fill: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: 13px;
      }

      .drl-secondary {
        fill: var(--drl-muted);
        font-family: var(--drl-font);
        font-size: 12px;
      }

      .drl-weight {
        fill: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: 12px;
      }

      .drl-arrow {
        stroke: var(--drl-muted);
        stroke-width: 1.5;
      }

      .drl-bias-arrow {
        stroke: var(--drl-fg);
        stroke-width: 2;
      }

      .drl-bias-head {
        fill: var(--drl-fg);
      }
    `,
  ];

  @property({ type: Number, attribute: 'learning-rate' })
  learningRate = 0.1;

  @property({ type: String, attribute: 'dataset' })
  dataset: ReactiveDataset = 'separable' as unknown as ReactiveDataset;

  private get datasetName(): DatasetName {
    return this.dataset;
  }

  private set datasetName(value: DatasetName) {
    (this as unknown as { dataset: DatasetName }).dataset = value;
  }

  private readonly core = new PerceptronCore();

  private activeClass: 0 | 1 = 1;

  private statusText: HTMLElement | null = null;

  private statusBadge: HTMLElement | null = null;

  private explanation: HTMLElement | null = null;

  private explanationHeadline: HTMLElement | null = null;

  private explanationDetail: HTMLElement | null = null;

  private explanationHint: HTMLElement | null = null;

  private explanationRule: HTMLElement | null = null;

  private ruleW1: HTMLElement | null = null;

  private ruleW2: HTMLElement | null = null;

  private ruleBias: HTMLElement | null = null;

  private boundary: SingleSelection<SVGLineElement> | null = null;

  private tintOne: SingleSelection<SVGPolygonElement> | null = null;

  private tintZero: SingleSelection<SVGPolygonElement> | null = null;

  private pointsGroup: SingleSelection<SVGGElement> | null = null;

  private halo: SingleSelection<SVGCircleElement> | null = null;

  private inputValueOne: SingleSelection<SVGTextElement> | null = null;

  private inputValueTwo: SingleSelection<SVGTextElement> | null = null;

  private output: SingleSelection<SVGCircleElement> | null = null;

  private arrowOne: SingleSelection<SVGLineElement> | null = null;

  private arrowTwo: SingleSelection<SVGLineElement> | null = null;

  private headOne: SingleSelection<SVGPolygonElement> | null = null;

  private headTwo: SingleSelection<SVGPolygonElement> | null = null;

  private weightOne: SingleSelection<SVGTextElement> | null = null;

  private weightTwo: SingleSelection<SVGTextElement> | null = null;

  private biasLabel: SingleSelection<SVGTextElement> | null = null;

  /** Dernier état diffusé, pour n'émettre drl-state qu'en cas de changement. */
  private lastState: PerceptronState | null = null;

  /** Prédiction erronée affichée en mode Step, en attente de la mise à jour. */
  private pendingPrediction: PendingPrediction | null = null;

  /** Mise à jour appliquée, affichée jusqu'au prochain pas manuel. */
  private appliedUpdate: AppliedUpdate | null = null;

  /** Index affiché : le point en attente prime sur le dernier point présenté. */
  private get displayedIndex(): number | null {
    return this.pendingPrediction?.index ?? this.core.lastPresentedIndex;
  }

  /** État courant de la formule : poids, seuil et dernier point présenté. */
  get state(): PerceptronState {
    const { w1, w2 } = this.core.weights;
    const bias = this.core.bias;
    const index = this.displayedIndex;
    const point =
      index !== null && index >= 0 && index < this.core.points.length
        ? this.core.points[index]
        : null;
    return {
      w1,
      w2,
      bias,
      threshold: -bias,
      x1: point?.x ?? null,
      x2: point?.y ?? null,
      y: point ? w1 * point.x + w2 * point.y : null,
      decision: point ? this.core.predictionAt(point.x, point.y) : null,
      expected: point?.label ?? null,
    };
  }

  protected override firstUpdated(): void {
    const svgElement = this.renderRoot.querySelector<SVGSVGElement>('svg');
    if (!svgElement) {
      return;
    }
    const root = select(svgElement);
    this.boundary = root.select<SVGLineElement>('.drl-boundary');
    this.tintOne = root.select<SVGPolygonElement>('.drl-tint--1');
    this.tintZero = root.select<SVGPolygonElement>('.drl-tint--0');
    this.pointsGroup = root.select<SVGGElement>('.drl-points');
    this.halo = root.select<SVGCircleElement>('.drl-halo');
    this.inputValueOne = root.select<SVGTextElement>('.drl-input-value--1');
    this.inputValueTwo = root.select<SVGTextElement>('.drl-input-value--2');
    this.output = root.select<SVGCircleElement>('.drl-output');
    this.arrowOne = root.select<SVGLineElement>('.drl-arrow--1');
    this.arrowTwo = root.select<SVGLineElement>('.drl-arrow--2');
    this.headOne = root.select<SVGPolygonElement>('.drl-head--1');
    this.headTwo = root.select<SVGPolygonElement>('.drl-head--2');
    this.weightOne = root.select<SVGTextElement>('.drl-weight--1');
    this.weightTwo = root.select<SVGTextElement>('.drl-weight--2');
    this.biasLabel = root.select<SVGTextElement>('.drl-bias-label');
    this.statusText = this.renderRoot.querySelector<HTMLElement>('.drl-status-text');
    this.statusBadge = this.renderRoot.querySelector<HTMLElement>('.drl-badge');
    this.explanation = this.renderRoot.querySelector<HTMLElement>('.drl-explanation');
    this.explanationHeadline = this.renderRoot.querySelector<HTMLElement>(
      '.drl-explanation-headline',
    );
    this.explanationDetail = this.renderRoot.querySelector<HTMLElement>('.drl-explanation-detail');
    this.explanationHint = this.renderRoot.querySelector<HTMLElement>('.drl-explanation-hint');
    this.explanationRule = this.renderRoot.querySelector<HTMLElement>('.drl-rule');
    this.ruleW1 = this.renderRoot.querySelector<HTMLElement>('.drl-rule-line--w1');
    this.ruleW2 = this.renderRoot.querySelector<HTMLElement>('.drl-rule-line--w2');
    this.ruleBias = this.renderRoot.querySelector<HTMLElement>('.drl-rule-line--b');
    this.draw();
  }

  override step(): void {
    if (this.running) {
      // En lecture automatique, la règle n'a pas le temps d'être lue : on
      // applique la mise à jour immédiatement, comme avant.
      this.clearManualState();
      this.core.tick(this.learningRate);
      this.draw();
      return;
    }
    this.manualStep();
  }

  /**
   * Pas manuel en deux temps : d'abord la prédiction (sans toucher aux poids),
   * puis, au clic suivant, la règle de mise à jour et son application.
   */
  private manualStep(): void {
    if (this.appliedUpdate !== null) {
      this.appliedUpdate = null;
    }
    if (this.pendingPrediction !== null) {
      this.applyPendingPrediction();
      return;
    }
    const index = this.core.nextIndex;
    const point = this.core.nextPoint;
    if (this.core.isConverged || index === null || point === null) {
      this.draw();
      return;
    }
    const prediction = this.core.predictionAt(point.x, point.y);
    if (prediction !== point.label) {
      this.pendingPrediction = { index, point, prediction };
      this.draw();
      return;
    }
    // Prédiction correcte : aucun poids à changer, on avance d'un point.
    this.core.tick(this.learningRate);
    this.draw();
  }

  private applyPendingPrediction(): void {
    const pending = this.pendingPrediction;
    if (pending === null) {
      return;
    }
    const before = this.weightSnapshot();
    this.core.tick(this.learningRate);
    const after = this.weightSnapshot();
    this.appliedUpdate = {
      point: pending.point,
      prediction: pending.prediction,
      expected: pending.point.label,
      learningRate: this.learningRate,
      before,
      after,
    };
    this.pendingPrediction = null;
    this.draw();
  }

  private weightSnapshot(): { w1: number; w2: number; bias: number } {
    return { w1: this.core.weights.w1, w2: this.core.weights.w2, bias: this.core.bias };
  }

  private clearManualState(): void {
    this.pendingPrediction = null;
    this.appliedUpdate = null;
  }

  protected override onReset(): void {
    this.clearManualState();
    this.core.reset();
    this.draw();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    super.willUpdate(changed);
    if (changed.has('dataset') && isDatasetName(this.datasetName)) {
      this.clearManualState();
      this.core.setData(PRESET_DATASETS[this.datasetName]());
      this.draw();
    }
  }

  private readonly handlePresetClick = (event: Event): void => {
    const button = event.currentTarget as HTMLButtonElement;
    const name = button.dataset.preset;
    if (name && isDatasetName(name)) {
      this.datasetName = name;
      this.requestUpdate();
    }
  };

  private readonly handleClassClick = (event: Event): void => {
    const button = event.currentTarget as HTMLButtonElement;
    this.activeClass = button.dataset.activeClass === '0' ? 0 : 1;
    this.requestUpdate();
  };

  private readonly handleClear = (): void => {
    this.clearManualState();
    this.core.clearPoints();
    this.draw();
    this.requestUpdate();
  };

  private readonly handlePointerDown = (event: PointerEvent): void => {
    const target = event.currentTarget;
    if (!(target instanceof SVGSVGElement)) {
      return;
    }
    const rect = target.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return;
    }
    const px = (event.clientX - rect.left) * (640 / rect.width);
    const py = (event.clientY - rect.top) * (360 / rect.height);
    if (px < PLANE_X || px > PLANE_RIGHT || py < PLANE_Y || py > PLANE_BOTTOM) {
      return;
    }
    const x = clamp((px - PLANE_X) / PLANE_SIZE, 0, 1);
    const y = clamp((PLANE_BOTTOM - py) / PLANE_SIZE, 0, 1);
    this.core.addPoint({ x, y, label: this.activeClass });
    this.draw();
  };

  private updateTint(
    selection: SingleSelection<SVGPolygonElement> | null,
    polygon: Point[] | null,
  ): void {
    if (!selection) {
      return;
    }
    if (!polygon || polygon.length < 3) {
      selection.attr('visibility', 'hidden');
      return;
    }
    selection
      .attr('points', polygon.map(([x, y]) => `${x},${y}`).join(' '))
      .attr('visibility', 'visible');
  }

  private broadcastState(): void {
    const state = this.state;
    const last = this.lastState;
    if (
      last !== null &&
      last.w1 === state.w1 &&
      last.w2 === state.w2 &&
      last.bias === state.bias &&
      last.x1 === state.x1 &&
      last.x2 === state.x2 &&
      last.decision === state.decision &&
      last.expected === state.expected
    ) {
      return;
    }
    this.lastState = state;
    this.dispatchEvent(
      new CustomEvent<PerceptronState>('drl-state', {
        detail: state,
        bubbles: true,
        composed: true,
      }),
    );
  }

  private renderExplanation(): void {
    const explanation = this.explanation;
    if (!explanation) {
      return;
    }
    const pending = this.pendingPrediction;
    const update = this.appliedUpdate;
    if (this.running || (pending === null && update === null)) {
      explanation.hidden = true;
      return;
    }
    explanation.hidden = false;
    explanation.dataset.phase = pending !== null ? 'prediction' : 'update';

    const prediction = pending?.prediction ?? update?.prediction;
    const expected = pending?.point.label ?? update?.expected;
    if (prediction === undefined || expected === undefined) {
      explanation.hidden = true;
      return;
    }

    if (this.explanationHeadline) {
      this.explanationHeadline.textContent =
        pending !== null ? 'Prédiction erronée' : 'Règle de mise à jour appliquée';
    }
    if (this.explanationDetail) {
      this.explanationDetail.textContent =
        `ŷ = ${prediction} (${classLabel(prediction)}) · ` +
        `attendu t = ${expected} (${classLabel(expected)}).`;
    }
    if (this.explanationHint) {
      this.explanationHint.textContent =
        pending !== null
          ? 'Step : afficher la règle de mise à jour.'
          : 'Step : présenter le point suivant.';
    }
    if (this.explanationRule) {
      this.explanationRule.hidden = update === null;
    }
    if (update === null) {
      return;
    }

    const rate = formatNumber(update.learningRate);
    const error = `(${update.expected} − ${update.prediction})`;
    const x1 = formatNumber(update.point.x);
    const x2 = formatNumber(update.point.y);
    if (this.ruleW1) {
      this.ruleW1.textContent =
        `w₁ ← w₁ + η·(t − ŷ)·x₁ = ${formatNumber(update.before.w1)} + ` +
        `${rate}·${error}·${x1} = ${formatNumber(update.after.w1)}`;
    }
    if (this.ruleW2) {
      this.ruleW2.textContent =
        `w₂ ← w₂ + η·(t − ŷ)·x₂ = ${formatNumber(update.before.w2)} + ` +
        `${rate}·${error}·${x2} = ${formatNumber(update.after.w2)}`;
    }
    if (this.ruleBias) {
      this.ruleBias.textContent =
        `b ← b + η·(t − ŷ) = ${formatNumber(update.before.bias)} + ` +
        `${rate}·${error} = ${formatNumber(update.after.bias)}`;
    }
  }

  private draw(): void {
    const boundary = this.boundary;
    const pointsGroup = this.pointsGroup;
    const halo = this.halo;
    if (!boundary || !pointsGroup || !halo) {
      return;
    }

    const { w1, w2 } = this.core.weights;
    const bias = this.core.bias;
    const hasBoundary = Math.hypot(w1, w2) >= EPSILON;
    const valueAt = (x: number, y: number): number =>
      w1 * (x - PLANE_X) + w2 * (PLANE_BOTTOM - y) + PLANE_SIZE * bias;

    const segment = hasBoundary ? linePlaneSegment(w1, w2, bias) : null;
    if (segment) {
      boundary
        .attr('x1', segment[0][0])
        .attr('y1', segment[0][1])
        .attr('x2', segment[1][0])
        .attr('y2', segment[1][1])
        .attr('visibility', 'visible');
    } else {
      boundary.attr('visibility', 'hidden');
    }

    this.updateTint(this.tintOne, hasBoundary ? clipHalfPlane(PLANE_CORNERS, valueAt, true) : null);
    this.updateTint(
      this.tintZero,
      hasBoundary ? clipHalfPlane(PLANE_CORNERS, valueAt, false) : null,
    );

    const points = this.core.points;
    const misclassified = new Set(this.core.misclassifiedIndices);
    pointsGroup
      .selectAll<SVGCircleElement, number>('circle')
      .data(
        points.map((_, index) => index),
        (index) => String(index),
      )
      .join('circle')
      .attr('r', 7)
      .attr('cx', (index) => dataX(points[index].x))
      .attr('cy', (index) => dataY(points[index].y))
      .attr('fill', (index) =>
        points[index].label === 1 ? 'var(--drl-accent)' : 'var(--drl-accent-2)',
      )
      .attr('stroke', (index) => (misclassified.has(index) ? 'var(--drl-fg)' : 'none'))
      .attr('stroke-width', (index) => (misclassified.has(index) ? 2 : 0));

    const currentIndex = this.displayedIndex;
    const current: LabeledPoint | null =
      currentIndex !== null && currentIndex >= 0 && currentIndex < points.length
        ? points[currentIndex]
        : null;

    if (current) {
      halo.attr('cx', dataX(current.x)).attr('cy', dataY(current.y)).attr('visibility', 'visible');
    } else {
      halo.attr('visibility', 'hidden');
    }

    this.inputValueOne?.text(formatCoord(current?.x));
    this.inputValueTwo?.text(formatCoord(current?.y));

    const outputFill =
      current === null
        ? 'var(--drl-surface)'
        : this.core.predictionAt(current.x, current.y) === 1
          ? 'var(--drl-accent)'
          : 'var(--drl-accent-2)';
    this.output?.attr('fill', outputFill);

    this.arrowOne?.attr('stroke-width', weightWidth(w1)).attr('stroke', weightColor(w1));
    this.arrowTwo?.attr('stroke-width', weightWidth(w2)).attr('stroke', weightColor(w2));
    this.headOne?.attr('fill', weightColor(w1));
    this.headTwo?.attr('fill', weightColor(w2));
    this.weightOne?.text(`w₁ = ${formatNumber(w1)}`);
    this.weightTwo?.text(`w₂ = ${formatNumber(w2)}`);
    this.biasLabel?.text(`b = ${formatNumber(bias)}`);

    if (this.statusText) {
      this.statusText.textContent =
        `Époque ${this.core.epochs} · Précision ${Math.round(this.core.accuracy * 100)} %` +
        ` · ${this.core.ticksElapsed} présentations`;
    }
    if (this.statusBadge) {
      const converged = this.core.isConverged;
      this.statusBadge.textContent = converged
        ? `Convergé en ${this.core.stepsToConvergence} étapes`
        : '';
      this.statusBadge.style.display = converged ? 'inline-block' : 'none';
    }

    this.renderExplanation();
    this.broadcastState();
  }

  protected override renderContent(): TemplateResult {
    return html`
      <div class="drl-status">
        <span class="drl-status-text"></span>
        <span class="drl-badge"></span>
      </div>
      <div class="drl-toolbar" role="group" aria-label="Contrôles du perceptron">
        <button
          type="button"
          class="drl-button"
          data-preset="separable"
          aria-pressed=${this.datasetName === 'separable' ? 'true' : 'false'}
          @click=${this.handlePresetClick}
        >
          Séparables
        </button>
        <button
          type="button"
          class="drl-button"
          data-preset="noisy"
          aria-pressed=${this.datasetName === 'noisy' ? 'true' : 'false'}
          @click=${this.handlePresetClick}
        >
          Bruités
        </button>
        <button
          type="button"
          class="drl-button"
          data-preset="non-separable"
          aria-pressed=${this.datasetName === 'non-separable' ? 'true' : 'false'}
          @click=${this.handlePresetClick}
        >
          Non séparables
        </button>
        <button
          type="button"
          class="drl-button"
          data-active-class="1"
          aria-pressed=${this.activeClass === 1 ? 'true' : 'false'}
          @click=${this.handleClassClick}
        >
          <span class="drl-dot drl-dot--1" aria-hidden="true"></span>Classe bleue
        </button>
        <button
          type="button"
          class="drl-button"
          data-active-class="0"
          aria-pressed=${this.activeClass === 0 ? 'true' : 'false'}
          @click=${this.handleClassClick}
        >
          <span class="drl-dot drl-dot--0" aria-hidden="true"></span>Classe violette
        </button>
        <button type="button" class="drl-button" @click=${this.handleClear}>Effacer</button>
      </div>
      <div class="drl-explanation" role="status" aria-live="polite" hidden>
        <p class="drl-explanation-headline"></p>
        <p class="drl-explanation-detail"></p>
        <div class="drl-rule" hidden>
          <p class="drl-rule-title">Règle de mise à jour</p>
          <p class="drl-rule-line drl-rule-line--w1"></p>
          <p class="drl-rule-line drl-rule-line--w2"></p>
          <p class="drl-rule-line drl-rule-line--b"></p>
        </div>
        <p class="drl-explanation-hint"></p>
      </div>
      <svg
        viewBox="0 0 640 360"
        role="img"
        aria-label="Plan de points et frontière de décision du perceptron"
        @pointerdown=${this.handlePointerDown}
      >
        <defs>
          <clipPath id="drl-perceptron-plane">
            <rect x="40" y="10" width="340" height="340"></rect>
          </clipPath>
        </defs>
        <rect
          class="drl-plane-fill"
          x="40"
          y="10"
          width="340"
          height="340"
          fill="var(--drl-bg)"
        ></rect>
        <g clip-path="url(#drl-perceptron-plane)">
          <polygon
            class="drl-tint drl-tint--1"
            fill="var(--drl-accent)"
            fill-opacity="0.07"
            visibility="hidden"
          ></polygon>
          <polygon
            class="drl-tint drl-tint--0"
            fill="var(--drl-accent-2)"
            fill-opacity="0.07"
            visibility="hidden"
          ></polygon>
          <line
            class="drl-boundary"
            stroke="var(--drl-fg)"
            stroke-width="2.5"
            visibility="hidden"
          ></line>
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
        <circle
          class="drl-input"
          cx="432"
          cy="110"
          r="16"
          fill="var(--drl-surface)"
          stroke="var(--drl-border)"
        ></circle>
        <text class="drl-neuron-label" x="432" y="115" text-anchor="middle">x₁</text>
        <text
          class="drl-secondary drl-input-value drl-input-value--1"
          x="432"
          y="148"
          text-anchor="middle"
        ></text>
        <circle
          class="drl-input"
          cx="432"
          cy="210"
          r="16"
          fill="var(--drl-surface)"
          stroke="var(--drl-border)"
        ></circle>
        <text class="drl-neuron-label" x="432" y="215" text-anchor="middle">x₂</text>
        <text
          class="drl-secondary drl-input-value drl-input-value--2"
          x="432"
          y="248"
          text-anchor="middle"
        ></text>
        <line
          class="drl-arrow drl-arrow--1"
          x1="446.09"
          y1="117.58"
          x2="502.10"
          y2="147.69"
          fill="none"
          stroke-linecap="round"
        ></line>
        <polygon class="drl-head drl-head--1" points="502.1,147.69 491.16,146.91 495.42,138.99"></polygon>
        <text class="drl-weight drl-weight--1" x="474" y="119" text-anchor="middle"></text>
        <line
          class="drl-arrow drl-arrow--2"
          x1="446.09"
          y1="202.42"
          x2="502.10"
          y2="172.31"
          fill="none"
          stroke-linecap="round"
        ></line>
        <polygon class="drl-head drl-head--2" points="502.1,172.31 495.42,181.01 491.16,173.09"></polygon>
        <text class="drl-weight drl-weight--2" x="474" y="201" text-anchor="middle"></text>
        <circle
          class="drl-sum"
          cx="525"
          cy="160"
          r="26"
          fill="var(--drl-surface)"
          stroke="var(--drl-border)"
        ></circle>
        <text class="drl-neuron-label" x="525" y="166" text-anchor="middle">Σ</text>
        <text class="drl-secondary" x="525" y="212" text-anchor="middle">seuil</text>
        <line
          class="drl-arrow drl-bias-arrow"
          x1="525"
          y1="330"
          x2="525"
          y2="196"
          fill="none"
          stroke-linecap="round"
        ></line>
        <polygon class="drl-head drl-bias-head" points="525,186 529.5,196 520.5,196"></polygon>
        <text class="drl-weight drl-bias-label" x="535" y="260" text-anchor="start"></text>
        <circle
          class="drl-output"
          cx="612"
          cy="160"
          r="18"
          fill="var(--drl-surface)"
          stroke="var(--drl-border)"
        ></circle>
        <text class="drl-neuron-label" x="612" y="165" text-anchor="middle">ŷ</text>
      </svg>
    `;
  }
}
