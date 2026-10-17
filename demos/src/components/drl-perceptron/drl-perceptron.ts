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
    this.draw();
  }

  override step(): void {
    this.core.tick(this.learningRate);
    this.draw();
  }

  protected override onReset(): void {
    this.core.reset();
    this.draw();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    super.willUpdate(changed);
    if (changed.has('dataset') && isDatasetName(this.datasetName)) {
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

    const currentIndex = this.core.lastPresentedIndex;
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
