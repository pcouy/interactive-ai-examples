import type { BaseType, Selection } from 'd3';
import { select } from 'd3';
import type { PropertyValues, TemplateResult } from 'lit';
import { css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { DemoElement } from '../../base/demo-element.js';
import type { FrozenLakeAction, FrozenLakeCell } from '../../core/frozen-lake.js';
import { FrozenLakeCore } from '../../core/frozen-lake.js';
import { drlTokens } from '../../styles/tokens.js';

const GRID_SIZE = 480;
const CURVE_WIDTH = 480;
const CURVE_HEIGHT = 90;
const ARROW_PATH =
  'M -0.30 -0.12 L 0.05 -0.12 L 0.05 -0.26 L 0.34 0 L 0.05 0.26 L 0.05 0.12 L -0.30 0.12 Z';

const ACTION_ROTATION: Record<FrozenLakeAction, number> = {
  right: 0,
  down: 90,
  left: 180,
  up: 270,
};

type SingleSelection<ElementType extends BaseType> = Selection<
  ElementType,
  unknown,
  null,
  undefined
>;

type JoinedSelection<ElementType extends BaseType, Datum> = Selection<
  ElementType,
  Datum,
  SVGGElement,
  unknown
>;

@customElement('drl-frozen-lake')
export class FrozenLake extends DemoElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: block;
      }

      .drl-title {
        margin: 0 0 var(--drl-space-2);
        font-size: 1.05em;
      }

      .drl-lead {
        margin: 0 0 var(--drl-space-3);
        max-width: 60ch;
        color: var(--drl-muted);
      }

      .drl-status-text {
        display: block;
        margin-bottom: var(--drl-space-2);
        color: var(--drl-muted);
        font-variant-numeric: tabular-nums;
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

      svg {
        display: block;
        width: 100%;
        max-width: 480px;
        height: auto;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
      }

      .drl-grid {
        margin-bottom: var(--drl-space-2);
      }

      .drl-cell {
        stroke: var(--drl-border);
        stroke-width: 1;
      }

      .drl-cell--start {
        fill: var(--drl-accent);
        fill-opacity: 0.15;
      }

      .drl-cell--frozen {
        fill: var(--drl-bg);
      }

      .drl-cell--hole {
        fill: var(--drl-fg);
      }

      .drl-cell--goal {
        fill: var(--drl-accent);
      }

      .drl-arrow {
        fill: var(--drl-fg);
      }

      .drl-agent {
        fill: var(--drl-accent-2);
        stroke: var(--drl-surface);
        stroke-width: 2;
      }

      .drl-curve {
        fill: none;
        stroke: var(--drl-accent);
        stroke-width: 2;
      }
    `,
  ];

  @property({ type: Number, attribute: 'learning-rate' })
  learningRate = 0.1;

  @property({ type: Number, attribute: 'epsilon' })
  epsilon = 0.1;

  @property({ type: Number, attribute: 'discount-factor' })
  discountFactor = 0.99;

  @property({ type: Number, attribute: 'grid-size' })
  gridSize = 4;

  @property({ type: Boolean, attribute: 'slippery' })
  slippery = false;

  private readonly core = new FrozenLakeCore();

  private cellsGroup: SingleSelection<SVGGElement> | null = null;

  private arrowsGroup: SingleSelection<SVGGElement> | null = null;

  private cells: JoinedSelection<SVGRectElement, number> | null = null;

  private arrows: JoinedSelection<SVGPathElement, number> | null = null;

  private agent: SingleSelection<SVGCircleElement> | null = null;

  private curve: SingleSelection<SVGPathElement> | null = null;

  private statusText: HTMLElement | null = null;

  private get size(): 4 | 8 {
    return this.gridSize === 8 ? 8 : 4;
  }

  protected override firstUpdated(): void {
    const gridSvg = this.renderRoot.querySelector<SVGSVGElement>('svg.drl-grid');
    if (gridSvg) {
      const root = select(gridSvg);
      this.cellsGroup = root.select<SVGGElement>('.drl-cells');
      this.arrowsGroup = root.select<SVGGElement>('.drl-arrows');
      this.agent = root.select<SVGCircleElement>('.drl-agent');
    }
    const curvePath = this.renderRoot.querySelector<SVGPathElement>('path.drl-curve');
    this.curve = curvePath ? select(curvePath) : null;
    this.statusText = this.renderRoot.querySelector<HTMLElement>('.drl-status-text');
    this.buildGrid();
    this.draw();
  }

  override step(): void {
    this.core.tick();
    this.draw();
  }

  protected override onReset(): void {
    this.core.reset();
    this.draw();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    super.willUpdate(changed);
    this.core.learningRate = this.learningRate;
    this.core.epsilon = this.epsilon;
    this.core.discountFactor = this.discountFactor;
    if (changed.has('gridSize')) {
      this.core.setSize(this.size);
      this.buildGrid();
      this.draw();
    }
    if (changed.has('slippery')) {
      this.core.setSlippery(this.slippery);
      this.draw();
    }
  }

  private readonly handleSize4 = (): void => {
    this.gridSize = 4;
    this.requestUpdate();
  };

  private readonly handleSize8 = (): void => {
    this.gridSize = 8;
    this.requestUpdate();
  };

  private readonly handleSlippery = (): void => {
    this.slippery = !this.slippery;
    this.requestUpdate();
  };

  private buildGrid(): void {
    const cellsGroup = this.cellsGroup;
    const arrowsGroup = this.arrowsGroup;
    if (!cellsGroup || !arrowsGroup) {
      return;
    }
    const size = this.size;
    const cellSize = GRID_SIZE / size;
    const indices = Array.from({ length: size * size }, (_, index) => index);

    this.cells = cellsGroup
      .selectAll<SVGRectElement, number>('rect.drl-cell')
      .data(indices, (index) => String(index))
      .join('rect')
      .attr('class', 'drl-cell')
      .attr('width', cellSize)
      .attr('height', cellSize)
      .attr('x', (index) => (index % size) * cellSize)
      .attr('y', (index) => Math.floor(index / size) * cellSize);

    this.arrows = arrowsGroup
      .selectAll<SVGPathElement, number>('path.drl-arrow')
      .data(indices, (index) => String(index))
      .join('path')
      .attr('class', 'drl-arrow')
      .attr('d', ARROW_PATH);
  }

  /**
   * The pinned core API does not expose the grid, so derive the type from the
   * standard FrozenLake layout: the start is the first cell, the goal is the
   * last cell, and every other terminal cell is a hole.
   */
  private cellTypeAt(index: number, actions: readonly (FrozenLakeAction | null)[]): FrozenLakeCell {
    if (index === 0) {
      return 'start';
    }
    if (index === this.size * this.size - 1) {
      return 'goal';
    }
    return actions[index] === null ? 'hole' : 'frozen';
  }

  private draw(): void {
    const cells = this.cells;
    const arrows = this.arrows;
    const agent = this.agent;
    if (!cells || !arrows || !agent) {
      return;
    }
    const size = this.size;
    const cellSize = GRID_SIZE / size;
    const count = size * size;
    const actions = this.core.greedyActions;

    cells.attr('class', (index) => `drl-cell drl-cell--${this.cellTypeAt(index, actions)}`);

    const values = this.core.greedyValues;
    let maxValue = 0;
    for (let index = 0; index < count; index += 1) {
      const value = values[index] ?? 0;
      if (value > maxValue) {
        maxValue = value;
      }
    }

    arrows
      .attr('transform', (index) => {
        const cx = (index % size) * cellSize + cellSize / 2;
        const cy = Math.floor(index / size) * cellSize + cellSize / 2;
        const action = actions[index];
        const rotation = action === null ? 0 : ACTION_ROTATION[action];
        return `translate(${cx}, ${cy}) rotate(${rotation}) scale(${cellSize})`;
      })
      .attr('visibility', (index) => (actions[index] === null ? 'hidden' : 'visible'))
      .attr('opacity', (index) => {
        const value = values[index] ?? 0;
        return maxValue > 0 ? 0.15 + 0.85 * (value / maxValue) : 0.15;
      });

    const observation = this.core.observation;
    const cx = (observation % size) * cellSize + cellSize / 2;
    const cy = Math.floor(observation / size) * cellSize + cellSize / 2;
    agent
      .attr('cx', cx)
      .attr('cy', cy)
      .attr('r', cellSize / 12);

    if (this.statusText) {
      const { episodes, steps, successRate } = this.core.stats;
      this.statusText.textContent = `Épisodes ${episodes} · Réussite ${Math.round(successRate * 100)} % · Pas ${steps}`;
    }

    const curve = this.curve;
    if (curve) {
      const history = this.core.rewardHistory;
      if (history.length === 0) {
        curve.attr('visibility', 'hidden');
      } else {
        const length = history.length;
        const d = history
          .map((reward, index) => {
            const x = length === 1 ? 0 : (index / (length - 1)) * CURVE_WIDTH;
            const y = CURVE_HEIGHT - reward * CURVE_HEIGHT;
            return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
          })
          .join(' ');
        curve.attr('visibility', 'visible').attr('d', d);
      }
    }
  }

  protected override renderHeader(): TemplateResult {
    return html`
      <h2 class="drl-title">Frozen Lake — Q-learning tabulaire</h2>
      <p class="drl-lead">
        Un agent apprend à traverser un lac gelé sans tomber dans un trou. Il découvre
        la valeur de chaque paire case et déplacement par essais et erreurs. Epsilon
        choisit parfois un déplacement au hasard pour explorer. Les flèches montrent le
        meilleur déplacement pour chaque case.
      </p>
    `;
  }

  protected override renderContent(): TemplateResult {
    return html`
      <span class="drl-status-text"></span>
      <div class="drl-toolbar" role="group" aria-label="Contrôles de Frozen Lake">
        <button
          type="button"
          class="drl-button"
          aria-pressed=${this.size === 4 ? 'true' : 'false'}
          @click=${this.handleSize4}
        >
          4×4
        </button>
        <button
          type="button"
          class="drl-button"
          aria-pressed=${this.size === 8 ? 'true' : 'false'}
          @click=${this.handleSize8}
        >
          8×8
        </button>
        <button
          type="button"
          class="drl-button"
          aria-pressed=${this.slippery ? 'true' : 'false'}
          @click=${this.handleSlippery}
        >
          Glissant
        </button>
      </div>
      <svg class="drl-grid" viewBox="0 0 480 480" role="img" aria-label="Grille Frozen Lake">
        <g class="drl-cells"></g>
        <g class="drl-arrows"></g>
        <circle class="drl-agent" r="10"></circle>
      </svg>
      <svg
        class="drl-curve-svg"
        viewBox="0 0 480 90"
        role="img"
        aria-label="Courbe des récompenses"
      >
        <path class="drl-curve" visibility="hidden"></path>
      </svg>
    `;
  }
}
