import type { BaseType, Selection } from 'd3';
import { select } from 'd3';
import type { PropertyValues, TemplateResult } from 'lit';
import { css, html, nothing } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { DemoElement } from '../../base/demo-element.js';
import type { FrozenLakeAction } from '../../core/frozen-lake.js';
import { FROZEN_LAKE_ACTIONS, FrozenLakeCore } from '../../core/frozen-lake.js';
import { drlTokens } from '../../styles/tokens.js';

const GRID_SIZE = 480;
const CURVE_WIDTH = 480;
const CURVE_HEIGHT = 90;
const ARROW_PATH =
  'M -0.30 -0.12 L 0.05 -0.12 L 0.05 -0.26 L 0.34 0 L 0.05 0.26 L 0.05 0.12 L -0.30 0.12 Z';

// Tooltip offset from the cursor, in pixels.
const TOOLTIP_OFFSET = 12;

const ACTION_ROTATION: Record<FrozenLakeAction, number> = {
  right: 0,
  down: 90,
  left: 180,
  up: 270,
};

const ACTION_GLYPHS: Record<FrozenLakeAction, string> = {
  left: '←',
  down: '↓',
  right: '→',
  up: '↑',
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

      .drl-grid-wrap {
        position: relative;
        margin-bottom: var(--drl-space-2);
      }

      .drl-tooltip {
        position: absolute;
        z-index: 2;
        padding: var(--drl-space-1) var(--drl-space-2);
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
        box-shadow: 0 1px 2px rgb(15 23 42 / 6%);
        color: var(--drl-fg);
        font-size: 0.85em;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        pointer-events: none;
      }

      .drl-tooltip[hidden] {
        display: none;
      }

      .drl-q-row {
        display: flex;
        align-items: baseline;
        gap: var(--drl-space-2);
      }

      .drl-q-action {
        display: inline-block;
        min-width: 1.2ch;
        text-align: center;
      }

      .drl-q-row--max {
        font-weight: 700;
        color: var(--drl-accent);
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

      .drl-cell--wall {
        fill: var(--drl-muted);
      }

      .drl-cell--candy {
        fill: var(--drl-accent-2);
      }

      .drl-arrow {
        fill: var(--drl-fg);
        pointer-events: none;
      }

      .drl-agent {
        fill: var(--drl-accent-2);
        stroke: var(--drl-surface);
        stroke-width: 2;
        pointer-events: none;
      }

      .drl-curve {
        fill: none;
        stroke: var(--drl-accent);
        stroke-width: 2;
      }

      .drl-replay {
        margin-bottom: var(--drl-space-2);
      }

      .drl-replay-header {
        margin: 0 0 var(--drl-space-1);
        color: var(--drl-muted);
        font-size: 0.85em;
        font-variant-numeric: tabular-nums;
      }

      .drl-replay-list {
        display: grid;
        gap: var(--drl-space-1);
        margin: 0;
        padding: 0;
        list-style: none;
      }

      .drl-replay-row {
        font-size: 0.9em;
        font-variant-numeric: tabular-nums;
      }

      .drl-replay-row--last {
        color: var(--drl-accent);
        font-weight: 600;
      }

      .drl-replay-row--muted {
        color: var(--drl-muted);
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

  @property({ type: Number, attribute: 'slip-probability' })
  slipProbability = 0;

  @property({ type: Number, attribute: 'candy-reward' })
  candyReward = 0.1;

  @property({ type: Number, attribute: 'initial-value' })
  initialValue = 0;

  @property({ type: Boolean, attribute: 'replay-enabled' })
  replayEnabled = false;

  @property({ type: Number, attribute: 'replay-batch-size' })
  replayBatchSize = 8;

  private readonly core = new FrozenLakeCore();

  private cellsGroup: SingleSelection<SVGGElement> | null = null;

  private arrowsGroup: SingleSelection<SVGGElement> | null = null;

  private cells: JoinedSelection<SVGRectElement, number> | null = null;

  private arrows: JoinedSelection<SVGPathElement, number> | null = null;

  private agent: SingleSelection<SVGCircleElement> | null = null;

  private curve: SingleSelection<SVGPathElement> | null = null;

  private statusText: HTMLElement | null = null;

  private tooltip: HTMLElement | null = null;

  private hoveredCell: number | null = null;

  private gridWrap: HTMLElement | null = null;

  private get size(): 4 | 6 | 8 {
    return this.gridSize === 8 ? 8 : this.gridSize === 6 ? 6 : 4;
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
    this.gridWrap = this.renderRoot.querySelector<HTMLElement>('.drl-grid-wrap');
    this.tooltip = this.renderRoot.querySelector<HTMLElement>('.drl-tooltip');
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
    this.core.initialValue = this.initialValue;
    this.core.candyReward = this.candyReward;
    this.core.replayEnabled = this.replayEnabled;
    this.core.replayBatchSize = this.replayBatchSize;
    if (changed.has('gridSize')) {
      this.core.setSize(this.size);
      this.buildGrid();
      this.draw();
    }
    if (changed.has('slipProbability')) {
      this.core.setSlipProbability(this.slipProbability);
      this.draw();
    }
  }

  private readonly handleSize4 = (): void => {
    this.gridSize = 4;
    this.requestUpdate();
  };

  private readonly handleSize6 = (): void => {
    this.gridSize = 6;
    this.requestUpdate();
  };

  private readonly handleSize8 = (): void => {
    this.gridSize = 8;
    this.requestUpdate();
  };

  private readonly handleReplayToggle = (): void => {
    this.replayEnabled = !this.replayEnabled;
    this.requestUpdate();
  };

  // The replay panel exists in the DOM only after the first toggle, so it is
  // refreshed once the new markup has rendered, not during willUpdate.
  protected override updated(changed: PropertyValues<this>): void {
    super.updated(changed);
    if (changed.has('replayEnabled')) {
      this.draw();
    }
  }

  private buildGrid(): void {
    // The grid is rebuilt when the grid size changes, so the tooltip must not
    // survive the rebuild.
    this.hideTooltip();
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
      .attr('y', (index) => Math.floor(index / size) * cellSize)
      .on('pointerenter', (event: PointerEvent, index: number) => {
        this.handleCellEnter(event, index);
      })
      .on('pointermove', (event: PointerEvent) => {
        this.handleCellMove(event);
      })
      .on('pointerleave', () => {
        this.hideTooltip();
      });

    this.arrows = arrowsGroup
      .selectAll<SVGPathElement, number>('path.drl-arrow')
      .data(indices, (index) => String(index))
      .join('path')
      .attr('class', 'drl-arrow')
      .attr('d', ARROW_PATH);
  }

  /**
   * Walls and terminal cells have no policy, so they get no tooltip either.
   * Navigable cells are the start, frozen and candy cells.
   */
  private handleCellEnter(event: PointerEvent, index: number): void {
    const type = this.core.cellTypeAt(index);
    if (type !== 'start' && type !== 'frozen' && type !== 'candy') {
      return;
    }
    this.hoveredCell = index;
    this.showTooltip(event, index);
  }

  private handleCellMove(event: PointerEvent): void {
    this.positionTooltip(event);
  }

  private hideTooltip(): void {
    this.hoveredCell = null;
    if (this.tooltip) {
      this.tooltip.hidden = true;
    }
  }

  private showTooltip(event: PointerEvent, index: number): void {
    const tooltip = this.tooltip;
    if (!tooltip) {
      return;
    }
    this.fillTooltip(index);
    // Unhide before positioning: a hidden tooltip has zero size, so the clamp
    // inside positionTooltip needs the rendered dimensions.
    tooltip.hidden = false;
    this.positionTooltip(event);
  }

  /** Rebuilds the tooltip content for `index` from the current Q values. */
  private fillTooltip(index: number): void {
    const tooltip = this.tooltip;
    if (!tooltip) {
      return;
    }
    const values = this.core.actionValues(index);
    const maxValue = Math.max(...values);
    const allEqual = values.every((value) => value === values[0]);
    tooltip.replaceChildren(
      ...FROZEN_LAKE_ACTIONS.map((action, position) => {
        const row = document.createElement('div');
        const isMax = !allEqual && values[position] === maxValue;
        row.className = `drl-q-row${isMax ? ' drl-q-row--max' : ''}`;
        const glyph = document.createElement('span');
        glyph.className = 'drl-q-action';
        glyph.textContent = ACTION_GLYPHS[action];
        const value = document.createElement('span');
        value.textContent = values[position].toFixed(2);
        row.append(glyph, value);
        return row;
      }),
    );
  }

  /** Places the tooltip at the cursor with a small offset, inside the grid wrap. */
  private positionTooltip(event: PointerEvent): void {
    const tooltip = this.tooltip;
    const wrap = this.gridWrap;
    if (!tooltip || !wrap) {
      return;
    }
    const bounds = wrap.getBoundingClientRect();
    const x = event.clientX - bounds.left + TOOLTIP_OFFSET;
    const y = event.clientY - bounds.top + TOOLTIP_OFFSET;
    const left = Math.min(Math.max(0, x), Math.max(0, bounds.width - tooltip.offsetWidth));
    const top = Math.min(Math.max(0, y), Math.max(0, bounds.height - tooltip.offsetHeight));
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
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

    cells.attr('class', (index) => `drl-cell drl-cell--${this.core.cellTypeAt(index)}`);

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
        const raw = maxValue > 0 ? 0.15 + 0.85 * (value / maxValue) : 0.15;
        return Math.min(Math.max(raw, 0.15), 1);
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
        // Candy rewards can push a return past 1, so scale by the observed
        // maximum instead of clipping the curve at the top edge. The floor of 1
        // keeps early episodes at full height.
        const maxReturn = Math.max(1, ...history);
        const d = history
          .map((reward, index) => {
            const x = length === 1 ? 0 : (index / (length - 1)) * CURVE_WIDTH;
            const y = CURVE_HEIGHT - (reward / maxReturn) * CURVE_HEIGHT;
            return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
          })
          .join(' ');
        curve.attr('visibility', 'visible').attr('d', d);
      }
    }

    // Refresh the tooltip while the pointer stays on a cell during training.
    if (this.hoveredCell !== null && this.tooltip && !this.tooltip.hidden) {
      this.fillTooltip(this.hoveredCell);
    }

    this.drawReplay();
  }

  /**
   * Fills the replay panel from the last sampled batch. The panel is queried
   * on every call because it exists in the DOM only while replay is enabled.
   * Rows are newest first, so the batch is reversed around its last entry.
   */
  private drawReplay(): void {
    const header = this.renderRoot.querySelector<HTMLElement>('.drl-replay-header');
    const list = this.renderRoot.querySelector<HTMLElement>('.drl-replay-list');
    if (!header || !list) {
      return;
    }
    const batch = this.core.lastReplayBatch;
    header.textContent = `Rejeu · échantillon ${batch.length} · mémoire ${this.core.replayBufferSize}/${this.core.replayCapacity}`;
    list.replaceChildren();
    if (batch.length === 0) {
      const placeholder = document.createElement('li');
      placeholder.className = 'drl-replay-row drl-replay-row--muted';
      placeholder.textContent = "Aucun échantillon pour l'instant.";
      list.append(placeholder);
      return;
    }
    const rows = [batch[batch.length - 1], ...batch.slice(0, -1)];
    rows.forEach((transition, position) => {
      const row = document.createElement('li');
      row.className = position === 0 ? 'drl-replay-row drl-replay-row--last' : 'drl-replay-row';
      let text = `case ${transition.observation} ${ACTION_GLYPHS[transition.action]} case ${transition.nextObservation} · r ${transition.reward}`;
      if (transition.done) {
        text += ' · fin';
      }
      if (position === 0) {
        text += ' · dernier pas';
      }
      row.textContent = text;
      list.append(row);
    });
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
          aria-pressed=${this.size === 6 ? 'true' : 'false'}
          @click=${this.handleSize6}
        >
          6×6
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
          aria-pressed=${this.replayEnabled ? 'true' : 'false'}
          @click=${this.handleReplayToggle}
        >
          Rejeu
        </button>
      </div>
      <div class="drl-grid-wrap">
        <svg class="drl-grid" viewBox="0 0 480 480" role="img" aria-label="Grille Frozen Lake">
          <g class="drl-cells"></g>
          <g class="drl-arrows"></g>
          <circle class="drl-agent" r="10"></circle>
        </svg>
        <div class="drl-tooltip" hidden></div>
      </div>
      <svg
        class="drl-curve-svg"
        viewBox="0 0 480 90"
        role="img"
        aria-label="Courbe des récompenses"
      >
        <path class="drl-curve" visibility="hidden"></path>
      </svg>
      ${
        this.replayEnabled
          ? html`
            <section class="drl-replay" aria-label="Rejeu">
              <p class="drl-replay-header"></p>
              <ul class="drl-replay-list"></ul>
            </section>
          `
          : nothing
      }
    `;
  }
}
