import type { BaseType, Selection } from 'd3';
import { select } from 'd3';
import type { PropertyValues, TemplateResult } from 'lit';
import { css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { DemoElement } from '../../base/demo-element.js';
import type { FineStepPhase, FrozenLakeAction, FrozenLakeSize } from '../../core/frozen-lake.js';
import { FINE_STEP_PHASES, FROZEN_LAKE_ACTIONS, FrozenLakeCore } from '../../core/frozen-lake.js';
import { drlTokens } from '../../styles/tokens.js';
import {
  ACTION_GLYPHS,
  FROZEN_LAKE_REPLAY_EVENT,
  type FrozenLakeReplayState,
  schematicView,
  updateView,
} from './frozen-lake-ui.js';

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

const PHASE_LABELS: Record<FineStepPhase, string> = {
  observe: 'Observation',
  act: 'Action',
  transition: 'Transition',
};

// Shown for a schematic field that has no value yet.
const EMPTY_VALUE = '—';

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

      .drl-layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 22rem);
        gap: var(--drl-space-3);
        align-items: start;
      }

      @media (max-width: 900px) {
        .drl-layout {
          grid-template-columns: minmax(0, 1fr);
        }
      }

      .drl-main,
      .drl-side {
        min-width: 0;
      }

      .drl-side {
        display: grid;
        gap: var(--drl-space-3);
      }

      .drl-panel {
        min-width: 0;
        padding: var(--drl-space-3);
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
      }

      .drl-panel-title {
        margin: 0 0 var(--drl-space-2);
        font-size: 1em;
      }

      .drl-panel-text {
        margin: 0 0 var(--drl-space-2);
        color: var(--drl-muted);
        font-size: 0.9em;
      }

      .drl-panel-note {
        margin: var(--drl-space-2) 0 0;
        color: var(--drl-muted);
        font-size: 0.85em;
      }

      .drl-loop {
        display: grid;
        gap: var(--drl-space-2);
      }

      .drl-loop-boxes {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: var(--drl-space-2);
      }

      .drl-loop-box {
        padding: var(--drl-space-2);
        background: var(--drl-bg);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
      }

      .drl-loop-name {
        display: block;
        margin-bottom: var(--drl-space-1);
        font-weight: 700;
      }

      .drl-loop-facts {
        display: grid;
        gap: var(--drl-space-1);
        margin: 0;
      }

      .drl-loop-fact {
        display: flex;
        justify-content: space-between;
        gap: var(--drl-space-2);
      }

      .drl-loop-fact dt {
        color: var(--drl-muted);
      }

      .drl-loop-fact dd {
        margin: 0;
        font-variant-numeric: tabular-nums;
      }

      .drl-loop-links {
        display: grid;
        gap: 2px;
        margin: 0;
        padding: 0;
        list-style: none;
        color: var(--drl-muted);
        font-size: 0.85em;
      }

      .drl-phases {
        display: flex;
        flex-wrap: wrap;
        margin: var(--drl-space-2) 0 0;
        padding: 0;
        list-style: none;
        font-size: 0.85em;
      }

      .drl-loop-action {
        justify-self: start;
      }

      .drl-phase {
        padding: 2px var(--drl-space-2);
        border: 1px solid var(--drl-border);
        color: var(--drl-muted);
      }

      .drl-phase + .drl-phase {
        margin-left: -1px;
      }

      .drl-phase:first-child {
        border-radius: calc(var(--drl-radius) / 2) 0 0 calc(var(--drl-radius) / 2);
      }

      .drl-phase:last-child {
        border-radius: 0 calc(var(--drl-radius) / 2) calc(var(--drl-radius) / 2) 0;
      }

      .drl-phase[aria-current='step'] {
        border-color: var(--drl-accent);
        color: var(--drl-accent);
        font-weight: 700;
      }

      .drl-formula-line {
        margin: 0 0 var(--drl-space-2);
        padding: var(--drl-space-2);
        background: var(--drl-bg);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        font-family: ui-monospace, monospace;
        font-size: 0.85em;
        overflow-wrap: anywhere;
      }

      .drl-update-grid {
        display: grid;
        gap: var(--drl-space-1);
        margin: 0 0 var(--drl-space-2);
      }

      .drl-update-row {
        display: flex;
        justify-content: space-between;
        gap: var(--drl-space-2);
      }

      .drl-update-row dt {
        color: var(--drl-muted);
      }

      .drl-update-row dd {
        margin: 0;
        font-variant-numeric: tabular-nums;
      }

      .drl-formula {
        margin: var(--drl-space-1) 0 0;
        font-family: ui-monospace, monospace;
        font-size: 0.85em;
        overflow-wrap: anywhere;
      }

      .drl-qtable {
        display: grid;
        gap: var(--drl-space-1);
        margin-top: var(--drl-space-2);
      }

      .drl-qtable-label {
        color: var(--drl-muted);
        font-size: 0.85em;
      }

      .drl-qtable-cells {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: var(--drl-space-1);
      }

      .drl-qtable-cell {
        display: grid;
        justify-items: center;
        gap: 2px;
        padding: var(--drl-space-1);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
      }

      .drl-qtable-cell--active {
        border-color: var(--drl-accent);
        color: var(--drl-accent);
        font-weight: 700;
      }

      .drl-qtable-glyph {
        font-size: 0.85em;
      }

      .drl-qtable-value {
        font-size: 0.85em;
        font-variant-numeric: tabular-nums;
      }

      .drl-update-placeholder {
        margin: 0;
        color: var(--drl-muted);
      }

      .drl-update-placeholder[hidden],
      .drl-update-content[hidden] {
        display: none;
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
    if (changed.has('initialValue')) {
      this.draw();
    }
    this.core.candyReward = this.candyReward;
    this.core.replayEnabled = this.replayEnabled;
    this.core.replayBatchSize = this.replayBatchSize;
    if (changed.has('gridSize')) {
      if (this.gridSize !== 4 && this.gridSize !== 6 && this.gridSize !== 8) {
        console.warn(
          `drl-frozen-lake: unsupported grid size ${this.gridSize}, using ${this.size}.`,
        );
        // Keep the property in sync with the coerced size that is rendered.
        this.gridSize = this.size;
      }
      this.core.setSize(this.size);
      this.buildGrid();
      this.draw();
    }
    if (changed.has('slipProbability')) {
      this.core.slipProbability = this.slipProbability;
      this.draw();
    }
  }

  private selectSize(size: FrozenLakeSize): void {
    this.gridSize = size;
    this.requestUpdate();
  }

  private readonly handleReplayToggle = (): void => {
    this.replayEnabled = !this.replayEnabled;
    this.requestUpdate();
  };

  // Advances exactly one phase of the MDP loop. Autoplay must stop first, so a
  // frame between the click and the redraw cannot consume the next phase.
  private readonly handleFineStep = (): void => {
    if (this.running) {
      this.running = false;
    }
    this.core.advanceFineStep();
    this.draw();
  };

  // The replay event carries the current batch, so it is refreshed once the
  // replay toggle has rendered and the core reflects the new setting.
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

  /**
   * Places the tooltip at the cursor with a small offset, inside the grid wrap.
   * This is generic grid logic: extract it when a second grid demo needs it.
   */
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
    this.drawGrid();
    this.drawStatus();
    this.drawCurve();
    // Refresh the tooltip while the pointer stays on a cell during training.
    if (this.hoveredCell !== null && this.tooltip && !this.tooltip.hidden) {
      this.fillTooltip(this.hoveredCell);
    }
    this.drawSchematic();
    this.drawUpdate();
    this.dispatchReplayState();
  }

  /** Draws the cells, the policy arrows and the agent on the grid. */
  private drawGrid(): void {
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
  }

  private drawStatus(): void {
    if (!this.statusText) {
      return;
    }
    const { episodes, steps, successRate } = this.core.stats;
    this.statusText.textContent = `Épisodes ${episodes} · Réussite ${Math.round(successRate * 100)} % · Pas ${steps}`;
  }

  private drawCurve(): void {
    const curve = this.curve;
    if (!curve) {
      return;
    }
    const history = this.core.rewardHistory;
    if (history.length === 0) {
      curve.attr('visibility', 'hidden');
      return;
    }
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

  /**
   * Fills the agent-environment schematic from the fine-step state. The observe
   * phase shows the current observation only.
   */
  private drawSchematic(): void {
    const state = this.core.fineStepState;
    const view = schematicView(state);
    const actionText =
      view.action === null ? EMPTY_VALUE : `${ACTION_GLYPHS[view.action]} ${view.action}`;

    this.setField(
      'schematic-observation',
      view.observation === null ? EMPTY_VALUE : String(view.observation),
    );
    this.setField('schematic-action', actionText);
    this.setField(
      'schematic-next',
      view.nextObservation === null ? EMPTY_VALUE : String(view.nextObservation),
    );
    this.setField('schematic-reward', view.reward === null ? EMPTY_VALUE : view.reward.toFixed(3));

    for (const phase of FINE_STEP_PHASES) {
      const node = this.renderRoot.querySelector<HTMLElement>(`[data-phase="${phase}"]`);
      if (node === null) {
        continue;
      }
      if (phase === state.phase) {
        node.setAttribute('aria-current', 'step');
      } else {
        node.removeAttribute('aria-current');
      }
    }
  }

  /**
   * Fills the Q update panel from the last update. The panel nodes are queried
   * on every call. The placeholder replaces the body until the first update.
   */
  private drawUpdate(): void {
    const update = this.core.lastUpdate;
    const placeholder = this.renderRoot.querySelector<HTMLElement>(
      '[data-field="update-placeholder"]',
    );
    const content = this.renderRoot.querySelector<HTMLElement>('[data-field="update-content"]');
    if (!placeholder || !content) {
      return;
    }
    if (update === null) {
      placeholder.hidden = false;
      content.hidden = true;
      return;
    }
    placeholder.hidden = true;
    content.hidden = false;

    const view = updateView(update);
    this.setField('reward', view.reward);
    this.setField('gamma', view.gamma);
    this.setField('next-value', view.nextValue);
    this.setField('bootstrap', view.bootstrap);
    this.setField('target', view.target);
    this.setField('previous', view.previous);
    this.setField('alpha', view.alpha);
    this.setField('updated', view.updated);
    this.setField('target-formula', view.targetFormula);
    this.setField('update-formula', view.updateFormula);

    const transition = update.transition;
    const glyph = ACTION_GLYPHS[transition.action];
    this.setField('qtable-label', `Case ${transition.observation} · ${glyph} ${transition.action}`);
    const values = this.core.actionValues(transition.observation);
    FROZEN_LAKE_ACTIONS.forEach((action, index) => {
      const cell = this.renderRoot.querySelector<HTMLElement>(`[data-qtable-action="${action}"]`);
      if (cell === null) {
        return;
      }
      cell.classList.toggle('drl-qtable-cell--active', action === transition.action);
      const cellGlyph = cell.querySelector<HTMLElement>('.drl-qtable-glyph');
      const cellValue = cell.querySelector<HTMLElement>('.drl-qtable-value');
      if (cellGlyph) {
        cellGlyph.textContent = ACTION_GLYPHS[action];
      }
      if (cellValue) {
        cellValue.textContent = values[index].toFixed(3);
      }
    });
  }

  /** Sets the text of the first `[data-field="name"]` node in the panel. */
  private setField(name: string, value: string): void {
    const node = this.renderRoot.querySelector<HTMLElement>(`[data-field="${name}"]`);
    if (node) {
      node.textContent = value;
    }
  }

  /** Publishes the replay state so the page can render the replay panel. */
  private dispatchReplayState(): void {
    this.dispatchEvent(
      new CustomEvent<FrozenLakeReplayState>(FROZEN_LAKE_REPLAY_EVENT, {
        detail: {
          enabled: this.replayEnabled,
          batch: [...this.core.lastReplayBatch],
          bufferSize: this.core.replayBufferSize,
          capacity: this.core.replayCapacity,
        },
        bubbles: true,
        composed: true,
      }),
    );
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
          @click=${() => this.selectSize(4)}
        >
          4×4
        </button>
        <button
          type="button"
          class="drl-button"
          aria-pressed=${this.size === 6 ? 'true' : 'false'}
          @click=${() => this.selectSize(6)}
        >
          6×6
        </button>
        <button
          type="button"
          class="drl-button"
          aria-pressed=${this.size === 8 ? 'true' : 'false'}
          @click=${() => this.selectSize(8)}
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
      <div class="drl-layout">
        <div class="drl-main">
          <div class="drl-grid-wrap">
            <svg
              class="drl-grid"
              viewBox="0 0 ${GRID_SIZE} ${GRID_SIZE}"
              role="img"
              aria-label="Grille Frozen Lake"
            >
              <g class="drl-cells"></g>
              <g class="drl-arrows"></g>
              <circle class="drl-agent" r="10"></circle>
            </svg>
            <div class="drl-tooltip" hidden></div>
          </div>
          <svg
            class="drl-curve-svg"
            viewBox="0 0 ${CURVE_WIDTH} ${CURVE_HEIGHT}"
            role="img"
            aria-label="Courbe des récompenses"
          >
            <path class="drl-curve" visibility="hidden"></path>
          </svg>
          <slot></slot>
        </div>
        <aside class="drl-side">
          <section class="drl-panel" aria-labelledby="drl-schematic-title">
            <h3 class="drl-panel-title" id="drl-schematic-title">
              Boucle agent-environnement
            </h3>
            <p class="drl-panel-text">
              L'environnement émet une observation. L'agent choisit une action.
              L'environnement renvoie une observation et une récompense.
            </p>
            <div class="drl-loop">
              <div class="drl-loop-boxes">
                <div class="drl-loop-box">
                  <span class="drl-loop-name">Environnement</span>
                  <dl class="drl-loop-facts">
                    <div class="drl-loop-fact">
                      <dt>Observation s</dt>
                      <dd data-field="schematic-observation">—</dd>
                    </div>
                    <div class="drl-loop-fact">
                      <dt>Observation s'</dt>
                      <dd data-field="schematic-next">—</dd>
                    </div>
                    <div class="drl-loop-fact">
                      <dt>Récompense r</dt>
                      <dd data-field="schematic-reward">—</dd>
                    </div>
                  </dl>
                </div>
                <div class="drl-loop-box">
                  <span class="drl-loop-name">Agent</span>
                  <dl class="drl-loop-facts">
                    <div class="drl-loop-fact">
                      <dt>Action a</dt>
                      <dd data-field="schematic-action">—</dd>
                    </div>
                  </dl>
                </div>
              </div>
              <ul class="drl-loop-links">
                <li class="drl-loop-link">observation s : Environnement → Agent</li>
                <li class="drl-loop-link">action a : Agent → Environnement</li>
                <li class="drl-loop-link">
                  observation s' et récompense r : Environnement → Agent
                </li>
              </ul>
              <ol class="drl-phases" aria-label="Phases du pas">
                ${FINE_STEP_PHASES.map(
                  (phase) => html`
                    <li class="drl-phase" data-phase=${phase}>${PHASE_LABELS[phase]}</li>
                  `,
                )}
              </ol>
              <button
                type="button"
                class="drl-button drl-loop-action"
                title="Avance d'une phase et met la lecture automatique en pause."
                @click=${this.handleFineStep}
              >
                Phase suivante
              </button>
            </div>
          </section>
          <section class="drl-panel" aria-labelledby="drl-update-title">
            <h3 class="drl-panel-title" id="drl-update-title">Dernière mise à jour</h3>
            <p class="drl-formula-line">
              Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') − Q(s, a) ]
            </p>
            <p class="drl-update-placeholder" data-field="update-placeholder">
              Aucune mise à jour pour l'instant.
            </p>
            <div class="drl-update-content" data-field="update-content" hidden>
              <dl class="drl-update-grid">
                <div class="drl-update-row">
                  <dt>Récompense r</dt>
                  <dd data-field="reward"></dd>
                </div>
                <div class="drl-update-row">
                  <dt>Facteur γ</dt>
                  <dd data-field="gamma"></dd>
                </div>
                <div class="drl-update-row">
                  <dt>Valeur suivante max Q(s', a')</dt>
                  <dd data-field="next-value"></dd>
                </div>
                <div class="drl-update-row">
                  <dt>Amorçage γ · max Q(s', a')</dt>
                  <dd data-field="bootstrap"></dd>
                </div>
                <div class="drl-update-row">
                  <dt>Cible</dt>
                  <dd data-field="target"></dd>
                </div>
                <div class="drl-update-row">
                  <dt>Valeur précédente Q(s, a)</dt>
                  <dd data-field="previous"></dd>
                </div>
                <div class="drl-update-row">
                  <dt>Taux α</dt>
                  <dd data-field="alpha"></dd>
                </div>
                <div class="drl-update-row">
                  <dt>Valeur mise à jour</dt>
                  <dd data-field="updated"></dd>
                </div>
              </dl>
              <p class="drl-formula" data-field="target-formula"></p>
              <p class="drl-formula" data-field="update-formula"></p>
              <div class="drl-qtable">
                <span class="drl-qtable-label" data-field="qtable-label"></span>
                <div class="drl-qtable-cells">
                  ${FROZEN_LAKE_ACTIONS.map(
                    (action) => html`
                      <span class="drl-qtable-cell" data-qtable-action=${action}>
                        <span class="drl-qtable-glyph"></span>
                        <span class="drl-qtable-value"></span>
                      </span>
                    `,
                  )}
                </div>
              </div>
            </div>
            <p class="drl-panel-note">
              La cible utilise la valeur de l'état suivant. La mise à jour écrit la
              nouvelle valeur dans la table apprise.
            </p>
          </section>
        </aside>
      </div>
    `;
  }
}
