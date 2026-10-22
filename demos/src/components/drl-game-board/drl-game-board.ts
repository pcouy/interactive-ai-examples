import type { SVGTemplateResult, TemplateResult } from 'lit';
import { css, html, nothing, svg } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { LocalizedElement } from '../../base/localized-element.js';
import type { BoardView, Cell, CellCoord } from '../../core/contracts.js';
import { drlTokens } from '../../styles/tokens.js';
import { gameBoardMessage } from './messages.js';

export type PieceStyle = 'discs' | 'marks';

export interface DrlCellClickDetail {
  readonly row: number;
  readonly col: number;
}

export type DrlCellClickEvent = CustomEvent<DrlCellClickDetail>;

declare global {
  interface HTMLElementEventMap {
    'drl-cell-click': DrlCellClickEvent;
  }
}

interface BoardCellData {
  readonly row: number;
  readonly col: number;
  readonly player: Cell;
  readonly clickable: boolean;
  readonly highlighted: boolean;
}

const MIN_CELL_SIZE = 4;
const MIN_STROKE = 2;
const MIN_HIGHLIGHT_INSET = 1.5;
const DISC_RADIUS_RATIO = 0.38;
const MARK_RADIUS_RATIO = 0.3;
const MARK_INSET_RATIO = 0.28;
const PIECE_STROKE_RATIO = 0.09;
const HIGHLIGHT_STROKE_RATIO = 0.06;
const HIGHLIGHT_INSET_RATIO = 0.05;

function cellKey(row: number, col: number): string {
  return `${row}:${col}`;
}

function coordKeys(coords: ReadonlyArray<CellCoord> | undefined): ReadonlySet<string> {
  const keys = new Set<string>();
  for (const coord of coords ?? []) {
    keys.add(cellKey(coord.row, coord.col));
  }
  return keys;
}

function clampCellSize(value: number): number {
  return Number.isFinite(value) && value > MIN_CELL_SIZE ? value : MIN_CELL_SIZE;
}

function pieceStroke(size: number): number {
  return Math.max(MIN_STROKE, size * PIECE_STROKE_RATIO);
}

@customElement('drl-game-board')
export class DrlGameBoard extends LocalizedElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: inline-block;
      }

      svg {
        display: block;
        max-width: 100%;
        height: auto;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
      }

      .drl-cell {
        fill: var(--drl-bg);
        stroke: var(--drl-border);
      }

      .drl-cross {
        fill: none;
        stroke: var(--drl-accent);
        stroke-linecap: round;
      }

      .drl-ring {
        fill: none;
        stroke: var(--drl-accent-2);
      }

      .drl-disc--0 {
        fill: var(--drl-accent);
      }

      .drl-disc--1 {
        fill: var(--drl-accent-2);
      }

      .drl-highlight {
        fill: none;
        stroke: var(--drl-accent);
      }

      .drl-hit {
        fill: none;
        pointer-events: all;
        cursor: pointer;
      }

      .drl-hit:hover {
        fill: var(--drl-accent);
        fill-opacity: 0.14;
      }
    `,
  ];

  @property({ attribute: false })
  view: BoardView | null = null;

  @property({ type: Number, attribute: 'cell-size' })
  cellSize = 48;

  @property({ type: String, attribute: 'piece-style' })
  pieceStyle: PieceStyle = 'discs';

  protected override render(): TemplateResult | typeof nothing {
    const view = this.view;
    if (!view) {
      return nothing;
    }
    const rows = Math.floor(view.rows);
    const cols = Math.floor(view.cols);
    if (!Number.isFinite(rows) || !Number.isFinite(cols) || rows <= 0 || cols <= 0) {
      return nothing;
    }
    const size = clampCellSize(this.cellSize);
    const width = cols * size;
    const height = rows * size;
    const clickable = coordKeys(view.clickable);
    const highlighted = coordKeys(view.highlight);
    const cells: BoardCellData[] = [];
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        cells.push({
          row,
          col,
          player: view.cells[row]?.[col] ?? null,
          clickable: clickable.has(cellKey(row, col)),
          highlighted: highlighted.has(cellKey(row, col)),
        });
      }
    }
    const radius = Math.min(12, Math.max(2, size * 0.16));
    return html`
      <svg
        role="img"
        aria-label=${gameBoardMessage(this.locale, 'ariaLabel', { rows, cols })}
        viewBox="0 0 ${width} ${height}"
        width=${width}
        height=${height}
        style="border-radius:${radius}px"
      >
        ${cells.map((data) => this.cellRect(data, size))}
        ${cells.map((data) => this.pieceNode(data, size))}
        ${cells.map((data) => this.highlightFrame(data, size))}
        ${cells.filter((data) => data.clickable).map((data) => this.hitRect(data, size))}
      </svg>
    `;
  }

  private cellRect(data: BoardCellData, size: number): SVGTemplateResult {
    return svg`<rect
      class="drl-cell"
      x=${data.col * size}
      y=${data.row * size}
      width=${size}
      height=${size}
    ></rect>`;
  }

  private pieceNode(data: BoardCellData, size: number): SVGTemplateResult | typeof nothing {
    if (data.player === null) {
      return nothing;
    }
    const cx = data.col * size + size / 2;
    const cy = data.row * size + size / 2;
    if (this.pieceStyle === 'marks') {
      const stroke = pieceStroke(size);
      if (data.player === 0) {
        const inset = size * MARK_INSET_RATIO;
        return svg`
          <line
            class="drl-cross"
            x1=${cx - inset}
            y1=${cy - inset}
            x2=${cx + inset}
            y2=${cy + inset}
            stroke-width=${stroke}
          ></line>
          <line
            class="drl-cross"
            x1=${cx + inset}
            y1=${cy - inset}
            x2=${cx - inset}
            y2=${cy + inset}
            stroke-width=${stroke}
          ></line>
        `;
      }
      return svg`<circle
        class="drl-ring"
        cx=${cx}
        cy=${cy}
        r=${size * MARK_RADIUS_RATIO}
        stroke-width=${stroke}
      ></circle>`;
    }
    return svg`<circle
      class="drl-disc drl-disc--${data.player === 0 ? '0' : '1'}"
      cx=${cx}
      cy=${cy}
      r=${size * DISC_RADIUS_RATIO}
    ></circle>`;
  }

  private highlightFrame(data: BoardCellData, size: number): SVGTemplateResult | typeof nothing {
    if (!data.highlighted) {
      return nothing;
    }
    const inset = Math.max(MIN_HIGHLIGHT_INSET, size * HIGHLIGHT_INSET_RATIO);
    const side = size - inset * 2;
    return svg`<rect
      class="drl-highlight"
      x=${data.col * size + inset}
      y=${data.row * size + inset}
      width=${side}
      height=${side}
      rx=${size * 0.1}
      stroke-width=${Math.max(MIN_STROKE, size * HIGHLIGHT_STROKE_RATIO)}
    ></rect>`;
  }

  private hitRect(data: BoardCellData, size: number): SVGTemplateResult {
    return svg`<rect
      class="drl-hit"
      x=${data.col * size}
      y=${data.row * size}
      width=${size}
      height=${size}
      @click=${() => this.emitCellClick(data.row, data.col)}
    ></rect>`;
  }

  private readonly emitCellClick = (row: number, col: number): void => {
    if (!this.view?.clickable?.some((coord) => coord.row === row && coord.col === col)) {
      return;
    }
    this.dispatchEvent(
      new CustomEvent<DrlCellClickDetail>('drl-cell-click', {
        detail: { row, col },
        bubbles: true,
        composed: true,
      }),
    );
  };
}
