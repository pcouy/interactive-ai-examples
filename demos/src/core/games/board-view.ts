import type { BoardView, Cell, CellCoord } from '../contracts.js';

export interface BoardViewOptions {
  readonly highlight?: ReadonlyArray<CellCoord>;
  readonly clickable?: ReadonlyArray<CellCoord>;
}

export function makeBoardView(
  rows: number,
  cols: number,
  cells: ReadonlyArray<readonly Cell[]>,
  options: BoardViewOptions = {},
): BoardView {
  return {
    rows,
    cols,
    cells: cells.map((row) => [...row]),
    ...(options.highlight === undefined
      ? {}
      : { highlight: options.highlight.map((c) => ({ ...c })) }),
    ...(options.clickable === undefined
      ? {}
      : { clickable: options.clickable.map((c) => ({ ...c })) }),
  };
}
