import { describe, expect, it } from 'vitest';
import type { MnistCnnFilters } from '../../core/mnist-cnn.js';
import {
  enumerateLinks,
  filterCanvasSelector,
  isUsableRect,
  linkGeometry,
  type RectLike,
} from './filter-links.js';

describe('enumerateLinks', () => {
  it('enumerates three boundaries of nine for the default architecture', () => {
    const links = enumerateLinks([3, 3, 3, 3]);
    expect(links).toHaveLength(27);
    const boundaries = links.map((link) => link.boundary);
    expect(boundaries.slice(0, 9)).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1]);
    expect(boundaries.slice(9, 18)).toEqual([2, 2, 2, 2, 2, 2, 2, 2, 2]);
    expect(boundaries.slice(18)).toEqual([3, 3, 3, 3, 3, 3, 3, 3, 3]);
  });

  it('orders by boundary, then input channel, then output channel', () => {
    const links = enumerateLinks([3, 3, 3, 3]);
    expect(links[0]).toEqual({ boundary: 1, inputChannel: 0, outputChannel: 0 });
    expect(links[1]).toEqual({ boundary: 1, inputChannel: 0, outputChannel: 1 });
    expect(links[2]).toEqual({ boundary: 1, inputChannel: 0, outputChannel: 2 });
    expect(links[3]).toEqual({ boundary: 1, inputChannel: 1, outputChannel: 0 });
    expect(links[8]).toEqual({ boundary: 1, inputChannel: 2, outputChannel: 2 });
    expect(links[26]).toEqual({ boundary: 3, inputChannel: 2, outputChannel: 2 });
  });

  it('counts each boundary as the product of the two channel counts', () => {
    const links = enumerateLinks([1, 2, 3, 4]);
    expect(links).toHaveLength(20);
    expect(links.filter((link) => link.boundary === 1)).toHaveLength(2);
    expect(links.filter((link) => link.boundary === 2)).toHaveLength(6);
    expect(links.filter((link) => link.boundary === 3)).toHaveLength(12);
    expect(links[0]).toEqual({ boundary: 1, inputChannel: 0, outputChannel: 0 });
    expect(links[19]).toEqual({ boundary: 3, inputChannel: 2, outputChannel: 3 });
  });

  it('rejects an invalid filter tuple', () => {
    expect(() => enumerateLinks([0, 3, 3, 3] as MnistCnnFilters)).toThrow(RangeError);
    expect(() => enumerateLinks([3, 3, 3, 5] as MnistCnnFilters)).toThrow(RangeError);
    expect(() => enumerateLinks([3, 3, 3] as unknown as MnistCnnFilters)).toThrow(RangeError);
  });
});

describe('isUsableRect', () => {
  it('accepts a normal rectangle', () => {
    expect(isUsableRect({ left: 4, top: 8, width: 40, height: 20 })).toBe(true);
  });

  it('rejects a zero, negative, or non-finite size', () => {
    expect(isUsableRect({ left: 0, top: 0, width: 0, height: 20 })).toBe(false);
    expect(isUsableRect({ left: 0, top: 0, width: 40, height: 0 })).toBe(false);
    expect(isUsableRect({ left: 0, top: 0, width: -40, height: 20 })).toBe(false);
    expect(isUsableRect({ left: 0, top: 0, width: 40, height: -20 })).toBe(false);
    expect(isUsableRect({ left: Number.NaN, top: 0, width: 40, height: 20 })).toBe(false);
    expect(isUsableRect({ left: 0, top: 0, width: Number.POSITIVE_INFINITY, height: 20 })).toBe(
      false,
    );
    expect(isUsableRect({ left: 0, top: Number.NEGATIVE_INFINITY, width: 40, height: 20 })).toBe(
      false,
    );
  });

  it('rejects a rectangle whose far corner overflows to infinity', () => {
    expect(
      isUsableRect({ left: Number.MAX_VALUE, top: 0, width: Number.MAX_VALUE, height: 20 }),
    ).toBe(false);
    expect(
      isUsableRect({ left: 0, top: Number.MAX_VALUE, width: 40, height: Number.MAX_VALUE }),
    ).toBe(false);
  });
});

describe('linkGeometry', () => {
  const CONTAINER: RectLike = { left: 5, top: 7, width: 500, height: 400 };

  it('draws from the bottom-centre of the source to the top-centre of the target', () => {
    const source: RectLike = { left: 10, top: 20, width: 40, height: 30 };
    const target: RectLike = { left: 100, top: 200, width: 20, height: 10 };
    expect(linkGeometry(source, target, CONTAINER)).toEqual({
      x1: 25,
      y1: 43,
      x2: 105,
      y2: 193,
    });
  });

  it('centres each endpoint on its own rectangle when the widths differ', () => {
    const wide: RectLike = { left: 0, top: 0, width: 100, height: 10 };
    const narrow: RectLike = { left: 200, top: 100, width: 20, height: 10 };
    // Source plus large que la cible : chaque extrémité suit le centre de son propre rectangle.
    const fromWide = linkGeometry(wide, narrow, CONTAINER);
    expect(fromWide.x1).toBe(45);
    expect(fromWide.x2).toBe(205);
    expect(fromWide.y1).toBe(3);
    expect(fromWide.y2).toBe(93);
    // Cible plus large que la source : les mêmes centres, dans l'autre sens.
    const fromNarrow = linkGeometry(narrow, wide, CONTAINER);
    expect(fromNarrow.x1).toBe(205);
    expect(fromNarrow.x2).toBe(45);
    expect(fromNarrow.y1).toBe(103);
    expect(fromNarrow.y2).toBe(-7);
  });

  it('draws between the centres of two one-pixel rectangles', () => {
    const source: RectLike = { left: 10, top: 20, width: 1, height: 1 };
    const target: RectLike = { left: 10, top: 30, width: 1, height: 1 };
    expect(linkGeometry(source, target, CONTAINER)).toEqual({
      x1: 5.5,
      y1: 14,
      x2: 5.5,
      y2: 23,
    });
  });

  it('throws a RangeError for an unusable container', () => {
    const rect: RectLike = { left: 0, top: 0, width: 10, height: 10 };
    expect(() => linkGeometry(rect, rect, { left: 0, top: 0, width: 0, height: 10 })).toThrow(
      RangeError,
    );
    expect(() =>
      linkGeometry(rect, rect, { left: Number.NaN, top: 0, width: 10, height: 10 }),
    ).toThrow(RangeError);
  });

  it('throws a RangeError for an unusable source or target', () => {
    const rect: RectLike = { left: 0, top: 0, width: 10, height: 10 };
    // Source et cible de taille nulle.
    expect(() => linkGeometry({ ...rect, width: 0 }, rect, CONTAINER)).toThrow(RangeError);
    expect(() => linkGeometry(rect, { ...rect, height: 0 }, CONTAINER)).toThrow(RangeError);
    // Champ non fini dans la source puis dans la cible.
    expect(() => linkGeometry({ ...rect, left: Number.NaN }, rect, CONTAINER)).toThrow(RangeError);
    expect(() => linkGeometry(rect, { ...rect, top: Number.NaN }, CONTAINER)).toThrow(RangeError);
    // Le message nomme le rectangle fautif.
    expect(() => linkGeometry({ ...rect, width: 0 }, rect, CONTAINER)).toThrow(/source/);
    expect(() => linkGeometry(rect, { ...rect, width: 0 }, CONTAINER)).toThrow(/target/);
  });
});

describe('filterCanvasSelector', () => {
  it('builds the data-attribute selector for a layer and channel', () => {
    expect(filterCanvasSelector(1, 2)).toBe(
      'canvas.drl-canvas--filter[data-layer="1"][data-channel="2"]',
    );
    expect(filterCanvasSelector(0, 0)).toBe(
      'canvas.drl-canvas--filter[data-layer="0"][data-channel="0"]',
    );
  });

  it('rejects a negative layer', () => {
    expect(() => filterCanvasSelector(-1, 0)).toThrow(RangeError);
  });

  it('rejects a negative channel', () => {
    expect(() => filterCanvasSelector(0, -1)).toThrow(RangeError);
  });
});
