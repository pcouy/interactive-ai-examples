import { describe, expect, it } from 'vitest';
import {
  argmaxIndex,
  brushStrength,
  filterKernelMean,
  formatLoss,
  formatPercent,
  formatSeconds,
  hexToRgb,
  mixRgb,
  paintDisc,
  pointerToCell,
  stratifiedExampleCountEstimate,
} from './drl-mnist-cnn.js';

function blankPixels(): Float32Array {
  return new Float32Array(784);
}

/** Toile simulée avec un rectangle fixe, pour tester `pointerToCell` sans DOM. */
function mockCanvas(width: number, height: number, left = 0, top = 0): HTMLCanvasElement {
  return {
    getBoundingClientRect: () => ({
      left,
      top,
      width,
      height,
      right: left + width,
      bottom: top + height,
      x: left,
      y: top,
      toJSON: () => ({}),
    }),
  } as unknown as HTMLCanvasElement;
}

function pointer(clientX: number, clientY: number): PointerEvent {
  return { clientX, clientY } as PointerEvent;
}

describe('brushStrength', () => {
  it('is 1 at the stroke center and fades linearly to 0 at the brush radius', () => {
    expect(brushStrength(0)).toBe(1);
    expect(brushStrength(0.8)).toBeCloseTo(0.5);
    expect(brushStrength(1.6)).toBe(0);
    expect(brushStrength(3)).toBe(0);
  });
});

describe('paintDisc', () => {
  it('paints a soft disc around the stroke point', () => {
    const pixels = blankPixels();
    paintDisc(pixels, 10.5, 10.5);
    expect(pixels[10 * 28 + 10]).toBeGreaterThan(0.9);
    expect(pixels[10 * 28 + 9]).toBeGreaterThan(0);
    expect(pixels[5 * 28 + 5]).toBe(0);
  });

  it('keeps the darkest value when strokes overlap', () => {
    const pixels = blankPixels();
    paintDisc(pixels, 5.5, 5.5);
    const first = pixels[5 * 28 + 5];
    paintDisc(pixels, 6.5, 6.5);
    expect(pixels[5 * 28 + 5]).toBe(first);
  });

  it('stays inside the 28×28 grid on edge strokes', () => {
    const pixels = blankPixels();
    paintDisc(pixels, 27.9, 0.1);
    expect(pixels[0 * 28 + 27]).toBeGreaterThan(0);
    expect(pixels[3 * 28 + 27]).toBe(0);
  });
});

describe('filterKernelMean', () => {
  it('returns the 3×3 kernel when the layer has one input channel', () => {
    // HWIO flat layout: 9 positions × 1 input channel × 2 output channels.
    const kernel = new Float32Array(18);
    for (let position = 0; position < 9; position += 1) {
      kernel[position * 2] = position + 1;
      kernel[position * 2 + 1] = 100;
    }
    expect(Array.from(filterKernelMean(kernel, 1, 0))).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(Array.from(filterKernelMean(kernel, 1, 1))).toEqual([
      100, 100, 100, 100, 100, 100, 100, 100, 100,
    ]);
  });

  it('averages the input channels for multi-channel kernels', () => {
    // HWIO flat layout: 9 positions × 3 input channels × 1 output channel;
    // channel ic contributes (ic + 1) × 10 + position.
    const kernel = new Float32Array(27);
    for (let position = 0; position < 9; position += 1) {
      for (let ic = 0; ic < 3; ic += 1) {
        kernel[position * 3 + ic] = (ic + 1) * 10 + position;
      }
    }
    const means = filterKernelMean(kernel, 3, 0);
    expect(means[0]).toBeCloseTo((10 + 20 + 30) / 3);
    expect(means[8]).toBeCloseTo((18 + 28 + 38) / 3);
  });
});

describe('argmaxIndex', () => {
  it('returns the index of the largest value', () => {
    expect(argmaxIndex(new Float32Array([0.1, 0.4, 0.3, 0.2]))).toBe(1);
  });

  it('returns the first index on ties', () => {
    expect(argmaxIndex(new Float32Array([0.5, 0.5, 0.1]))).toBe(0);
  });
});

describe('hexToRgb', () => {
  it('parses six-digit hex colors', () => {
    expect(hexToRgb('#2563eb')).toEqual([37, 99, 235]);
  });

  it('falls back to the accent blue on malformed input', () => {
    expect(hexToRgb('blue')).toEqual([37, 99, 235]);
  });
});

describe('mixRgb', () => {
  it('interpolates between two colors', () => {
    expect(mixRgb([255, 255, 255], [0, 0, 0], 0.5)).toEqual([128, 128, 128]);
    expect(mixRgb([255, 255, 255], [255, 0, 0], 0)).toEqual([255, 255, 255]);
  });
});

describe('formatPercent', () => {
  it('formats with one decimal and a French comma', () => {
    expect(formatPercent(12.34)).toBe('12,3 %');
    expect(formatPercent(0)).toBe('0,0 %');
    expect(formatPercent(99.96)).toBe('100,0 %');
  });
});

describe('pointerToCell', () => {
  it('maps the canvas corners to the 28×28 grid corners', () => {
    const canvas = mockCanvas(280, 280, 10, 20);
    expect(pointerToCell(canvas, pointer(10, 20))).toEqual([0, 0]);
    expect(pointerToCell(canvas, pointer(290, 300))).toEqual([28, 28]);
  });

  it('maps the canvas center to the grid center', () => {
    const canvas = mockCanvas(280, 280, 10, 20);
    expect(pointerToCell(canvas, pointer(150, 160))).toEqual([14, 14]);
  });

  it('returns null for a zero-size canvas', () => {
    expect(pointerToCell(mockCanvas(0, 0), pointer(5, 5))).toBeNull();
    expect(pointerToCell(mockCanvas(0, 100), pointer(5, 5))).toBeNull();
    expect(pointerToCell(mockCanvas(100, 0), pointer(5, 5))).toBeNull();
  });
});

describe('formatLoss', () => {
  it('formats with three decimals and a French comma', () => {
    expect(formatLoss(0.34567)).toBe('0,346');
    expect(formatLoss(0)).toBe('0,000');
  });

  it('renders a dash for a null or non-finite loss', () => {
    expect(formatLoss(null)).toBe('–');
    expect(formatLoss(Number.NaN)).toBe('–');
    expect(formatLoss(Number.POSITIVE_INFINITY)).toBe('–');
  });
});

describe('formatSeconds', () => {
  it('formats durations with one decimal and a French comma', () => {
    expect(formatSeconds(0)).toBe('0,0 s');
    expect(formatSeconds(1500)).toBe('1,5 s');
    expect(formatSeconds(12345)).toBe('12,3 s');
  });
});

describe('stratifiedExampleCountEstimate', () => {
  it('estimates one tenth of the official split sizes', () => {
    expect(stratifiedExampleCountEstimate(60000, 0.1)).toBe(6000);
    expect(stratifiedExampleCountEstimate(10000, 0.1)).toBe(1000);
  });

  it('keeps at least one example per class for tiny fractions', () => {
    expect(stratifiedExampleCountEstimate(60000, 0.001)).toBe(60);
    expect(stratifiedExampleCountEstimate(10000, 0.001)).toBe(10);
    expect(stratifiedExampleCountEstimate(10000, 0.01)).toBe(100);
  });

  it('returns the full size for a fraction of one', () => {
    expect(stratifiedExampleCountEstimate(60000, 1)).toBe(60000);
    expect(stratifiedExampleCountEstimate(10000, 1)).toBe(10000);
  });

  it('rejects invalid totals and fractions', () => {
    expect(() => stratifiedExampleCountEstimate(5, 0.1)).toThrow(RangeError);
    expect(() => stratifiedExampleCountEstimate(1.5, 0.1)).toThrow(RangeError);
    expect(() => stratifiedExampleCountEstimate(60000, 0)).toThrow(RangeError);
    expect(() => stratifiedExampleCountEstimate(60000, 1.5)).toThrow(RangeError);
    expect(() => stratifiedExampleCountEstimate(60000, Number.NaN)).toThrow(RangeError);
  });
});
