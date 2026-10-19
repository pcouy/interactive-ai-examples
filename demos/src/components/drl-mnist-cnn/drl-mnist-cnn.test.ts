import { describe, expect, it } from 'vitest';
import {
  argmaxIndex,
  brushStrength,
  filterKernelMean,
  formatPercent,
  hexToRgb,
  mixRgb,
  paintDisc,
} from './drl-mnist-cnn.js';

function blankPixels(): Float32Array {
  return new Float32Array(784);
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
