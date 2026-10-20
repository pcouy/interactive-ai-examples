import { describe, expect, it } from 'vitest';
import {
  applyHeightScale,
  cameraRadius,
  clampFiniteHeight,
  DEFAULT_HEIGHT_SCALE,
  easeSmoothstep,
  gridIndex,
  heightRangeFromSamples,
  normalizeHeight,
  sampleGridHeights,
  sampleHeightRange,
  sanitizeHeightScale,
} from './landscape-scene.js';

describe('gridIndex', () => {
  it('emits res·res·2·3 indices within the vertex range', () => {
    const indices = gridIndex(70);
    expect(indices).toHaveLength(70 * 70 * 2 * 3);
    const vertexCount = 71 * 71;
    for (const index of indices) {
      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThan(vertexCount);
    }
  });

  it('emits no degenerate triangles', () => {
    const indices = gridIndex(70);
    for (let t = 0; t < indices.length; t += 3) {
      const a = indices[t];
      const b = indices[t + 1];
      const c = indices[t + 2];
      expect(a === b && b === c).toBe(false);
    }
  });

  it('is deterministic', () => {
    expect(gridIndex(70)).toEqual(gridIndex(70));
  });

  it('matches the single-quad layout for res=1', () => {
    expect(gridIndex(1)).toEqual([0, 2, 1, 1, 2, 3]);
  });
});

describe('sampleHeightRange', () => {
  it('returns span 0 for a constant field', () => {
    const { min, span } = sampleHeightRange(() => 1.5, 70, 5);
    expect(min).toBe(1.5);
    expect(span).toBe(0);
  });

  it('samples min and span over the grid for a general field', () => {
    const { min, span } = sampleHeightRange((w1, w2) => w1 * w1 + w2 * w2, 70, 1);
    expect(min).toBeCloseTo(0, 10);
    expect(span).toBeCloseTo(2, 10);
  });

  it('scales with the domain', () => {
    const { min, span } = sampleHeightRange((w1) => w1, 70, 2);
    expect(min).toBeCloseTo(-2, 10);
    expect(span).toBeCloseTo(4, 10);
  });

  it('reports a flat range for a field that is Infinity everywhere', () => {
    const { min, span } = sampleHeightRange(() => Number.POSITIVE_INFINITY, 70, 2);
    expect(min).toBe(0);
    expect(span).toBe(0);
  });

  it('reports a flat range for a field that is NaN everywhere', () => {
    const { min, span } = sampleHeightRange(() => Number.NaN, 70, 2);
    expect(min).toBe(0);
    expect(span).toBe(0);
  });

  it('skips non-finite samples and keeps the finite ones', () => {
    const { min, span } = sampleHeightRange(
      (w1) => (w1 > 0 ? Number.POSITIVE_INFINITY : w1),
      70,
      2,
    );
    expect(min).toBeCloseTo(-2, 10);
    expect(span).toBeCloseTo(2, 10);
  });

  it('flattens a finite field whose span overflows', () => {
    const range = sampleHeightRange((w1: number) => 1e308 * w1, 70, 2);
    expect(range).toEqual({ min: 0, span: 0 });
  });
});

describe('clampFiniteHeight', () => {
  it('clamps Infinity to the top of the range', () => {
    expect(clampFiniteHeight(Number.POSITIVE_INFINITY, 3, 4)).toBe(7);
  });

  it('clamps -Infinity to the bottom of the range', () => {
    expect(clampFiniteHeight(Number.NEGATIVE_INFINITY, 3, 4)).toBe(3);
  });

  it('clamps NaN to the bottom of the range', () => {
    expect(clampFiniteHeight(Number.NaN, 3, 4)).toBe(3);
  });

  it('passes finite values through unchanged', () => {
    expect(clampFiniteHeight(5, 3, 4)).toBe(5);
  });
});

describe('easeSmoothstep', () => {
  it('hits the keyframes', () => {
    expect(easeSmoothstep(0)).toBe(0);
    expect(easeSmoothstep(0.5)).toBe(0.5);
    expect(easeSmoothstep(1)).toBe(1);
  });

  it('is monotonic on [0, 1]', () => {
    let previous = easeSmoothstep(0);
    for (let i = 1; i <= 100; i += 1) {
      const current = easeSmoothstep(i / 100);
      expect(current).toBeGreaterThanOrEqual(previous);
      previous = current;
    }
  });
});

describe('normalizeHeight', () => {
  it('maps finite values linearly to the scene height', () => {
    expect(normalizeHeight(5, 3, 4)).toBeCloseTo(0.75, 12);
  });

  it('uses the explicit scale instead of the default', () => {
    expect(normalizeHeight(5, 3, 4, 5)).toBeCloseTo(2.5, 12);
  });

  it('sends non-finite values to the nearest end', () => {
    expect(normalizeHeight(Number.POSITIVE_INFINITY, 3, 4)).toBe(1.5);
    expect(normalizeHeight(Number.NEGATIVE_INFINITY, 3, 4)).toBe(0);
    expect(normalizeHeight(Number.NaN, 3, 4)).toBe(0);
  });

  it('returns zero for a flat range', () => {
    expect(normalizeHeight(7, 3, 0)).toBe(0);
    expect(normalizeHeight(7, 3, 1e-10)).toBe(0);
  });

  it('still overflows when the span is infinite (the sampler prevents this upstream)', () => {
    // sampleHeightRange now returns span 0 for such fields; this pins the raw helper behavior.
    expect(Number.isFinite(normalizeHeight(1.5e308, -1.771e308, Number.POSITIVE_INFINITY))).toBe(
      false,
    );
  });
});

describe('applyHeightScale', () => {
  // Scene-level gap: setHeightScale's wiring (cache reuse, camera radius,
  // pan preservation) is not covered here because it needs WebGL, which the
  // node test environment does not provide. These tests cover the pure scale
  // math of the helper that setHeightScale relies on.
  it('re-scales cached samples without calling the height function again', () => {
    let calls = 0;
    const heightFn = (w1: number, w2: number) => {
      calls += 1;
      return w1 + w2;
    };
    const samples = sampleGridHeights(heightFn, 2, 1);
    const callsAfterSampling = calls;
    expect(callsAfterSampling).toBe(9); // (res + 1)^2

    const scaled = applyHeightScale(samples, 0, 2, 3);
    expect(calls).toBe(callsAfterSampling);
    expect(scaled).toHaveLength(samples.length);
    expect(scaled[0]).toBeCloseTo(normalizeHeight(samples[0], 0, 2, 3), 12);
  });

  it('keeps a flat field flat under any scale', () => {
    expect(applyHeightScale([5, 5, 5], 5, 0, 2)).toEqual([0, 0, 0]);
    expect(applyHeightScale([5, 5, 5], 5, 0, 5)).toEqual([0, 0, 0]);
  });

  it('sends non-finite samples to the nearest end', () => {
    expect(applyHeightScale([Number.NaN, 1, 3], 1, 2, 2)).toEqual([0, 0, 2]);
  });
});

describe('sanitizeHeightScale', () => {
  it('passes a valid scale through unchanged', () => {
    expect(sanitizeHeightScale(0.5)).toBe(0.5);
    expect(sanitizeHeightScale(3)).toBe(3);
  });

  it('falls back to the default for zero, negative, and non-finite scales', () => {
    expect(sanitizeHeightScale(0)).toBe(DEFAULT_HEIGHT_SCALE);
    expect(sanitizeHeightScale(-1)).toBe(DEFAULT_HEIGHT_SCALE);
    expect(sanitizeHeightScale(Number.NaN)).toBe(DEFAULT_HEIGHT_SCALE);
    expect(sanitizeHeightScale(Number.POSITIVE_INFINITY)).toBe(DEFAULT_HEIGHT_SCALE);
    expect(sanitizeHeightScale(Number.NEGATIVE_INFINITY)).toBe(DEFAULT_HEIGHT_SCALE);
  });
});

describe('cameraRadius', () => {
  it('uses the minimum radius for the default landscape', () => {
    expect(cameraRadius(5, 1.5)).toBe(10);
  });

  it('grows with the height scale', () => {
    expect(cameraRadius(5, 10)).toBe(20);
  });

  it('grows with the domain', () => {
    expect(cameraRadius(20, 1.5)).toBe(32);
  });
});

describe('sampleGridHeights', () => {
  it('samples the (res+1)² grid in row-major order (j outer, i inner)', () => {
    const samples = sampleGridHeights((w1, w2) => w1 * 10 + w2, 1, 1);
    expect(samples).toBeInstanceOf(Float64Array);
    expect(Array.from(samples)).toEqual([-11, 9, -9, 11]);
  });

  it('returns one sample per grid vertex', () => {
    const samples = sampleGridHeights(() => 0, 3, 2);
    expect(samples).toHaveLength(16);
  });
});

describe('heightRangeFromSamples', () => {
  it('skips non-finite samples and keeps the finite ones', () => {
    expect(
      heightRangeFromSamples([
        Number.POSITIVE_INFINITY,
        2,
        Number.NEGATIVE_INFINITY,
        4,
        Number.NaN,
      ]),
    ).toEqual({
      min: 2,
      span: 2,
    });
  });

  it('reports a flat range when no finite sample exists', () => {
    expect(heightRangeFromSamples([Number.NaN, Number.POSITIVE_INFINITY])).toEqual({
      min: 0,
      span: 0,
    });
    expect(heightRangeFromSamples([])).toEqual({ min: 0, span: 0 });
  });

  it('flattens a finite field whose span overflows', () => {
    expect(heightRangeFromSamples([1e308, -1e308])).toEqual({ min: 0, span: 0 });
  });
});
