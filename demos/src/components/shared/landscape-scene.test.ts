import { describe, expect, it } from 'vitest';
import {
  clampFiniteHeight,
  easeSmoothstep,
  gridIndex,
  sampleHeightRange,
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
