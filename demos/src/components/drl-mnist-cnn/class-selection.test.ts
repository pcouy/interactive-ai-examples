import { describe, expect, it } from 'vitest';
import { getDataset } from '../../core/dataset-registry.js';
import {
  classOrder,
  defaultEnabledClasses,
  displayClass,
  MIN_ENABLED_CLASSES,
  networkClassIndex,
  normalizedEnabledClasses,
  toggleEnabledClass,
} from './class-selection.js';

const QUICKDRAW = getDataset('quickdraw');
const MNIST = getDataset('mnist');

describe('MIN_ENABLED_CLASSES', () => {
  it('keeps at least two classes active', () => {
    expect(MIN_ENABLED_CLASSES).toBe(2);
  });
});

describe('defaultEnabledClasses', () => {
  it('uses the default selected classes of a selectable dataset', () => {
    expect(defaultEnabledClasses(QUICKDRAW)).toEqual([0, 1, 2, 3, 4]);
  });

  it('enables every class of a non-selectable dataset', () => {
    expect(defaultEnabledClasses(MNIST)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('ignores the default selection of a non-selectable dataset', () => {
    const dataset = { ...QUICKDRAW, selectable: false, defaultSelectedClasses: [2] };
    expect(defaultEnabledClasses(dataset)).toEqual(Array.from({ length: 15 }, (_v, i) => i));
  });

  it('enables every class when a selectable dataset has no default selection', () => {
    const dataset = { ...QUICKDRAW, selectable: true, defaultSelectedClasses: undefined };
    expect(defaultEnabledClasses(dataset)).toEqual(Array.from({ length: 15 }, (_v, i) => i));
  });

  it('normalises a default selection with duplicates and out-of-range indices', () => {
    const dataset = { ...QUICKDRAW, selectable: true, defaultSelectedClasses: [3, 1, 1, 20, -1] };
    expect(defaultEnabledClasses(dataset)).toEqual([1, 3]);
  });

  it('enables every class when the default selection has no valid index', () => {
    const dataset = { ...QUICKDRAW, selectable: true, defaultSelectedClasses: [20, -1] };
    expect(defaultEnabledClasses(dataset)).toEqual(Array.from({ length: 15 }, (_v, i) => i));
  });
});

describe('normalizedEnabledClasses', () => {
  it('returns every class for null or an empty list', () => {
    expect(normalizedEnabledClasses(4, null)).toEqual([0, 1, 2, 3]);
    expect(normalizedEnabledClasses(4, [])).toEqual([0, 1, 2, 3]);
  });

  it('sorts, deduplicates and drops out-of-range indices', () => {
    expect(normalizedEnabledClasses(5, [4, 1, 1, 9, -1, 1.5])).toEqual([1, 4]);
  });

  it('returns every class when no index is valid', () => {
    expect(normalizedEnabledClasses(5, [9, -1, Number.NaN])).toEqual([0, 1, 2, 3, 4]);
  });

  it('rejects an invalid class count', () => {
    for (const classCount of [0, -1, 1.5, Number.NaN]) {
      expect(() => normalizedEnabledClasses(classCount, null)).toThrow(RangeError);
    }
  });
});

describe('toggleEnabledClass', () => {
  it('checks a class and returns a sorted list', () => {
    expect(toggleEnabledClass([0, 1], 4)).toEqual([0, 1, 4]);
    expect(toggleEnabledClass([3, 1], 2)).toEqual([1, 2, 3]);
  });

  it('unchecks a class above the minimum', () => {
    expect(toggleEnabledClass([0, 1, 2], 1)).toEqual([0, 2]);
  });

  it('refuses to uncheck below the minimum', () => {
    expect(toggleEnabledClass([0, 1], 0)).toBeNull();
    expect(toggleEnabledClass([0, 1], 1)).toBeNull();
  });

  it('counts duplicate input entries once', () => {
    expect(toggleEnabledClass([0, 0, 1], 0)).toBeNull();
  });

  it('allows checking from an empty selection', () => {
    expect(toggleEnabledClass([], 3)).toEqual([3]);
  });

  it('rejects an invalid class index', () => {
    for (const index of [-1, 1.5, Number.NaN]) {
      expect(() => toggleEnabledClass([0, 1], index)).toThrow(RangeError);
    }
  });
});

describe('classOrder', () => {
  it('returns a sorted copy without changing the input', () => {
    const enabled = [3, 0, 2];
    expect(classOrder(enabled)).toEqual([0, 2, 3]);
    expect(enabled).toEqual([3, 0, 2]);
  });
});

describe('networkClassIndex and displayClass', () => {
  it('maps an original class to its network index and back', () => {
    const enabled = [2, 5];
    expect(networkClassIndex(enabled, 2)).toBe(0);
    expect(networkClassIndex(enabled, 5)).toBe(1);
    expect(networkClassIndex(enabled, 0)).toBeNull();
    expect(displayClass(enabled, 0)).toBe(2);
    expect(displayClass(enabled, 1)).toBe(5);
  });

  it('uses the order of the enabled list as the network order', () => {
    const enabled = [5, 2];
    expect(networkClassIndex(enabled, 5)).toBe(0);
    expect(networkClassIndex(enabled, 2)).toBe(1);
    expect(displayClass(enabled, 0)).toBe(5);
  });

  it('round-trips every enabled class', () => {
    const enabled = [1, 4, 7];
    for (const classIndex of enabled) {
      expect(displayClass(enabled, networkClassIndex(enabled, classIndex) ?? -1)).toBe(classIndex);
    }
  });

  it('rejects a network index outside the selection', () => {
    for (const networkIndex of [-1, 2, 1.5, Number.NaN]) {
      expect(() => displayClass([2, 5], networkIndex)).toThrow(RangeError);
    }
    expect(() => displayClass([], 0)).toThrow(RangeError);
  });
});
