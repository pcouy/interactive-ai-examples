import { describe, expect, it } from 'vitest';
import {
  firstGalleryClass,
  GALLERY_SPLITS,
  galleryCacheKey,
  galleryClassOrder,
  galleryPerClassCount,
  gallerySourceLabel,
  gallerySplitLabel,
} from './gallery.js';

describe('gallerySplitLabel', () => {
  it('names the training and validation splits', () => {
    expect(gallerySplitLabel('train')).toBe('Entraînement');
    expect(gallerySplitLabel('test')).toBe('Validation');
  });

  it('names the splits in English', () => {
    expect(gallerySplitLabel('train', 'en')).toBe('Training');
    expect(gallerySplitLabel('test', 'en')).toBe('Validation');
  });
});

describe('gallerySourceLabel', () => {
  it('returns the phrase complement for each split', () => {
    expect(gallerySourceLabel('train')).toBe("d'entraînement");
    expect(gallerySourceLabel('test')).toBe('de validation');
  });

  it('returns the English phrase complement for each split', () => {
    expect(gallerySourceLabel('train', 'en')).toBe('training');
    expect(gallerySourceLabel('test', 'en')).toBe('validation');
  });
});

describe('GALLERY_SPLITS', () => {
  it('lists train then test', () => {
    expect(GALLERY_SPLITS).toEqual(['train', 'test']);
  });
});

describe('galleryCacheKey', () => {
  it('joins the dataset id, split and per-class count', () => {
    expect(galleryCacheKey('mnist', 'train', 10)).toBe('mnist/train/10');
    expect(galleryCacheKey('quickdraw', 'test', 5)).toBe('quickdraw/test/5');
  });

  it('produces different keys for the train and test splits', () => {
    expect(galleryCacheKey('mnist', 'train', 10)).not.toBe(galleryCacheKey('mnist', 'test', 10));
  });
});

describe('galleryPerClassCount', () => {
  it('divides the sample count by the class count', () => {
    expect(galleryPerClassCount(100, 10)).toBe(10);
    expect(galleryPerClassCount(6, 2)).toBe(3);
    expect(galleryPerClassCount(0, 10)).toBe(0);
  });

  it('rejects a sample count that does not divide evenly', () => {
    expect(() => galleryPerClassCount(10, 3)).toThrow(RangeError);
    expect(() => galleryPerClassCount(10, 3)).toThrow(/not divisible/);
  });

  it('rejects invalid counts', () => {
    for (const classCount of [0, -1, 1.5, Number.NaN]) {
      expect(() => galleryPerClassCount(100, classCount)).toThrow(RangeError);
    }
    for (const sampleCount of [-1, 1.5, Number.NaN]) {
      expect(() => galleryPerClassCount(sampleCount, 10)).toThrow(RangeError);
    }
  });
});

describe('galleryClassOrder', () => {
  it('returns every class when the filter is null or empty', () => {
    expect(galleryClassOrder(4, null)).toEqual([0, 1, 2, 3]);
    expect(galleryClassOrder(4, [])).toEqual([0, 1, 2, 3]);
  });

  it('sorts, deduplicates and drops out-of-range classes', () => {
    expect(galleryClassOrder(5, [3, 1, 3, 2])).toEqual([1, 2, 3]);
    expect(galleryClassOrder(3, [2, 5, -1, 1.5, 0])).toEqual([0, 2]);
  });

  it('rejects an invalid class count', () => {
    for (const classCount of [0, -2, 1.5, Number.NaN]) {
      expect(() => galleryClassOrder(classCount, null)).toThrow(RangeError);
    }
  });
});

describe('firstGalleryClass', () => {
  it('keeps the current class when it is still offered', () => {
    expect(firstGalleryClass([1, 3, 5], 3)).toBe(3);
  });

  it('falls back to the first offered class', () => {
    expect(firstGalleryClass([1, 3, 5], 2)).toBe(1);
    expect(firstGalleryClass([1, 3, 5], null)).toBe(1);
  });

  it('returns null when no class is offered', () => {
    expect(firstGalleryClass([], 2)).toBeNull();
    expect(firstGalleryClass([], null)).toBeNull();
  });
});
