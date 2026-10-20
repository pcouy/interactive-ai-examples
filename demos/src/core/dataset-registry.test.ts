import { describe, expect, it } from 'vitest';
import { DATASET_IDS, DATASETS, datasetClassCount, getDataset } from './dataset-registry.js';

describe('dataset-registry', () => {
  it('lists two datasets and starts with MNIST', () => {
    expect(DATASETS).toHaveLength(2);
    expect(DATASETS[0].id).toBe('mnist');
    expect(DATASET_IDS).toEqual(['mnist', 'fashion-mnist']);
  });

  it('gives each dataset a unique lowercase id without slash', () => {
    expect(new Set(DATASET_IDS).size).toBe(DATASET_IDS.length);
    for (const id of DATASET_IDS) {
      expect(id).toBe(id.toLowerCase());
      expect(id).not.toContain('/');
    }
  });

  it('gives each dataset ten classes and reports the class count', () => {
    for (const dataset of DATASETS) {
      expect(dataset.classNames).toHaveLength(10);
      expect(datasetClassCount(dataset)).toBe(10);
    }
  });

  it('gives each dataset positive counts and four distinct file names', () => {
    for (const dataset of DATASETS) {
      expect(dataset.trainCount).toBeGreaterThan(0);
      expect(dataset.testCount).toBeGreaterThan(0);
      const names = [
        dataset.files.trainImages,
        dataset.files.trainLabels,
        dataset.files.testImages,
        dataset.files.testLabels,
      ];
      expect(new Set(names).size).toBe(4);
    }
  });

  it('keeps each default percentage between 1 and 100', () => {
    for (const dataset of DATASETS) {
      expect(dataset.defaultTrainPercent).toBeGreaterThanOrEqual(1);
      expect(dataset.defaultTrainPercent).toBeLessThanOrEqual(100);
      expect(dataset.defaultValidationPercent).toBeGreaterThanOrEqual(1);
      expect(dataset.defaultValidationPercent).toBeLessThanOrEqual(100);
    }
  });

  it('resolves a known id and rejects an unknown id', () => {
    expect(getDataset('mnist').name).toBe('MNIST');
    expect(getDataset('fashion-mnist').name).toBe('Fashion-MNIST');
    expect(() => getDataset('nope')).toThrow(/unknown dataset/i);
  });
});
