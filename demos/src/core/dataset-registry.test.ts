import { describe, expect, it } from 'vitest';
import { DATASET_IDS, DATASETS, datasetClassCount, getDataset } from './dataset-registry.js';

describe('dataset-registry', () => {
  it('lists three datasets and starts with MNIST', () => {
    expect(DATASETS).toHaveLength(3);
    expect(DATASETS[0].id).toBe('mnist');
    expect(DATASET_IDS).toEqual(['mnist', 'fashion-mnist', 'quickdraw']);
  });

  it('gives each dataset a unique lowercase id without slash', () => {
    expect(new Set(DATASET_IDS).size).toBe(DATASET_IDS.length);
    for (const id of DATASET_IDS) {
      expect(id).toBe(id.toLowerCase());
      expect(id).not.toContain('/');
    }
  });

  it('reports the class counts 10, 10 then 5', () => {
    expect(DATASETS.map((dataset) => datasetClassCount(dataset))).toEqual([10, 10, 5]);
  });

  it('gives each dataset positive counts and defaults between 1 and 100', () => {
    for (const dataset of DATASETS) {
      expect(dataset.trainCount).toBeGreaterThan(0);
      expect(dataset.testCount).toBeGreaterThan(0);
      expect(dataset.defaultTrainPercent).toBeGreaterThanOrEqual(1);
      expect(dataset.defaultTrainPercent).toBeLessThanOrEqual(100);
      expect(dataset.defaultValidationPercent).toBeGreaterThanOrEqual(1);
      expect(dataset.defaultValidationPercent).toBeLessThanOrEqual(100);
    }
  });

  it('narrows MNIST and Fashion-MNIST to four distinct IDX file names', () => {
    for (const id of ['mnist', 'fashion-mnist']) {
      const dataset = getDataset(id);
      expect(dataset.kind).toBe('idx');
      if (dataset.kind !== 'idx') {
        throw new Error('expected an idx dataset');
      }
      const names = [
        dataset.files.trainImages,
        dataset.files.trainLabels,
        dataset.files.testImages,
        dataset.files.testLabels,
      ];
      expect(new Set(names).size).toBe(4);
    }
  });

  it('narrows QuickDraw to five classes, sharded files and 2 / 2 defaults', () => {
    const quickdraw = getDataset('quickdraw');
    expect(quickdraw.kind).toBe('sharded');
    if (quickdraw.kind !== 'sharded') {
      throw new Error('expected a sharded dataset');
    }
    expect(quickdraw.classNames).toEqual(['livre', 'lune', 'chat', 'parapluie', 'gâteau']);
    expect(quickdraw.indexFile).toBe('index.json');
    expect(quickdraw.galleryFile).toBe('gallery.idx.gz');
    expect(quickdraw.galleryLabelsFile).toBe('gallery-labels.idx.gz');
    expect(quickdraw.shardSize).toBe(1000);
    expect(quickdraw.defaultTrainPercent).toBe(2);
    expect(quickdraw.defaultValidationPercent).toBe(2);
  });

  it('resolves a known id and rejects an unknown id', () => {
    expect(getDataset('mnist').name).toBe('MNIST');
    expect(getDataset('fashion-mnist').name).toBe('Fashion-MNIST');
    expect(getDataset('quickdraw').name).toBe('QuickDraw');
    expect(() => getDataset('nope')).toThrow(/unknown dataset/i);
  });
});
