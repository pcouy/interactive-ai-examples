import { describe, expect, it } from 'vitest';
import { DATASET_IDS, DATASETS, datasetClassCount, getDataset } from './dataset-registry.js';

/** Identifiants des sept jeux MedMNIST, dans l'ordre d'affichage. */
const MEDMNIST_IDS = [
  'octmnist',
  'pneumoniamnist',
  'breastmnist',
  'tissuemnist',
  'organamnist',
  'organcmnist',
  'organsmnist',
];

/** Nombre de classes de chaque jeu MedMNIST, dans le même ordre. */
const MEDMNIST_CLASS_COUNTS = [4, 2, 2, 8, 11, 11, 11];

describe('dataset-registry', () => {
  it('lists ten datasets in display order and starts with MNIST', () => {
    expect(DATASETS).toHaveLength(10);
    expect(DATASETS[0].id).toBe('mnist');
    expect(DATASET_IDS).toEqual(['mnist', 'fashion-mnist', 'quickdraw', ...MEDMNIST_IDS]);
  });

  it('gives each dataset a unique lowercase id without slash', () => {
    expect(new Set(DATASET_IDS).size).toBe(DATASET_IDS.length);
    for (const id of DATASET_IDS) {
      expect(id).toBe(id.toLowerCase());
      expect(id).not.toContain('/');
    }
  });

  it('reports the class counts 10, 10, 15 then the MedMNIST counts', () => {
    expect(DATASETS.map((dataset) => datasetClassCount(dataset))).toEqual([
      10,
      10,
      15,
      ...MEDMNIST_CLASS_COUNTS,
    ]);
  });

  it('gives each dataset between two and fifteen classes', () => {
    for (const dataset of DATASETS) {
      expect(datasetClassCount(dataset)).toBeGreaterThanOrEqual(2);
      expect(datasetClassCount(dataset)).toBeLessThanOrEqual(15);
    }
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

  it('narrows QuickDraw to fifteen classes, an index file and 2 / 2 defaults', () => {
    const quickdraw = getDataset('quickdraw');
    expect(quickdraw.kind).toBe('sharded');
    if (quickdraw.kind !== 'sharded') {
      throw new Error('expected a sharded dataset');
    }
    expect(quickdraw.classNames).toEqual([
      'livre',
      'lune',
      'chat',
      'parapluie',
      'gâteau',
      'chien',
      'poisson',
      'fleur',
      'étoile',
      'maison',
      'arbre',
      'soleil',
      'vélo',
      'pizza',
      'pomme',
    ]);
    expect(quickdraw.trainCount).toBe(1597997);
    expect(quickdraw.testCount).toBe(399507);
    expect(quickdraw.indexFile).toBe('index.json');
    expect(quickdraw.shardSize).toBe(1000);
    expect(quickdraw.defaultTrainPercent).toBe(2);
    expect(quickdraw.defaultValidationPercent).toBe(2);
    expect(quickdraw.downloadCommand).toBe('npm run quickdraw:data');
  });

  it('makes QuickDraw the only dataset with a selectable class subset', () => {
    const quickdraw = getDataset('quickdraw');
    expect(quickdraw.selectable).toBe(true);
    expect(quickdraw.defaultSelectedClasses).toEqual([0, 1, 2, 3, 4]);
    expect(quickdraw.cacheVersion).toBe(2);
    for (const dataset of DATASETS) {
      if (dataset.id === 'quickdraw') {
        continue;
      }
      expect(dataset.selectable).toBeUndefined();
      expect(dataset.defaultSelectedClasses).toBeUndefined();
      expect(dataset.cacheVersion).toBeUndefined();
    }
  });

  it('selects valid QuickDraw class indices by default', () => {
    const quickdraw = getDataset('quickdraw');
    const selected = quickdraw.defaultSelectedClasses ?? [];
    expect(selected.length).toBeGreaterThan(0);
    for (const index of selected) {
      expect(Number.isInteger(index)).toBe(true);
      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThan(quickdraw.classNames.length);
    }
  });

  it('narrows each MedMNIST game to a sharded index with 10 / 10 defaults', () => {
    for (const id of MEDMNIST_IDS) {
      const dataset = getDataset(id);
      expect(dataset.kind).toBe('sharded');
      if (dataset.kind !== 'sharded') {
        throw new Error('expected a sharded dataset');
      }
      expect(dataset.indexFile).toBe('index.json');
      expect(dataset.shardSize).toBe(1000);
      expect(dataset.defaultTrainPercent).toBe(10);
      expect(dataset.defaultValidationPercent).toBe(10);
      expect(dataset.license).toBe('CC BY 4.0');
      expect(dataset.attribution).toBe('MedMNIST (Zenodo 10519652)');
      expect(dataset.downloadCommand).toBe('npm run medmnist:data');
    }
  });

  it('gives each MedMNIST game the official class count and class names', () => {
    for (const [index, id] of MEDMNIST_IDS.entries()) {
      const dataset = getDataset(id);
      expect(datasetClassCount(dataset)).toBe(MEDMNIST_CLASS_COUNTS[index]);
      expect(dataset.classNames).toHaveLength(MEDMNIST_CLASS_COUNTS[index]);
      for (const name of dataset.classNames) {
        expect(name.length).toBeGreaterThan(0);
      }
    }
  });

  it('names the MedMNIST games and the eleven organ classes once', () => {
    expect(MEDMNIST_IDS.map((id) => getDataset(id).name)).toEqual([
      'OCTMNIST',
      'PneumoniaMNIST',
      'BreastMNIST',
      'TissueMNIST',
      'OrganAMNIST',
      'OrganCMNIST',
      'OrganSMNIST',
    ]);
    for (const id of ['organamnist', 'organcmnist', 'organsmnist']) {
      expect(getDataset(id).classNames).toEqual([
        'bladder',
        'femur-left',
        'femur-right',
        'heart',
        'kidney-left',
        'kidney-right',
        'liver',
        'lung-left',
        'lung-right',
        'pancreas',
        'spleen',
      ]);
    }
  });

  it('resolves a known id and rejects an unknown id', () => {
    expect(getDataset('mnist').name).toBe('MNIST');
    expect(getDataset('fashion-mnist').name).toBe('Fashion-MNIST');
    expect(getDataset('quickdraw').name).toBe('QuickDraw');
    expect(getDataset('octmnist').name).toBe('OCTMNIST');
    expect(getDataset('organsmnist').name).toBe('OrganSMNIST');
    expect(() => getDataset('nope')).toThrow(/unknown dataset/i);
  });
});
