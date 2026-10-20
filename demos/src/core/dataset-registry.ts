/**
 * Registre des jeux de données sélectionnables dans la démo MNIST-CNN.
 *
 * Le module est purement déclaratif : aucune dépendance, aucun accès DOM, aucun
 * chargement. Chaque entrée décrit un jeu (identifiant, classes, effectifs,
 * fichiers IDX servis, réglages par défaut, licence) sans contenir les données.
 *
 * Les jeux QuickDraw et MedMNIST seront ajoutés dans les commits suivants.
 */

/** Format de la source téléchargée, avant conversion en IDX. */
export type DatasetSourceKind = 'idx' | 'quickdraw' | 'medmnist';

/** Noms des quatre fichiers IDX servis sous `public/datasets/<id>/`. */
export interface DatasetFiles {
  readonly trainImages: string;
  readonly trainLabels: string;
  readonly testImages: string;
  readonly testLabels: string;
}

/** Description complète d'un jeu sélectionnable. */
export interface DatasetDefinition {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly classNames: readonly string[];
  readonly trainCount: number;
  readonly testCount: number;
  readonly files: DatasetFiles;
  readonly defaultTrainPercent: number;
  readonly defaultValidationPercent: number;
  readonly license: string;
  readonly attribution: string;
  readonly sourceKind: DatasetSourceKind;
}

/** Jeu MNIST : les images IDX de Yann LeCun et Corinna Cortes. */
const MNIST: DatasetDefinition = {
  id: 'mnist',
  name: 'MNIST',
  description:
    'Chiffres manuscrits en niveaux de gris, 28 × 28 pixels : 60 000 images d’apprentissage et 10 000 images de test.',
  classNames: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
  trainCount: 60000,
  testCount: 10000,
  files: {
    trainImages: 'train-images-idx3-ubyte.gz',
    trainLabels: 'train-labels-idx1-ubyte.gz',
    testImages: 't10k-images-idx3-ubyte.gz',
    testLabels: 't10k-labels-idx1-ubyte.gz',
  },
  defaultTrainPercent: 10,
  defaultValidationPercent: 10,
  license: 'Voir la notice MNIST',
  attribution: 'Yann LeCun et Corinna Cortes',
  sourceKind: 'idx',
};

/** Jeu Fashion-MNIST : les articles de mode de Zalando Research. */
const FASHION_MNIST: DatasetDefinition = {
  id: 'fashion-mnist',
  name: 'Fashion-MNIST',
  description:
    'Articles de mode en niveaux de gris, 28 × 28 pixels : 60 000 images d’apprentissage et 10 000 images de test.',
  classNames: [
    'T-shirt',
    'Pantalon',
    'Pull',
    'Robe',
    'Manteau',
    'Sandale',
    'Chemise',
    'Baskets',
    'Sac',
    'Bottine',
  ],
  trainCount: 60000,
  testCount: 10000,
  files: {
    trainImages: 'train-images-idx3-ubyte.gz',
    trainLabels: 'train-labels-idx1-ubyte.gz',
    testImages: 't10k-images-idx3-ubyte.gz',
    testLabels: 't10k-labels-idx1-ubyte.gz',
  },
  defaultTrainPercent: 10,
  defaultValidationPercent: 10,
  license: 'MIT',
  attribution: 'Zalando Research',
  sourceKind: 'idx',
};

/** Jeux sélectionnables, dans l’ordre d’affichage. Le premier est MNIST. */
export const DATASETS: readonly DatasetDefinition[] = [MNIST, FASHION_MNIST];

/** Identifiants des jeux, dans le même ordre que `DATASETS`. */
export const DATASET_IDS: readonly string[] = DATASETS.map((dataset) => dataset.id);

/**
 * Retourne le jeu d’identifiant `id`. Rejette avec un `RangeError` qui nomme
 * l’identifiant inconnu et rappelle les identifiants disponibles.
 */
export function getDataset(id: string): DatasetDefinition {
  const found = DATASETS.find((dataset) => dataset.id === id);
  if (found === undefined) {
    throw new RangeError(
      `dataset-registry: unknown dataset "${id}", expected one of ${DATASET_IDS.join(', ')}`,
    );
  }
  return found;
}

/** Nombre de classes d’un jeu, c’est-à-dire `classNames.length`. */
export function datasetClassCount(dataset: DatasetDefinition): number {
  return dataset.classNames.length;
}
