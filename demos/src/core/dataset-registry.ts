/**
 * Registre des jeux de données sélectionnables dans la démo MNIST-CNN.
 *
 * Le module est purement déclaratif : aucune dépendance, aucun accès DOM, aucun
 * chargement. Chaque entrée décrit un jeu (identifiant, classes, effectifs,
 * fichiers servis, réglages par défaut, licence) sans contenir les données.
 *
 * Le jeu MedMNIST sera ajouté dans les commits suivants.
 */

/** Format des fichiers servis sous `public/datasets/<id>/`. */
export type DatasetKind = 'idx' | 'sharded';

interface DatasetBase {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly classNames: readonly string[];
  readonly trainCount: number;
  readonly testCount: number;
  readonly defaultTrainPercent: number;
  readonly defaultValidationPercent: number;
  readonly license: string;
  readonly attribution: string;
  readonly kind: DatasetKind;
}

/** Noms des quatre fichiers IDX servis sous `public/datasets/<id>/`. */
export interface DatasetFiles {
  readonly trainImages: string;
  readonly trainLabels: string;
  readonly testImages: string;
  readonly testLabels: string;
}

/** Jeu IDX : quatre fichiers IDX servis sous `public/datasets/<id>/`. */
export interface IdxDatasetDefinition extends DatasetBase {
  readonly kind: 'idx';
  readonly files: DatasetFiles;
}

/**
 * Jeu segmenté : un index de fragments, une galerie d'images compressée et ses
 * étiquettes compressées servis sous `public/datasets/<id>/`.
 */
export interface ShardedDatasetDefinition extends DatasetBase {
  readonly kind: 'sharded';
  readonly indexFile: string;
  readonly galleryFile: string;
  readonly galleryLabelsFile: string;
  readonly shardSize: number;
}

/** Description complète d'un jeu sélectionnable. */
export type DatasetDefinition = IdxDatasetDefinition | ShardedDatasetDefinition;

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
  kind: 'idx',
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
  kind: 'idx',
};

/** Jeu QuickDraw : les croquis dessinés à la main de Google. */
const QUICKDRAW: ShardedDatasetDefinition = {
  id: 'quickdraw',
  name: 'QuickDraw',
  description:
    'Croquis dessinés à la main en niveaux de gris, 28 × 28 pixels : 490 571 dessins d’apprentissage et 122 645 dessins de test, répartis en cinq catégories.',
  classNames: ['livre', 'lune', 'chat', 'parapluie', 'gâteau'],
  trainCount: 490571,
  testCount: 122645,
  indexFile: 'index.json',
  galleryFile: 'gallery.idx.gz',
  galleryLabelsFile: 'gallery-labels.idx.gz',
  shardSize: 1000,
  defaultTrainPercent: 2,
  defaultValidationPercent: 2,
  license: 'CC BY 4.0',
  attribution: 'Google, Quick, Draw!',
  kind: 'sharded',
};

/** Jeux sélectionnables, dans l’ordre d’affichage. Le premier est MNIST. */
export const DATASETS: readonly DatasetDefinition[] = [MNIST, FASHION_MNIST, QUICKDRAW];

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
