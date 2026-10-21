/**
 * Registre des jeux de données sélectionnables dans la démo MNIST-CNN.
 *
 * Le module est purement déclaratif : aucune dépendance, aucun accès DOM, aucun
 * chargement. Chaque entrée décrit un jeu (identifiant, classes, effectifs,
 * fichiers servis, réglages par défaut, licence) sans contenir les données.
 *
 * Les jeux QuickDraw et MedMNIST utilisent le format segmenté : ils sont décrits
 * par un index de fragments, et non par quatre fichiers IDX.
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
  /** Vrai si l'utilisateur peut choisir un sous-ensemble de classes. */
  readonly selectable?: boolean;
  /** Indices des classes cochées au départ (sélectionnables seulement). */
  readonly defaultSelectedClasses?: readonly number[];
  /** Version du cache navigateur des shards. */
  readonly cacheVersion?: number;
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
 * Jeu segmenté : un index de fragments servi sous `public/datasets/<id>/`. Les
 * noms des fichiers de la galerie et des fragments viennent de l'index, qui en
 * est la source autoritative.
 */
export interface ShardedDatasetDefinition extends DatasetBase {
  readonly kind: 'sharded';
  readonly indexFile: string;
  readonly shardSize: number;
  /** Commande qui regénère les fichiers de ce jeu. */
  readonly downloadCommand: string;
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
    'Croquis dessinés à la main en niveaux de gris, 28 × 28 pixels : 1 597 997 dessins d’apprentissage et 399 507 dessins de test, répartis en quinze catégories.',
  classNames: [
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
  ],
  trainCount: 1597997,
  testCount: 399507,
  indexFile: 'index.json',
  shardSize: 1000,
  selectable: true,
  defaultSelectedClasses: [0, 1, 2, 3, 4],
  cacheVersion: 2,
  defaultTrainPercent: 2,
  defaultValidationPercent: 2,
  license: 'CC BY 4.0',
  attribution: 'Google, Quick, Draw!',
  kind: 'sharded',
  downloadCommand: 'npm run quickdraw:data',
};

/**
 * Jeu OCTMNIST : les coupes de tomographie optique de la rétine du consortium MedMNIST.
 */
const OCTMNIST: ShardedDatasetDefinition = {
  id: 'octmnist',
  name: 'OCTMNIST',
  description:
    'Imagerie médicale : coupes de tomographie optique de la rétine, 28 × 28 pixels : 108 309 images issues des divisions d’apprentissage et de validation fusionnées, et 1 000 images de test officielles réservées à la validation et à la galerie.',
  classNames: ['choroidal neovascularization', 'diabetic macular edema', 'drusen', 'normal'],
  trainCount: 108309,
  testCount: 1000,
  indexFile: 'index.json',
  shardSize: 1000,
  defaultTrainPercent: 10,
  defaultValidationPercent: 10,
  license: 'CC BY 4.0',
  attribution: 'MedMNIST (Zenodo 10519652)',
  kind: 'sharded',
  downloadCommand: 'npm run medmnist:data',
};

/**
 * Jeu PneumoniaMNIST : les radiographies pulmonaires du consortium MedMNIST.
 */
const PNEUMONIAMNIST: ShardedDatasetDefinition = {
  id: 'pneumoniamnist',
  name: 'PneumoniaMNIST',
  description:
    'Imagerie médicale : radiographies pulmonaires, 28 × 28 pixels : 5 232 images issues des divisions d’apprentissage et de validation fusionnées, et 624 images de test officielles réservées à la validation et à la galerie.',
  classNames: ['normal', 'pneumonia'],
  trainCount: 5232,
  testCount: 624,
  indexFile: 'index.json',
  shardSize: 1000,
  defaultTrainPercent: 10,
  defaultValidationPercent: 10,
  license: 'CC BY 4.0',
  attribution: 'MedMNIST (Zenodo 10519652)',
  kind: 'sharded',
  downloadCommand: 'npm run medmnist:data',
};

/**
 * Jeu BreastMNIST : les coupes histopathologiques du sein du consortium MedMNIST.
 */
const BREASTMNIST: ShardedDatasetDefinition = {
  id: 'breastmnist',
  name: 'BreastMNIST',
  description:
    'Imagerie médicale : coupes histopathologiques du sein, 28 × 28 pixels : 624 images issues des divisions d’apprentissage et de validation fusionnées, et 156 images de test officielles réservées à la validation et à la galerie.',
  classNames: ['malignant', 'normal benign'],
  trainCount: 624,
  testCount: 156,
  indexFile: 'index.json',
  shardSize: 1000,
  defaultTrainPercent: 10,
  defaultValidationPercent: 10,
  license: 'CC BY 4.0',
  attribution: 'MedMNIST (Zenodo 10519652)',
  kind: 'sharded',
  downloadCommand: 'npm run medmnist:data',
};

/**
 * Jeu TissueMNIST : les coupes histopathologiques de cellules rénales du consortium MedMNIST.
 */
const TISSUEMNIST: ShardedDatasetDefinition = {
  id: 'tissuemnist',
  name: 'TissueMNIST',
  description:
    'Imagerie médicale : coupes histopathologiques de cellules rénales, 28 × 28 pixels : 189 106 images issues des divisions d’apprentissage et de validation fusionnées, et 47 280 images de test officielles réservées à la validation et à la galerie.',
  classNames: [
    'Collecting Duct, Connecting Tubule',
    'Distal Convoluted Tubule',
    'Glomerular endothelial cells',
    'Interstitial endothelial cells',
    'Leukocytes',
    'Podocytes',
    'Proximal Tubule Segments',
    'Thick Ascending Limb',
  ],
  trainCount: 189106,
  testCount: 47280,
  indexFile: 'index.json',
  shardSize: 1000,
  defaultTrainPercent: 10,
  defaultValidationPercent: 10,
  license: 'CC BY 4.0',
  attribution: 'MedMNIST (Zenodo 10519652)',
  kind: 'sharded',
  downloadCommand: 'npm run medmnist:data',
};

/** Noms des onze organes communs aux jeux OrganA, OrganC et OrganS. */
const ORGAN_CLASS_NAMES: readonly string[] = [
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
];

/** Jeu OrganAMNIST : les coupes axiales de tomodensitométrie du consortium MedMNIST. */
const ORGANAMNIST: ShardedDatasetDefinition = {
  id: 'organamnist',
  name: 'OrganAMNIST',
  description:
    'Imagerie médicale : coupes axiales de tomodensitométrie abdominale, 28 × 28 pixels : 41 052 images issues des divisions d’apprentissage et de validation fusionnées, et 17 778 images de test officielles réservées à la validation et à la galerie.',
  classNames: [...ORGAN_CLASS_NAMES],
  trainCount: 41052,
  testCount: 17778,
  indexFile: 'index.json',
  shardSize: 1000,
  defaultTrainPercent: 10,
  defaultValidationPercent: 10,
  license: 'CC BY 4.0',
  attribution: 'MedMNIST (Zenodo 10519652)',
  kind: 'sharded',
  downloadCommand: 'npm run medmnist:data',
};

/** Jeu OrganCMNIST : les coupes coronales de tomodensitométrie du consortium MedMNIST. */
const ORGANCMNIST: ShardedDatasetDefinition = {
  id: 'organcmnist',
  name: 'OrganCMNIST',
  description:
    'Imagerie médicale : coupes coronales de tomodensitométrie abdominale, 28 × 28 pixels : 15 367 images issues des divisions d’apprentissage et de validation fusionnées, et 8 216 images de test officielles réservées à la validation et à la galerie.',
  classNames: [...ORGAN_CLASS_NAMES],
  trainCount: 15367,
  testCount: 8216,
  indexFile: 'index.json',
  shardSize: 1000,
  defaultTrainPercent: 10,
  defaultValidationPercent: 10,
  license: 'CC BY 4.0',
  attribution: 'MedMNIST (Zenodo 10519652)',
  kind: 'sharded',
  downloadCommand: 'npm run medmnist:data',
};

/** Jeu OrganSMNIST : les coupes sagittales de tomodensitométrie du consortium MedMNIST. */
const ORGANSMNIST: ShardedDatasetDefinition = {
  id: 'organsmnist',
  name: 'OrganSMNIST',
  description:
    'Imagerie médicale : coupes sagittales de tomodensitométrie abdominale, 28 × 28 pixels : 16 384 images issues des divisions d’apprentissage et de validation fusionnées, et 8 827 images de test officielles réservées à la validation et à la galerie.',
  classNames: [...ORGAN_CLASS_NAMES],
  trainCount: 16384,
  testCount: 8827,
  indexFile: 'index.json',
  shardSize: 1000,
  defaultTrainPercent: 10,
  defaultValidationPercent: 10,
  license: 'CC BY 4.0',
  attribution: 'MedMNIST (Zenodo 10519652)',
  kind: 'sharded',
  downloadCommand: 'npm run medmnist:data',
};

/** Jeux sélectionnables, dans l’ordre d’affichage. Le premier est MNIST. */
export const DATASETS: readonly DatasetDefinition[] = [
  MNIST,
  FASHION_MNIST,
  QUICKDRAW,
  OCTMNIST,
  PNEUMONIAMNIST,
  BREASTMNIST,
  TISSUEMNIST,
  ORGANAMNIST,
  ORGANCMNIST,
  ORGANSMNIST,
];

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
