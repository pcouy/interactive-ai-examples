/**
 * Sources des jeux de données IDX servis sous `demos/public/datasets/<id>/`.
 *
 * Ce module est volontairement pur : aucun accès au système de fichiers, aucune
 * dépendance. Il est importé par le script de téléchargement
 * `fetch-datasets.mjs` et par ses tests. Les identifiants des jeux sont les clés
 * de l'objet `DATASET_SOURCES` ; ils doivent correspondre à
 * `DATASET_IDS` dans `src/core/dataset-registry.ts`.
 *
 * Pour MNIST, l'URL de base, les tailles et les MD5 viennent de
 * `mnist-data-manifest.mjs` : ce module ne fait que les restructurer. Pour
 * Fashion-MNIST, les tailles et les MD5 ont été vérifiés octet par octet contre
 * les fichiers téléchargés depuis le dépôt `zalandoresearch/fashion-mnist`.
 *
 * QuickDraw et MedMNIST ne sont pas dans `DATASET_SOURCES` : leurs
 * convertisseurs `convert-quickdraw.mjs` et `convert-medmnist.mjs` téléchargent
 * leurs fichiers directement.
 */

import { MNIST_BASE_URL, MNIST_FILES } from './mnist-data-manifest.mjs';

/** Dépôt GitHub de Fashion-MNIST, dossier `data/fashion/`. */
const FASHION_MNIST_BASE_URL =
  'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/';

/** Les quatre archives de Fashion-MNIST, avec leur taille en octets et leur MD5. */
const FASHION_MNIST_FILES = [
  {
    name: 'train-images-idx3-ubyte.gz',
    size: 26421880,
    md5: '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
  },
  {
    name: 'train-labels-idx1-ubyte.gz',
    size: 29515,
    md5: '25c81989df183df01b3e8a0aad5dffbe',
  },
  {
    name: 't10k-images-idx3-ubyte.gz',
    size: 4422102,
    md5: 'bef4ecab320f06d8554ea6380940ec79',
  },
  {
    name: 't10k-labels-idx1-ubyte.gz',
    size: 5148,
    md5: 'bb300cfdad3c16e7a12a480ee83cd310',
  },
];

/** URL de base des fichiers QuickDraw dans le dépôt `quickdraw_dataset`. */
export const QUICKDRAW_BASE_URL =
  'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/';

/** Les cinq fichiers QuickDraw `.npy`, avec leur taille en octets et leur MD5. */
export const QUICKDRAW_FILES = [
  { name: 'book.npy', size: 93581456, md5: '2ed1a413e8e9dc677c77804144a0a81a' },
  { name: 'moon.npy', size: 95382304, md5: 'b0461b58eca7651c581914093ab11b9e' },
  { name: 'cat.npy', size: 96590448, md5: 'e4bb2f2c14813c33b9bcc21e06eacb43' },
  { name: 'umbrella.npy', size: 97281936, md5: '04dd435714b9c2068f7f799111ef4e35' },
  { name: 'cake.npy', size: 97925600, md5: 'e816ab482b45ef6302a92f4a7f20cf2c' },
];

/** URL de base des fichiers MedMNIST sur Zenodo, dossier `files/`. */
export const MEDMNIST_BASE_URL = 'https://zenodo.org/records/10519652/files/';

/**
 * Les sept archives MedMNIST `.npz`, avec leur taille en octets et leur MD5.
 * MedMNIST n'est pas dans `DATASET_SOURCES` : son convertisseur
 * `convert-medmnist.mjs` télécharge les archives directement depuis Zenodo.
 */
export const MEDMNIST_FILES = [
  {
    dataset: 'octmnist',
    name: 'octmnist.npz',
    size: 54938180,
    md5: 'c68d92d5b585d8d81f7112f81e2d0842',
  },
  {
    dataset: 'pneumoniamnist',
    name: 'pneumoniamnist.npz',
    size: 4170669,
    md5: '28209eda62fecd6e6a2d98b1501bb15f',
  },
  {
    dataset: 'breastmnist',
    name: 'breastmnist.npz',
    size: 559580,
    md5: '750601b1f35ba3300ea97c75c52ff8f6',
  },
  {
    dataset: 'tissuemnist',
    name: 'tissuemnist.npz',
    size: 124962739,
    md5: 'ebe78ee8b05294063de985d821c1c34b',
  },
  {
    dataset: 'organamnist',
    name: 'organamnist.npz',
    size: 38247708,
    md5: '68e3f8846a6bd62f0c9bf841c0d9eacc',
  },
  {
    dataset: 'organcmnist',
    name: 'organcmnist.npz',
    size: 15526411,
    md5: 'b9ceb9546e10131b32923c5bbeaea2b1',
  },
  {
    dataset: 'organsmnist',
    name: 'organsmnist.npz',
    size: 16528359,
    md5: '9ab87b696fb54e2a387ebe992d6ed5f1',
  },
];

/**
 * Jeux de données téléchargeables, indexés par identifiant.
 *
 * Chaque entrée donne l'URL de base du miroir et la liste des fichiers attendus
 * sous `public/datasets/<id>/`.
 */
export const DATASET_SOURCES = {
  mnist: {
    files: MNIST_FILES.map((file) => ({ ...file })),
  },
  'fashion-mnist': {
    files: FASHION_MNIST_FILES.map((file) => ({ ...file })),
  },
};

/** URL de base d'un jeu de données. */
export function datasetBaseUrl(id) {
  if (id === 'mnist') {
    return MNIST_BASE_URL;
  }
  if (id === 'fashion-mnist') {
    return FASHION_MNIST_BASE_URL;
  }
  throw new RangeError(`dataset-sources: unknown dataset "${id}"`);
}

/** Identifiants des jeux de données, dans l'ordre de `DATASET_SOURCES`. */
export const DATASET_SOURCE_IDS = Object.keys(DATASET_SOURCES);

/** URL complète d'un fichier d'un jeu de données. */
export function datasetFileUrl(id, file) {
  return `${datasetBaseUrl(id)}${file.name}`;
}
