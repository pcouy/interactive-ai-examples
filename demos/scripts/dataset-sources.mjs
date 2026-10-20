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
