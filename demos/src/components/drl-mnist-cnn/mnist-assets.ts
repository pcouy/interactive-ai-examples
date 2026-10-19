/**
 * Chargeur navigateur des ressources MNIST locales de la démo `drl-mnist-cnn`.
 *
 * Le module télécharge les quatre fichiers IDX compressés servis sous
 * `import.meta.env.BASE_URL + 'mnist/'`, les décompresse avec
 * `DecompressionStream('gzip')` et retourne les deux jeux IDX bruts (60 000
 * images d'entraînement, 10 000 images de test). Il ne sélectionne aucune
 * fraction : le composant choisit lui-même la part d'entraînement et de
 * validation via `selectStratifiedIndices`.
 *
 * La promesse de chargement est mise en cache : plusieurs appels concurrents
 * (ou une reconnexion) ne déclenchent qu'un seul téléchargement.
 *
 * Ce module est réservé au navigateur : il utilise `fetch`, `Blob`, `Response`
 * et `DecompressionStream`. Le core ne l'importe jamais.
 */

import {
  loadIdxDataset,
  MNIST_CLASS_COUNT,
  type MnistIdxDataset,
  type MnistSample,
  materializeSample,
} from '../../core/mnist-data.js';

/** Répertoire public des ressources MNIST, relatif à la base Vite. */
const MNIST_ASSET_DIRECTORY = 'mnist/';

/** Noms des quatre fichiers IDX compressés livrés dans `demos/public/mnist/`. */
export const MNIST_ASSET_FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages: 't10k-images-idx3-ubyte.gz',
  testLabels: 't10k-labels-idx1-ubyte.gz',
} as const;

/** Les deux jeux IDX bruts : entraînement (60 000) et test (10 000). */
export interface MnistSplits {
  readonly train: MnistIdxDataset;
  readonly test: MnistIdxDataset;
}

interface ViteImportMeta extends ImportMeta {
  readonly env?: { readonly BASE_URL?: string };
}

/** Base Vite, avec un « / » final, pour construire les URL des ressources. */
function assetBaseUrl(): string {
  const base = (import.meta as ViteImportMeta).env?.BASE_URL ?? '/';
  return base.endsWith('/') ? base : `${base}/`;
}

/** Nom de fichier (dernier segment d'URL) utilisé dans les messages d'erreur. */
export function mnistFileName(url: string): string {
  const withoutQuery = url.split('?')[0].split('#')[0];
  const segments = withoutQuery.split('/');
  return segments[segments.length - 1] || url;
}

/**
 * Message d'erreur unique et testable pour un téléchargement impossible : il
 * nomme le fichier manquant et indique la commande qui le télécharge.
 */
export function mnistFetchErrorMessage(url: string, detail: string): string {
  return `mnist-assets: could not load "${mnistFileName(url)}". Run "npm run mnist:data" to download the MNIST data files. (${detail})`;
}

/**
 * Décompresse un flux gzip. Signale clairement l'absence de
 * `DecompressionStream` plutôt que d'échouer silencieusement.
 */
export async function decompressGzip(buffer: ArrayBuffer): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error(
      'mnist-assets: gzip decompression is unavailable because this browser does not support DecompressionStream.',
    );
  }
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Télécharge et décompresse un fichier `.gz` local, avec des erreurs explicites. */
async function fetchGzip(url: string): Promise<Uint8Array> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new Error(mnistFetchErrorMessage(url, `network error: ${(error as Error).message}`));
  }
  if (!response.ok) {
    throw new Error(mnistFetchErrorMessage(url, `HTTP ${response.status} ${response.statusText}`));
  }
  try {
    return await decompressGzip(await response.arrayBuffer());
  } catch (error) {
    throw new Error(
      `mnist-assets: could not decompress "${mnistFileName(url)}". Run "npm run mnist:data" to download a fresh copy. (${(error as Error).message})`,
    );
  }
}

async function downloadSplits(): Promise<MnistSplits> {
  const base = `${assetBaseUrl()}${MNIST_ASSET_DIRECTORY}`;
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
    fetchGzip(`${base}${MNIST_ASSET_FILES.trainImages}`),
    fetchGzip(`${base}${MNIST_ASSET_FILES.trainLabels}`),
    fetchGzip(`${base}${MNIST_ASSET_FILES.testImages}`),
    fetchGzip(`${base}${MNIST_ASSET_FILES.testLabels}`),
  ]);
  return {
    train: loadIdxDataset(trainImages, trainLabels),
    test: loadIdxDataset(testImages, testLabels),
  };
}

let inFlight: Promise<MnistSplits> | null = null;

/**
 * Retourne les deux jeux IDX bruts. Le premier appel lance le téléchargement
 * des quatre fichiers ; les appels suivants réutilisent la même promesse tant
 * qu'elle est en vol. En cas d'échec, le cache est vidé pour permettre une
 * nouvelle tentative.
 */
export function loadMnistSplits(): Promise<MnistSplits> {
  if (inFlight === null) {
    inFlight = downloadSplits().catch((error: unknown) => {
      inFlight = null;
      throw error;
    });
  }
  return inFlight;
}

/**
 * Sélectionne un indice par classe `0` à `9` dans le jeu donné (première
 * occurrence rencontrée), de façon déterministe. Rejette une classe absente.
 */
export function selectGalleryIndices(dataset: MnistIdxDataset): Uint32Array {
  const first = new Array<number>(MNIST_CLASS_COUNT).fill(-1);
  let missing = MNIST_CLASS_COUNT;
  for (let index = 0; index < dataset.labels.length && missing > 0; index += 1) {
    const label = dataset.labels[index];
    if (label < MNIST_CLASS_COUNT && first[label] === -1) {
      first[label] = index;
      missing -= 1;
    }
  }
  for (let label = 0; label < MNIST_CLASS_COUNT; label += 1) {
    if (first[label] === -1) {
      throw new RangeError(`mnist-assets: the dataset has no example for class ${label}.`);
    }
  }
  return Uint32Array.from(first);
}

/**
 * Galerie de dix exemples, un par classe, échantillonnés dans le jeu de test
 * officiel et matérialisés en `MnistSample` (`Float32` dans `[0, 1]`).
 */
export async function loadMnistGallery(): Promise<MnistSample> {
  const splits = await loadMnistSplits();
  return materializeSample(splits.test, selectGalleryIndices(splits.test));
}
