/**
 * Chargeur navigateur des ressources MNIST locales de la démo `drl-mnist-cnn`.
 *
 * Le module télécharge les quatre fichiers IDX compressés servis sous
 * `import.meta.env.BASE_URL + 'mnist/'`, les décode de façon tolérante et
 * retourne les deux jeux IDX bruts (60 000 images d'entraînement, 10 000
 * images de test). Il ne sélectionne aucune fraction : le composant choisit
 * lui-même la part d'entraînement et de validation via `selectStratifiedIndices`.
 *
 * Le décodage est tolérant : un serveur qui annonce `Content-Encoding: gzip`
 * (Vite en développement, par exemple) a déjà décodé le corps, et le
 * décompresser une seconde fois échoue avec « incorrect header check ». On ne
 * décompresser que si le corps porte le préfixe gzip et que l'en-tête ne
 * l'annonce pas. Le payload décodé est validé avant l'analyse : un magic IDX
 * inconnu (page d'erreur HTML, fichier corrompu) produit une erreur qui nomme
 * le fichier et la commande de téléchargement.
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

/** Nom du manifeste de poids pré-entraîné, servi sous `mnist/model-v1.json`. */
const MNIST_MODEL_MANIFEST_FILE = 'model-v1.json';

/** Magics IDX gros-boutiste : `2051` pour les images, `2049` pour les libellés. */
const IDX_IMAGE_MAGIC = 2051;
const IDX_LABEL_MAGIC = 2049;

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
 * `DecompressionStream` plutôt que d'échouer silencieusement. N'est appelé
 * que quand la décompression est réellement nécessaire.
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

/** Vrai si `bytes` commence par le préfixe gzip `0x1f 0x8b`. */
export function isGzipPayload(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

/**
 * Décodage tolérant d'un payload IDX. Un serveur qui annonce
 * `Content-Encoding: gzip` (Vite en développement, par exemple) a déjà décodé
 * le corps : le décompresser une seconde fois échoue avec « incorrect header
 * check ». On ne décompresser que si le corps porte le préfixe gzip et que
 * l'en-tête ne l'annonce pas ; sinon le corps est déjà décodé.
 */
export async function decodeIdxPayload(
  bytes: Uint8Array,
  contentEncoding: string | null,
): Promise<Uint8Array> {
  const encoding = contentEncoding?.toLowerCase() ?? '';
  if (encoding.includes('gzip')) {
    return bytes;
  }
  if (isGzipPayload(bytes)) {
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return decompressGzip(buffer as ArrayBuffer);
  }
  return bytes;
}

/** Magic IDX gros-boutiste lu sur les quatre premiers octets. */
function readIdxMagic(bytes: Uint8Array): number {
  return ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]) >>> 0;
}

/**
 * Vérifie qu'un payload IDX décodé commence par le magic gros-boutiste attendu
 * (`2051` images, `2049` libellés). Une page d'erreur HTML (qui commence par
 * `<`) ou un magic inconnu signale un fichier corrompu ou manquant : l'erreur
 * nomme le fichier et la commande de téléchargement.
 */
function assertIdxPayload(bytes: Uint8Array, url: string, expectedMagic: number): void {
  const looksLikeHtml = bytes.length > 0 && bytes[0] === 0x3c /* '<' */;
  const magic = bytes.length >= 4 ? readIdxMagic(bytes) : -1;
  if (looksLikeHtml || magic !== expectedMagic) {
    throw new Error(
      `mnist-assets: "${mnistFileName(url)}" is not a valid MNIST IDX file (expected magic ${expectedMagic}, received ${magic}). Run "npm run mnist:data" to download the MNIST data files.`,
    );
  }
}

/**
 * Télécharge et décode un fichier `.gz` local, avec des erreurs explicites. Le
 * décodage est tolérant (voir `decodeIdxPayload`) et le payload est validé
 * avant l'analyse (voir `assertIdxPayload`).
 */
async function fetchGzip(url: string, expectedMagic: number): Promise<Uint8Array> {
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
    const bytes = new Uint8Array(await response.arrayBuffer());
    const decoded = await decodeIdxPayload(bytes, response.headers.get('content-encoding'));
    assertIdxPayload(decoded, url, expectedMagic);
    return decoded;
  } catch (error) {
    throw new Error(
      `mnist-assets: could not load "${mnistFileName(url)}". Run "npm run mnist:data" to download the MNIST data files. (${(error as Error).message})`,
    );
  }
}

/**
 * Télécharge le manifeste de poids pré-entraîné `mnist/model-v1.json` et
 * retourne son texte JSON brut. L'analyse et la validation sont déléguées à
 * `parseMnistCnnManifestJson` (core) : ce chargeur ne fait que réseau et
 * décodage, avec les mêmes erreurs explicites que `fetchGzip`.
 */
export async function fetchMnistCnnManifest(): Promise<string> {
  const url = `${assetBaseUrl()}${MNIST_ASSET_DIRECTORY}${MNIST_MODEL_MANIFEST_FILE}`;
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new Error(mnistFetchErrorMessage(url, `network error: ${(error as Error).message}`));
  }
  if (!response.ok) {
    throw new Error(mnistFetchErrorMessage(url, `HTTP ${response.status} ${response.statusText}`));
  }
  return response.text();
}

async function downloadSplits(): Promise<MnistSplits> {
  const base = `${assetBaseUrl()}${MNIST_ASSET_DIRECTORY}`;
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
    fetchGzip(`${base}${MNIST_ASSET_FILES.trainImages}`, IDX_IMAGE_MAGIC),
    fetchGzip(`${base}${MNIST_ASSET_FILES.trainLabels}`, IDX_LABEL_MAGIC),
    fetchGzip(`${base}${MNIST_ASSET_FILES.testImages}`, IDX_IMAGE_MAGIC),
    fetchGzip(`${base}${MNIST_ASSET_FILES.testLabels}`, IDX_LABEL_MAGIC),
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
