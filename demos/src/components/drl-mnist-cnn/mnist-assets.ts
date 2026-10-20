/**
 * Chargeur navigateur des ressources des jeux IDX locaux de la démo
 * `drl-mnist-cnn`.
 *
 * Le module télécharge les quatre fichiers IDX compressés servis sous
 * `import.meta.env.BASE_URL + 'datasets/<id>/'`, les décode de façon tolérante
 * et retourne les deux jeux IDX bruts. Il ne sélectionne aucune fraction : le
 * composant choisit lui-même la part d'entraînement et de validation via
 * `selectStratifiedIndices`.
 *
 * Le répertoire et les noms de fichiers viennent du registre
 * `dataset-registry.ts` : chaque jeu possède son propre répertoire
 * `datasets/<id>/` et son propre nombre de classes.
 *
 * Le décodage est tolérant : un serveur qui annonce `Content-Encoding: gzip`
 * (Vite en développement, par exemple) a déjà décodé le corps, et le
 * décompresser une seconde fois échoue avec « incorrect header check ». On ne
 * décompresser que si le corps porte le préfixe gzip et que l'en-tête ne
 * l'annonce pas. Le payload décodé est validé avant l'analyse : un magic IDX
 * inconnu (page d'erreur HTML, fichier corrompu) produit une erreur qui nomme
 * le fichier et la commande de téléchargement.
 *
 * La promesse de chargement est mise en cache par identifiant de jeu : plusieurs
 * appels concurrents (ou une reconnexion) ne déclenchent qu'un seul
 * téléchargement pour ce jeu.
 *
 * Ce module est réservé au navigateur : il utilise `fetch`, `Blob`, `Response`
 * et `DecompressionStream`. Le core ne l'importe jamais.
 */

import {
  type DatasetDefinition,
  datasetClassCount,
  getDataset,
  type IdxDatasetDefinition,
} from '../../core/dataset-registry.js';
import {
  loadIdxDataset,
  type MnistIdxDataset,
  type MnistSample,
  materializeSample,
} from '../../core/mnist-data.js';

/** Répertoire public des ressources des jeux, relatif à la base Vite. */
const DATASET_ASSET_DIRECTORY = 'datasets';

/** Nom du manifeste de poids pré-entraîné MNIST, servi sous `datasets/mnist/`. */
const MNIST_MODEL_MANIFEST_FILE = 'model-v1.json';

/** Magics IDX gros-boutiste : `2051` pour les images, `2049` pour les libellés. */
const IDX_IMAGE_MAGIC = 2051;
const IDX_LABEL_MAGIC = 2049;

/** Les deux jeux IDX bruts : entraînement et test. */
export interface MnistSplits {
  readonly train: MnistIdxDataset;
  readonly test: MnistIdxDataset;
}

/** Nombre d'exemples proposés par classe dans la galerie par défaut. */
export const GALLERY_PER_CLASS = 10;

interface ViteImportMeta extends ImportMeta {
  readonly env?: { readonly BASE_URL?: string };
}

/** Base Vite, avec un « / » final, pour construire les URL des ressources. */
export function assetBaseUrl(): string {
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
 * nomme le fichier manquant et indique la commande qui le télécharge. La
 * commande vaut `npm run datasets:data` par défaut ; le chargeur segmenté passe
 * `npm run quickdraw:data`.
 */
export function mnistFetchErrorMessage(
  url: string,
  detail: string,
  downloadCommand: string = 'npm run datasets:data',
): string {
  return `mnist-assets: could not load "${mnistFileName(url)}". Run "${downloadCommand}" to download the data files. (${detail})`;
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
function assertIdxPayload(
  bytes: Uint8Array,
  url: string,
  expectedMagic: number,
  downloadCommand: string,
): void {
  const looksLikeHtml = bytes.length > 0 && bytes[0] === 0x3c /* '<' */;
  const magic = bytes.length >= 4 ? readIdxMagic(bytes) : -1;
  if (looksLikeHtml || magic !== expectedMagic) {
    throw new Error(
      `mnist-assets: "${mnistFileName(url)}" is not a valid IDX file (expected magic ${expectedMagic}, received ${magic}). Run "${downloadCommand}" to download the data files.`,
    );
  }
}

/**
 * Télécharge et décode un fichier `.gz` local, avec des erreurs explicites. Le
 * décodage est tolérant (voir `decodeIdxPayload`) et le payload est validé
 * avant l'analyse (voir `assertIdxPayload`). `downloadCommand` nomme la
 * commande qui télécharge le fichier dans les messages d'erreur.
 */
export async function fetchGzip(
  url: string,
  expectedMagic: number,
  downloadCommand: string = 'npm run datasets:data',
): Promise<Uint8Array> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new Error(
      mnistFetchErrorMessage(url, `network error: ${(error as Error).message}`, downloadCommand),
    );
  }
  if (!response.ok) {
    throw new Error(
      mnistFetchErrorMessage(
        url,
        `HTTP ${response.status} ${response.statusText}`,
        downloadCommand,
      ),
    );
  }
  try {
    const bytes = new Uint8Array(await response.arrayBuffer());
    const decoded = await decodeIdxPayload(bytes, response.headers.get('content-encoding'));
    assertIdxPayload(decoded, url, expectedMagic, downloadCommand);
    return decoded;
  } catch (error) {
    throw new Error(
      `mnist-assets: could not load "${mnistFileName(url)}". Run "${downloadCommand}" to download the data files. (${(error as Error).message})`,
    );
  }
}

/**
 * Télécharge le manifeste de poids pré-entraîné `datasets/mnist/model-v1.json`
 * et retourne son texte JSON brut. L'analyse et la validation sont déléguées à
 * `parseMnistCnnManifestJson` (core) : ce chargeur ne fait que réseau et
 * décodage, avec les mêmes erreurs explicites que `fetchGzip`.
 */
export async function fetchMnistCnnManifest(): Promise<string> {
  const url = `${assetBaseUrl()}${DATASET_ASSET_DIRECTORY}/mnist/${MNIST_MODEL_MANIFEST_FILE}`;
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

/**
 * Réduit un jeu du registre à sa branche `idx`, avec une erreur explicite qui
 * nomme le jeu et son format réel. Les jeux segmentés utilisent le chargeur
 * dédié `sharded-assets.ts`.
 */
function requireIdxDataset(dataset: DatasetDefinition): IdxDatasetDefinition {
  if (dataset.kind !== 'idx') {
    throw new RangeError(
      `mnist-assets: dataset "${dataset.id}" is ${dataset.kind}, expected an idx dataset.`,
    );
  }
  return dataset;
}

/**
 * Télécharge et décode les quatre fichiers IDX d'un jeu. Le répertoire et les
 * noms de fichiers viennent de `dataset` ; le nombre de classes accepté vient de
 * `datasetClassCount(dataset)`.
 */
async function downloadDatasetSplits(dataset: IdxDatasetDefinition): Promise<MnistSplits> {
  const base = `${assetBaseUrl()}${DATASET_ASSET_DIRECTORY}/${dataset.id}/`;
  const classCount = datasetClassCount(dataset);
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
    fetchGzip(`${base}${dataset.files.trainImages}`, IDX_IMAGE_MAGIC),
    fetchGzip(`${base}${dataset.files.trainLabels}`, IDX_LABEL_MAGIC),
    fetchGzip(`${base}${dataset.files.testImages}`, IDX_IMAGE_MAGIC),
    fetchGzip(`${base}${dataset.files.testLabels}`, IDX_LABEL_MAGIC),
  ]);
  return {
    train: loadIdxDataset(trainImages, trainLabels, classCount),
    test: loadIdxDataset(testImages, testLabels, classCount),
  };
}

/** Une promesse de chargement par jeu : les jeux ne partagent jamais leur cache. */
const inFlightSplits = new Map<string, Promise<MnistSplits>>();

/**
 * Retourne les deux jeux IDX bruts d'un jeu. Le premier appel lance le
 * téléchargement des quatre fichiers ; les appels suivants réutilisent la même
 * promesse tant qu'elle est en vol. En cas d'échec, l'entrée du cache est
 * supprimée pour permettre une nouvelle tentative.
 */
export function loadDatasetSplits(dataset: IdxDatasetDefinition): Promise<MnistSplits> {
  const cached = inFlightSplits.get(dataset.id);
  if (cached !== undefined) {
    return cached;
  }
  const pending = downloadDatasetSplits(dataset).catch((error: unknown) => {
    inFlightSplits.delete(dataset.id);
    throw error;
  });
  inFlightSplits.set(dataset.id, pending);
  return pending;
}

/**
 * Retourne les deux jeux IDX bruts de MNIST. Le premier appel lance le
 * téléchargement des quatre fichiers ; les appels suivants réutilisent la même
 * promesse tant qu'elle est en vol. En cas d'échec, le cache est vidé pour
 * permettre une nouvelle tentative.
 */
export function loadMnistSplits(): Promise<MnistSplits> {
  return loadDatasetSplits(requireIdxDataset(getDataset('mnist')));
}

/**
 * Sélectionne les `perClass` premiers indices de chaque classe `0` à
 * `classCount - 1` dans le jeu donné, classe par classe (les `perClass` premiers
 * de la classe 0, puis ceux de la classe 1, et ainsi de suite), de façon
 * déterministe : l'ordre naturel du jeu est conservé dans chaque classe.
 * Rejette un `perClass` qui n'est pas un entier positif et une classe qui n'a
 * pas assez d'exemples.
 *
 * `classCount` vaut le nombre de classes de MNIST (10) par défaut, lu dans le
 * registre : les appelants MNIST n'ont donc rien à passer.
 */
export function selectGalleryIndices(
  dataset: MnistIdxDataset,
  perClass: number = GALLERY_PER_CLASS,
  classCount: number = datasetClassCount(getDataset('mnist')),
): Uint32Array {
  if (!Number.isInteger(perClass) || perClass < 1) {
    throw new RangeError(
      `mnist-assets: perClass must be a positive integer, received ${String(perClass)}.`,
    );
  }
  if (!Number.isInteger(classCount) || classCount < 2) {
    throw new RangeError(
      `mnist-assets: classCount must be an integer of at least 2, received ${String(classCount)}.`,
    );
  }
  const picked: number[] = [];
  for (let label = 0; label < classCount; label += 1) {
    let found = 0;
    for (let index = 0; index < dataset.labels.length && found < perClass; index += 1) {
      if (dataset.labels[index] === label) {
        picked.push(index);
        found += 1;
      }
    }
    if (found < perClass) {
      throw new RangeError(
        `mnist-assets: the dataset has only ${found} example(s) for class ${label}, need ${perClass}.`,
      );
    }
  }
  return Uint32Array.from(picked);
}

/**
 * Galerie de `perClass` exemples par classe d'un jeu, échantillonnés dans son
 * jeu de test et matérialisés en `MnistSample` (`Float32` dans `[0, 1]`). Le
 * nombre de classes vient de `datasetClassCount(dataset)`.
 */
export async function loadDatasetGallery(
  dataset: DatasetDefinition,
  perClass: number = GALLERY_PER_CLASS,
): Promise<MnistSample> {
  const idxDataset = requireIdxDataset(dataset);
  const splits = await loadDatasetSplits(idxDataset);
  const indices = selectGalleryIndices(splits.test, perClass, datasetClassCount(idxDataset));
  return materializeSample(splits.test, indices);
}

/**
 * Galerie de `perClass` exemples par chiffre MNIST, échantillonnés dans le jeu
 * de test officiel et matérialisés en `MnistSample` (`Float32` dans `[0, 1]`).
 */
export function loadMnistGallery(perClass: number = GALLERY_PER_CLASS): Promise<MnistSample> {
  return loadDatasetGallery(getDataset('mnist'), perClass);
}
