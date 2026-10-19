/**
 * Lecteur MNIST autonome : décodage des fichiers IDX bruts (images + libellés)
 * en tableaux typés normalisés, puis échantillonnage équilibré déterministe.
 *
 * Le module est purement TypeScript : aucune dépendance, aucun accès DOM,
 * aucun téléchargement. Les buffers d'entrée ne sont jamais modifiés.
 *
 * Format IDX (gros-boutiste) :
 * - images : magic `2051`, puis `count`, `rows`, `cols` (u32), puis
 *   `count × rows × cols` octets non signés (0 à 255).
 * - libellés : magic `2049`, puis `count` (u32), puis `count` octets (0 à 9).
 */

/** Largeur imposée des images MNIST. */
export const MNIST_IMAGE_WIDTH = 28;

/** Hauteur imposée des images MNIST. */
export const MNIST_IMAGE_HEIGHT = 28;

/** Nombre de pixels d'une image MNIST (28 × 28). */
export const MNIST_PIXELS_PER_IMAGE = MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT;

/** Nombre de classes MNIST (chiffres 0 à 9). */
export const MNIST_CLASS_COUNT = 10;

const IMAGE_MAGIC = 2051;
const LABEL_MAGIC = 2049;
const IMAGE_HEADER_BYTES = 16;
const LABEL_HEADER_BYTES = 8;

/** Buffer IDX brut accepté en entrée : `ArrayBuffer` ou vue `Uint8Array`. */
export type IdxBuffer = ArrayBuffer | Uint8Array;

/** Images MNIST décodées, aplatis en ligne-major, un `Float32Array` de 784 par image. */
export interface IdxImages {
  readonly count: number;
  readonly rows: number;
  readonly cols: number;
  readonly images: Float32Array;
}

/** Jeu de données MNIST complet et cohérent (autant d'images que de libellés). */
export interface MnistDataset {
  readonly count: number;
  /** Pixels normalisés dans `[0, 1]`, ligne-major : `image * 784 + row * 28 + col`. */
  readonly images: Float32Array;
  /** Libellés `0` à `9`, un par image. */
  readonly labels: Uint8Array;
}

/** Options d'échantillonnage équilibré. */
export interface BalancedSampleOptions {
  /** Nombre d'exemples à retenir pour chacune des 10 classes. */
  perClass: number;
  /** Graine entière non négative : la même graine donne toujours le même tirage. */
  seed: number;
}

/** Échantillon équilibré extrait d'un jeu de données MNIST. */
export interface MnistSample {
  readonly count: number;
  /** Même disposition que `MnistDataset.images`, copie fraîche. */
  readonly images: Float32Array;
  /** Même disposition que `MnistDataset.labels`, copie fraîche. */
  readonly labels: Uint8Array;
}

function asBytes(input: IdxBuffer): Uint8Array {
  return input instanceof Uint8Array ? input : new Uint8Array(input);
}

function readU32At(bytes: Uint8Array, offset: number): number {
  return (
    bytes[offset] * 0x1000000 +
    bytes[offset + 1] * 0x10000 +
    bytes[offset + 2] * 0x100 +
    bytes[offset + 3]
  );
}

/**
 * Décode un fichier IDX d'images MNIST. Exige le magic `2051`, des images
 * 28 × 28 et une longueur de buffer exacte (en-tête 16 octets incluse).
 * Chaque octet devient un `Float32` dans `[0, 1]` (octet / 255).
 * Le buffer d'entrée n'est pas modifié.
 */
export function parseIdxImages(input: IdxBuffer): IdxImages {
  const bytes = asBytes(input);
  if (bytes.length < IMAGE_HEADER_BYTES) {
    throw new RangeError(
      `mnist-data: truncated image header, expected at least ${IMAGE_HEADER_BYTES} bytes, got ${bytes.length}`,
    );
  }
  const magic = readU32At(bytes, 0);
  if (magic !== IMAGE_MAGIC) {
    throw new RangeError(
      `mnist-data: unexpected image file magic ${magic}, expected ${IMAGE_MAGIC}`,
    );
  }
  const count = readU32At(bytes, 4);
  const rows = readU32At(bytes, 8);
  const cols = readU32At(bytes, 12);
  if (rows !== MNIST_IMAGE_HEIGHT || cols !== MNIST_IMAGE_WIDTH) {
    throw new RangeError(
      `mnist-data: unsupported image dimensions ${rows}x${cols}, expected ${MNIST_IMAGE_HEIGHT}x${MNIST_IMAGE_WIDTH}`,
    );
  }
  const expectedLength = IMAGE_HEADER_BYTES + count * MNIST_PIXELS_PER_IMAGE;
  if (bytes.length < expectedLength) {
    throw new RangeError(
      `mnist-data: truncated image payload, expected ${expectedLength} bytes, got ${bytes.length}`,
    );
  }
  if (bytes.length > expectedLength) {
    throw new RangeError(
      `mnist-data: unexpected extra bytes after image payload, expected ${expectedLength} bytes, got ${bytes.length}`,
    );
  }
  const images = new Float32Array(count * MNIST_PIXELS_PER_IMAGE);
  for (let index = 0; index < images.length; index += 1) {
    images[index] = bytes[IMAGE_HEADER_BYTES + index] / 255;
  }
  return { count, rows, cols, images };
}

/**
 * Décode un fichier IDX de libellés MNIST. Exige le magic `2049`, une longueur
 * de buffer exacte (en-tête 8 octets incluse) et des valeurs `0` à `9`.
 * Le buffer d'entrée n'est pas modifié.
 */
export function parseIdxLabels(input: IdxBuffer): Uint8Array {
  const bytes = asBytes(input);
  if (bytes.length < LABEL_HEADER_BYTES) {
    throw new RangeError(
      `mnist-data: truncated label header, expected at least ${LABEL_HEADER_BYTES} bytes, got ${bytes.length}`,
    );
  }
  const magic = readU32At(bytes, 0);
  if (magic !== LABEL_MAGIC) {
    throw new RangeError(
      `mnist-data: unexpected label file magic ${magic}, expected ${LABEL_MAGIC}`,
    );
  }
  const count = readU32At(bytes, 4);
  const expectedLength = LABEL_HEADER_BYTES + count;
  if (bytes.length < expectedLength) {
    throw new RangeError(
      `mnist-data: truncated label payload, expected ${expectedLength} bytes, got ${bytes.length}`,
    );
  }
  if (bytes.length > expectedLength) {
    throw new RangeError(
      `mnist-data: unexpected extra bytes after label payload, expected ${expectedLength} bytes, got ${bytes.length}`,
    );
  }
  const labels = new Uint8Array(count);
  for (let index = 0; index < count; index += 1) {
    const value = bytes[LABEL_HEADER_BYTES + index];
    if (value > 9) {
      throw new RangeError(`mnist-data: invalid label ${value} at index ${index}, expected 0 to 9`);
    }
    labels[index] = value;
  }
  return labels;
}

/**
 * Charge un jeu de données MNIST complet depuis un fichier d'images et un
 * fichier de libellés IDX. Les deux fichiers doivent décrire le même nombre
 * d'exemples. Les tableaux retournés sont des copies : les buffers d'entrée
 * et le jeu source ne sont jamais modifiés.
 */
export function loadMnistDataset(imageBuffer: IdxBuffer, labelBuffer: IdxBuffer): MnistDataset {
  const parsedImages = parseIdxImages(imageBuffer);
  const labels = parseIdxLabels(labelBuffer);
  if (parsedImages.count !== labels.length) {
    throw new RangeError(
      `mnist-data: image and label counts differ, ${parsedImages.count} images vs ${labels.length} labels`,
    );
  }
  return { count: parsedImages.count, images: parsedImages.images, labels };
}

/**
 * PRNG mulberry32, même convention que `perceptron.ts` : retourne des
 * `number` dans `[0, 1)` de façon déterministe pour une graine donnée.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function validateSeed(seed: number): void {
  if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(`mnist-data: invalid seed ${seed}, expected a non-negative integer`);
  }
}

function validatePerClass(perClass: number): void {
  if (!Number.isInteger(perClass) || perClass < 1) {
    throw new RangeError(`mnist-data: invalid perClass ${perClass}, expected a positive integer`);
  }
}

function validateDataset(dataset: MnistDataset): void {
  if (dataset.count !== dataset.labels.length) {
    throw new RangeError(
      `mnist-data: dataset count ${dataset.count} does not match label count ${dataset.labels.length}`,
    );
  }
  const expectedImages = dataset.labels.length * MNIST_PIXELS_PER_IMAGE;
  if (dataset.images.length !== expectedImages) {
    throw new RangeError(
      `mnist-data: dataset image length ${dataset.images.length} does not match label count ${dataset.labels.length}, expected ${expectedImages}`,
    );
  }
}

function shuffledIndices(length: number, rng: () => number): number[] {
  const indices: number[] = new Array<number>(length);
  for (let index = 0; index < length; index += 1) {
    indices[index] = index;
  }
  for (let index = length - 1; index > 0; index -= 1) {
    const swap = Math.floor(rng() * (index + 1));
    const temporary = indices[index];
    indices[index] = indices[swap];
    indices[swap] = temporary;
  }
  return indices;
}

/**
 * Échantillonne `perClass` exemples de chaque chiffre `0` à `9`, de façon
 * déterministe pour une graine donnée. La sélection se fait par classe
 * (mélange de Fisher-Yates avec le PRNG de la graine), puis l'ordre des
 * exemples retenus est mélangé une dernière fois pour entrelacer les classes.
 *
 * Rejette une graine ou un `perClass` invalides, un `perClass` supérieur
 * au nombre d'exemples disponibles dans une classe, et un jeu dont `count`
 * ou la longueur des images ne correspond pas aux libellés.
 *
 * Ne modifie jamais le jeu source : les tableaux retournés sont des copies.
 */
export function sampleBalanced(dataset: MnistDataset, options: BalancedSampleOptions): MnistSample {
  const { perClass, seed } = options;
  validateSeed(seed);
  validatePerClass(perClass);
  validateDataset(dataset);

  const classCounts = new Array<number>(MNIST_CLASS_COUNT).fill(0);
  for (const label of dataset.labels) {
    if (label >= MNIST_CLASS_COUNT) {
      throw new RangeError(
        `mnist-data: dataset label ${label} outside the 0 to ${MNIST_CLASS_COUNT - 1} range`,
      );
    }
    classCounts[label] += 1;
  }
  for (let label = 0; label < MNIST_CLASS_COUNT; label += 1) {
    if (classCounts[label] < perClass) {
      throw new RangeError(
        `mnist-data: class ${label} has only ${classCounts[label]} examples, need ${perClass}`,
      );
    }
  }

  const rng = mulberry32(seed);
  const perImage = MNIST_PIXELS_PER_IMAGE;
  const images = new Float32Array(perClass * MNIST_CLASS_COUNT * perImage);
  const labels = new Uint8Array(perClass * MNIST_CLASS_COUNT);
  let cursor = 0;
  for (let label = 0; label < MNIST_CLASS_COUNT; label += 1) {
    const classIndices: number[] = [];
    for (let index = 0; index < dataset.labels.length; index += 1) {
      if (dataset.labels[index] === label) {
        classIndices.push(index);
      }
    }
    const order = shuffledIndices(classIndices.length, rng);
    for (let picked = 0; picked < perClass; picked += 1) {
      const sourceIndex = classIndices[order[picked]];
      const source = dataset.images.subarray(sourceIndex * perImage, (sourceIndex + 1) * perImage);
      images.set(source, cursor * perImage);
      labels[cursor] = label;
      cursor += 1;
    }
  }
  const finalOrder = shuffledIndices(cursor, rng);
  const sampledImages = new Float32Array(cursor * perImage);
  const sampledLabels = new Uint8Array(cursor);
  for (let index = 0; index < cursor; index += 1) {
    const sourceIndex = finalOrder[index];
    const source = images.subarray(sourceIndex * perImage, (sourceIndex + 1) * perImage);
    sampledImages.set(source, index * perImage);
    sampledLabels[index] = labels[sourceIndex];
  }
  return { count: cursor, images: sampledImages, labels: sampledLabels };
}
