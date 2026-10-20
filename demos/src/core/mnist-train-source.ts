/**
 * Source d'entraînement MNIST adossée à un jeu IDX brut et à une liste
 * d'indices : `readBatch` matérialise uniquement la tranche demandée via
 * `materializeBatch`, et `reshuffle` réordonne les indices en place de façon
 * déterministe, sans jamais relire les octets du jeu.
 *
 * Le module est purement TypeScript : aucune dépendance, aucun DOM. Il
 * implémente le contrat `MnistTrainSource` de `mnist-cnn-tfjs.ts` sans
 * l'importer statiquement (import de type uniquement).
 */

import type { MnistTrainSource } from './mnist-cnn-tfjs.js';
import { type MnistIdxDataset, materializeBatch, type SampleIndices } from './mnist-data.js';

/** Générateur pseudo-aléatoire déterministe, même implantation que `mnist-data.ts`. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Mélange de Fisher-Yates en place, même convention que `mnist-data.ts`. */
function shuffleInPlace(indices: Uint32Array, rng: () => number): void {
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(rng() * (index + 1));
    const temporary = indices[index];
    indices[index] = indices[swap];
    indices[swap] = temporary;
  }
}

/** Source d'entraînement IDX : un `MnistTrainSource` dont `reshuffle` est toujours présent. */
export interface IdxTrainSource extends MnistTrainSource {
  reshuffle(seed: number): void;
}

function validateSeed(seed: number): void {
  if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(
      `mnist-train-source: invalid seed ${String(seed)}, expected a non-negative integer`,
    );
  }
}

/**
 * Construit une `MnistTrainSource` sur un jeu IDX brut et une liste d'indices
 * (par exemple le résultat de `selectStratifiedIndices`). Les indices sont
 * copiés : les relire ou les permuter ne modifie jamais le jeu source, et une
 * mutation ultérieure de la liste d'entrée n'affecte pas la source.
 *
 * `count` vaut `indices.length` ; `readBatch(start, end)` délègue à
 * `materializeBatch`, qui rejette toute plage invalide ; `reshuffle(seed)`
 * réordonne les indices en place avec un mélange de Fisher-Yates déterministe
 * (même graine = même ordre), sans relire les octets du jeu.
 *
 * Rejette une liste vide, un indice non entier ou hors de `[0, count)`, et une
 * graine invalide.
 */
export function createIdxTrainSource(
  dataset: MnistIdxDataset,
  indices: SampleIndices,
): IdxTrainSource {
  if (dataset === null || typeof dataset !== 'object') {
    throw new TypeError('mnist-train-source: an IDX dataset object is required.');
  }
  // Validation avant copie : `Uint32Array.from` tronquerait une valeur non
  // entière au lieu de la rejeter.
  if (indices.length === 0) {
    throw new RangeError('mnist-train-source: at least one index is required.');
  }
  for (let position = 0; position < indices.length; position += 1) {
    const index = indices[position];
    if (!Number.isInteger(index) || index < 0 || index >= dataset.count) {
      throw new RangeError(
        `mnist-train-source: index ${String(index)} at position ${position} is outside 0 to ${dataset.count - 1}`,
      );
    }
  }
  const selected = new Uint32Array(indices);
  return {
    count: selected.length,
    readBatch(start: number, end: number) {
      return materializeBatch(dataset, selected, start, end);
    },
    reshuffle(seed: number) {
      validateSeed(seed);
      shuffleInPlace(selected, mulberry32(seed));
    },
  };
}
