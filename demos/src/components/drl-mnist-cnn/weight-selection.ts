/**
 * Sélection de l'éditeur de poids du composant `drl-mnist-cnn` : le geste
 * (liaison puis case de noyau), la vue d'une toile de filtre, le libellé, la
 * lecture d'un poids et l'analyse d'une saisie au clavier.
 *
 * Ce module n'importe ni Lit ni le DOM : il ne dépend que du cœur
 * `core/mnist-cnn.ts`. Le composant `drl-mnist-cnn.ts` le réexporte pour ne
 * pas rompre les chemins d'import existants.
 */

import {
  CONV_KERNEL_SIZE,
  CONV_PADDINGS,
  CONV_STRIDES,
  convKernelIndex,
  convOutputLength,
  MNIST_INPUT_SIZE,
  type MnistCnnFilters,
  type MnistCnnWeights,
} from '../../core/mnist-cnn.js';

/**
 * Tailles des cartes de caractéristiques après chaque convolution, dérivées du
 * noyau (26, 13, 11 et 5 avec l'architecture de référence). Servent à
 * `convLayerGeometry` et aux toiles de cartes du composant.
 */
export const FEATURE_MAP_SIZES: readonly number[] = (() => {
  const sizes: number[] = [];
  let size = MNIST_INPUT_SIZE;
  for (let layer = 0; layer < CONV_STRIDES.length; layer += 1) {
    size = convOutputLength(size, CONV_STRIDES[layer], CONV_PADDINGS[layer]);
    sizes.push(size);
  }
  return sizes;
})();

/**
 * Moyenne des canaux d'entrée d'un noyau de convolution pour un canal de sortie
 * donné. Les noyaux sont stockés à plat en HWIO ; pour les couches à plusieurs
 * canaux d'entrée, la moyenne donne une vue 3×3 lisible.
 */
export function filterKernelMean(
  kernel: Float32Array,
  inputChannels: number,
  outputChannel: number,
): Float32Array {
  const outputChannels = kernel.length / (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * inputChannels);
  const means = new Float32Array(CONV_KERNEL_SIZE * CONV_KERNEL_SIZE);
  for (let ky = 0; ky < CONV_KERNEL_SIZE; ky += 1) {
    for (let kx = 0; kx < CONV_KERNEL_SIZE; kx += 1) {
      let sum = 0;
      const base = (ky * CONV_KERNEL_SIZE + kx) * inputChannels * outputChannels;
      for (let ic = 0; ic < inputChannels; ic += 1) {
        sum += kernel[base + ic * outputChannels + outputChannel];
      }
      means[ky * CONV_KERNEL_SIZE + kx] = sum / inputChannels;
    }
  }
  return means;
}

/**
 * Extrait le noyau 3×3 exact d'un couple (canal d'entrée, canal de sortie), dans
 * l'ordre [ky][kx]. Les noyaux sont stockés à plat en HWIO ; contrairement à
 * `filterKernelMean`, aucun canal n'est moyenné.
 */
export function filterKernelSlice(
  kernel: Float32Array,
  inputChannels: number,
  outputChannel: number,
  inputChannel: number,
): Float32Array {
  if (!Number.isInteger(inputChannels) || inputChannels <= 0) {
    throw new RangeError(
      `filterKernelSlice: inputChannels must be a positive integer, received ${String(inputChannels)}.`,
    );
  }
  const outputChannels = kernel.length / (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * inputChannels);
  if (!Number.isInteger(outputChannels) || outputChannels <= 0) {
    throw new RangeError(
      `filterKernelSlice: kernel length ${String(kernel.length)} is inconsistent with ${String(inputChannels)} input channels.`,
    );
  }
  if (!Number.isInteger(outputChannel) || outputChannel < 0 || outputChannel >= outputChannels) {
    throw new RangeError(
      `filterKernelSlice: outputChannel must be an integer in [0, ${String(outputChannels - 1)}], received ${String(outputChannel)}.`,
    );
  }
  if (!Number.isInteger(inputChannel) || inputChannel < 0 || inputChannel >= inputChannels) {
    throw new RangeError(
      `filterKernelSlice: inputChannel must be an integer in [0, ${String(inputChannels - 1)}], received ${String(inputChannel)}.`,
    );
  }
  const slice = new Float32Array(CONV_KERNEL_SIZE * CONV_KERNEL_SIZE);
  for (let ky = 0; ky < CONV_KERNEL_SIZE; ky += 1) {
    for (let kx = 0; kx < CONV_KERNEL_SIZE; kx += 1) {
      const index =
        (ky * CONV_KERNEL_SIZE + kx) * inputChannels * outputChannels +
        inputChannel * outputChannels +
        outputChannel;
      slice[ky * CONV_KERNEL_SIZE + kx] = kernel[index];
    }
  }
  return slice;
}

/**
 * Sélection de l'éditeur de poids : une liaison (canal d'entrée, canal de sortie)
 * à une frontière entre couches, un poids exact de noyau, ou null quand rien
 * n'est sélectionné.
 *
 * Une frontière `boundary` relie la sortie de la couche `boundary` à l'entrée de
 * la couche `boundary + 1` ; la frontière 0 relie l'image d'entrée à `conv1`.
 */
export type WeightSelection =
  | { kind: 'link'; boundary: number; inputChannel: number; outputChannel: number }
  | {
      kind: 'weight';
      boundary: number;
      inputChannel: number;
      outputChannel: number;
      ky: number;
      kx: number;
    }
  | null;

/**
 * Événement de l'éditeur de poids : clic sur une liaison ou une case de noyau.
 */
export type WeightEditorEvent =
  | { kind: 'link'; boundary: number; inputChannel: number; outputChannel: number }
  | {
      kind: 'filterCell';
      layer: number;
      inputChannel: number;
      outputChannel: number;
      ky: number;
      kx: number;
    };

/**
 * Applique un événement de l'éditeur à la sélection courante. Un clic sur une
 * liaison sélectionne puis désélectionne (toggle) ; un clic sur une case de noyau
 * ne produit une sélection de poids que si la sélection courante est la liaison de
 * la frontière immédiatement en amont de la couche, et ne fait rien sinon. Un
 * clic sur une liaison remplace une sélection de poids.
 */
export function applyWeightEditorEvent(
  selection: WeightSelection,
  event: WeightEditorEvent,
): WeightSelection {
  switch (event.kind) {
    case 'link':
      if (
        selection !== null &&
        selection.kind === 'link' &&
        selection.boundary === event.boundary &&
        selection.inputChannel === event.inputChannel &&
        selection.outputChannel === event.outputChannel
      ) {
        return null;
      }
      return {
        kind: 'link',
        boundary: event.boundary,
        inputChannel: event.inputChannel,
        outputChannel: event.outputChannel,
      };
    case 'filterCell': {
      if (
        selection !== null &&
        selection.kind === 'link' &&
        selection.boundary === event.layer - 1 &&
        selection.outputChannel === event.outputChannel
      ) {
        return {
          kind: 'weight',
          boundary: selection.boundary,
          inputChannel: selection.inputChannel,
          outputChannel: event.outputChannel,
          ky: event.ky,
          kx: event.kx,
        };
      }
      return selection;
    }
  }
}

/**
 * Mode de rendu d'une toile de filtre : `kernel` (tranche 3×3 du canal d'entrée)
 * pour la toile désignée par une sélection de liaison ou de poids, `mean`
 * (moyenne des canaux d'entrée) pour toutes les autres.
 */
export type FilterView = { mode: 'mean' } | { mode: 'kernel'; inputChannel: number };

export function filterViewMode(
  selection: WeightSelection,
  layer: number,
  outputChannel: number,
): FilterView {
  if (
    selection !== null &&
    (selection.kind === 'link' || selection.kind === 'weight') &&
    selection.boundary + 1 === layer &&
    selection.outputChannel === outputChannel
  ) {
    return { mode: 'kernel', inputChannel: selection.inputChannel };
  }
  return { mode: 'mean' };
}

/**
 * Géométrie de convolution d'une couche : pas, remplissage et taille de la carte
 * d'entrée. La frontière 0 relie l'image 28×28 à `conv1` ; les autres relient la
 * carte de la couche précédente. C'est cette géométrie qui permet de calculer
 * les taps d'une cellule de carte.
 */
export function convLayerGeometry(layer: number): {
  stride: number;
  padding: number;
  inputSize: number;
} {
  if (!Number.isInteger(layer) || layer < 0 || layer >= CONV_STRIDES.length) {
    throw new RangeError(
      `convLayerGeometry: layer must be an integer from 0 to ${String(CONV_STRIDES.length - 1)}, received ${String(layer)}.`,
    );
  }
  return {
    stride: CONV_STRIDES[layer],
    padding: CONV_PADDINGS[layer],
    inputSize: layer === 0 ? MNIST_INPUT_SIZE : FEATURE_MAP_SIZES[layer - 1],
  };
}

/**
 * Canal d'entrée affiché par une toile de filtre de `(layer, outputChannel)` : le
 * canal de la liaison désignée par une sélection de liaison. Null quand aucune
 * sélection ne désigne cette toile.
 */
export function filterCanvasInputChannel(
  selection: WeightSelection,
  layer: number,
  outputChannel: number,
): number | null {
  if (selection === null) {
    return null;
  }
  if (selection.kind === 'link') {
    return selection.boundary + 1 === layer && selection.outputChannel === outputChannel
      ? selection.inputChannel
      : null;
  }
  return null;
}

/**
 * Vrai si la liaison `(boundary, inputChannel, outputChannel)` existe pour
 * l'architecture donnée. La frontière 0 relie l'image (un seul canal) à `conv1`.
 */
function lineInArchitecture(
  boundary: number,
  inputChannel: number,
  outputChannel: number,
  filters: MnistCnnFilters,
): boolean {
  if (
    !Number.isInteger(boundary) ||
    boundary < 0 ||
    boundary >= filters.length ||
    !Number.isInteger(outputChannel) ||
    outputChannel < 0 ||
    outputChannel >= filters[boundary]
  ) {
    return false;
  }
  const inputChannels = boundary === 0 ? 1 : filters[boundary - 1];
  return Number.isInteger(inputChannel) && inputChannel >= 0 && inputChannel < inputChannels;
}

/**
 * Vrai si la sélection désigne des indices existants pour l'architecture donnée.
 * Prédicat de type : après un appel positif, la sélection est non nulle.
 */
function selectionIsInArchitecture(
  selection: WeightSelection,
  filters: MnistCnnFilters,
): selection is Exclude<WeightSelection, null> {
  if (selection === null) {
    return false;
  }
  return (
    lineInArchitecture(
      selection.boundary,
      selection.inputChannel,
      selection.outputChannel,
      filters,
    ) &&
    (selection.kind === 'link' ||
      (Number.isInteger(selection.ky) &&
        selection.ky >= 0 &&
        selection.ky < CONV_KERNEL_SIZE &&
        Number.isInteger(selection.kx) &&
        selection.kx >= 0 &&
        selection.kx < CONV_KERNEL_SIZE))
  );
}

/**
 * Libellé français de la sélection pour l'éditeur de poids, ou null si rien n'est
 * sélectionné ou si la sélection ne correspond pas à l'architecture. Les indices
 * sont affichés en base 1.
 */
export function describeWeightSelection(
  selection: WeightSelection,
  filters: MnistCnnFilters,
): string | null {
  if (!selectionIsInArchitecture(selection, filters)) {
    return null;
  }
  const link = `conv${selection.boundary + 1} — entrée ${selection.inputChannel + 1}, sortie ${selection.outputChannel + 1}`;
  if (selection.kind === 'link') {
    return link;
  }
  return `${link}, position (${selection.ky + 1}, ${selection.kx + 1})`;
}

/**
 * Valeur courante d'un poids sélectionné, ou null si la sélection n'est pas un
 * poids ou si ses indices sont hors bornes. Le nombre de canaux d'entrée est
 * déduit de la longueur du noyau et du biais (nombre de canaux de sortie).
 *
 * Tout indice fractionnaire ou `NaN` donne null plutôt qu'une lecture hors
 * bornes : les indices non entiers sont refusés avant tout accès au noyau.
 */
export function selectedWeightValue(
  weights: MnistCnnWeights,
  selection: WeightSelection,
): number | null {
  if (selection === null || selection.kind !== 'weight') {
    return null;
  }
  const layers = [weights.conv1, weights.conv2, weights.conv3, weights.conv4] as const;
  const layer = layers[selection.boundary];
  if (!layer || !Number.isInteger(selection.boundary)) {
    return null;
  }
  const outputChannels = layer.bias.length;
  const inputChannels =
    layer.kernel.length / (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * outputChannels);
  if (
    !Number.isInteger(selection.inputChannel) ||
    !Number.isInteger(selection.outputChannel) ||
    !Number.isInteger(selection.ky) ||
    !Number.isInteger(selection.kx) ||
    !Number.isInteger(inputChannels) ||
    inputChannels <= 0 ||
    selection.inputChannel < 0 ||
    selection.inputChannel >= inputChannels ||
    selection.outputChannel < 0 ||
    selection.outputChannel >= outputChannels ||
    selection.ky < 0 ||
    selection.ky >= CONV_KERNEL_SIZE ||
    selection.kx < 0 ||
    selection.kx >= CONV_KERNEL_SIZE
  ) {
    return null;
  }
  return layer.kernel[
    convKernelIndex(
      inputChannels,
      outputChannels,
      selection.ky,
      selection.kx,
      selection.inputChannel,
      selection.outputChannel,
    )
  ];
}

/** Motif d'un nombre décimal saisi au clavier : signe, chiffres, un séparateur `.` ou `,`. */
const WEIGHT_INPUT_PATTERN = /^[+-]?\d+([.,]\d+)?$/;

/**
 * Analyse un poids saisi au clavier : accepte un signe initial, des chiffres et un
 * séparateur décimal `.` ou `,`. Retourne null pour toute autre forme ou un
 * résultat non fini. Un signe seul sur zéro (`-0`) vaut 0.
 */
export function parseWeightInput(text: string): number | null {
  if (!WEIGHT_INPUT_PATTERN.test(text)) {
    return null;
  }
  const value = Number.parseFloat(text.replace(',', '.'));
  if (!Number.isFinite(value)) {
    return null;
  }
  return value === 0 ? 0 : value;
}
