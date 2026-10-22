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
  CONV_SCHEDULES,
  convKernelIndex,
  convOutputLength,
  DEFAULT_CONV_LAYERS,
  MNIST_INPUT_SIZE,
  type MnistCnnConfig,
  type MnistCnnWeights,
} from '../../core/mnist-cnn.js';
import type { Locale } from '../../i18n/locale.js';
import { weightSelectionMessage } from './messages.js';

/** Tailles des cartes après chaque convolution pour un nombre de couches donné. */
export function featureMapSizes(convLayers: number = DEFAULT_CONV_LAYERS): readonly number[] {
  const sizes: number[] = [];
  let size = MNIST_INPUT_SIZE;
  for (const spec of CONV_SCHEDULES[convLayers] ?? []) {
    size = convOutputLength(size, spec.stride, spec.padding);
    sizes.push(size);
  }
  return sizes;
}

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
 * Une frontière `boundary` est l'indice de la couche de convolution aval
 * (conv1 = 0) : la frontière `boundary` relie la couche `boundary - 1` à la
 * couche `boundary`. La frontière 0 n'est pas produite par `enumerateLinks`.
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
        selection.boundary === event.layer &&
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
    selection.boundary === layer &&
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
export function convLayerGeometry(
  layer: number,
  convLayers: number = DEFAULT_CONV_LAYERS,
): {
  stride: number;
  padding: number;
  inputSize: number;
} {
  const schedule = CONV_SCHEDULES[convLayers] ?? [];
  if (!Number.isInteger(layer) || layer < 0 || layer >= schedule.length) {
    throw new RangeError(
      `convLayerGeometry: layer must be an integer from 0 to ${String(schedule.length - 1)} for ${String(convLayers)} convolution layer(s), received ${String(layer)}.`,
    );
  }
  return {
    stride: schedule[layer].stride,
    padding: schedule[layer].padding,
    inputSize: layer === 0 ? MNIST_INPUT_SIZE : featureMapSizes(convLayers)[layer - 1],
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
    return selection.boundary === layer && selection.outputChannel === outputChannel
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
  config: MnistCnnConfig,
): boolean {
  const convLayers = config.convLayers ?? DEFAULT_CONV_LAYERS;
  if (
    !Number.isInteger(boundary) ||
    boundary < 0 ||
    boundary >= convLayers ||
    boundary >= config.filters.length ||
    !Number.isInteger(outputChannel) ||
    outputChannel < 0 ||
    outputChannel >= config.filters[boundary]
  ) {
    return false;
  }
  const inputChannels = boundary === 0 ? 1 : config.filters[boundary - 1];
  return Number.isInteger(inputChannel) && inputChannel >= 0 && inputChannel < inputChannels;
}

/**
 * Vrai si la sélection désigne des indices existants pour l'architecture donnée.
 * Prédicat de type : après un appel positif, la sélection est non nulle.
 */
function selectionIsInArchitecture(
  selection: WeightSelection,
  config: MnistCnnConfig,
): selection is Exclude<WeightSelection, null> {
  if (selection === null) {
    return false;
  }
  return (
    lineInArchitecture(
      selection.boundary,
      selection.inputChannel,
      selection.outputChannel,
      config,
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
  config: MnistCnnConfig,
  locale: Locale = 'fr',
): string | null {
  if (!selectionIsInArchitecture(selection, config)) {
    return null;
  }
  const link = weightSelectionMessage(locale, 'linkLabel', {
    layer: selection.boundary + 1,
    input: selection.inputChannel + 1,
    output: selection.outputChannel + 1,
  });
  if (selection.kind === 'link') {
    return link;
  }
  return weightSelectionMessage(locale, 'position', {
    link,
    ky: selection.ky + 1,
    kx: selection.kx + 1,
  });
}

/**
 * Valeur courante d'un poids sélectionné, ou null si la sélection n'est pas un
 * poids ou si ses indices sont hors bornes. Le noyau est lu dans `weights.conv`,
 * à l'indice de la couche active ; le nombre de canaux d'entrée est déduit de la
 * longueur du noyau et du biais (nombre de canaux de sortie).
 *
 * Tout indice fractionnaire ou `NaN` donne null plutôt qu'une lecture hors
 * bornes : les indices non entiers sont refusés avant tout accès au noyau.
 */
export function selectedWeightValue(
  selection: WeightSelection,
  weights: MnistCnnWeights,
  config: MnistCnnConfig,
): number | null {
  if (selection === null || selection.kind !== 'weight') {
    return null;
  }
  const convLayers = config.convLayers ?? DEFAULT_CONV_LAYERS;
  const layer = weights.conv[selection.boundary];
  if (
    !layer ||
    !Number.isInteger(selection.boundary) ||
    selection.boundary < 0 ||
    selection.boundary >= convLayers
  ) {
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
