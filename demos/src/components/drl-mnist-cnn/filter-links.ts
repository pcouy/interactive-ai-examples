/**
 * Géométrie et énumération pures des lignes de liaison du composant
 * `drl-mnist-cnn`.
 *
 * Une liaison relie le canal de sortie d'une couche de convolution au canal
 * d'entrée de la couche suivante. Ce module énumère ces couples pour une
 * architecture donnée et calcule, pour chacun, le segment à tracer entre la
 * toile de la couche amont et celle de la couche aval. Il n'importe ni Lit ni le
 * DOM : il ne dépend que des types et du validateur du cœur `core/mnist-cnn.ts`.
 */

import { type MnistCnnFilters, validateFilterCounts } from '../../core/mnist-cnn.js';

/**
 * Une liaison à une frontière : le couple (canal de sortie de la couche amont,
 * canal d'entrée de la couche aval). `boundary` est l'indice de la couche de
 * convolution aval (conv1 = 0) : la frontière 1 relie `conv1` à `conv2`. La
 * frontière 0 relie l'image d'entrée à `conv1` et n'a pas de couche amont :
 * `enumerateLinks` ne l'énumère pas.
 */
export interface LinkEndpoint {
  readonly boundary: number;
  readonly inputChannel: number;
  readonly outputChannel: number;
}

/**
 * Énumère toutes les liaisons entre couches de convolution de l'architecture,
 * dans l'ordre : frontière croissante, puis canal d'entrée croissant, puis canal
 * de sortie croissant. Une frontière `b` porte `filters[b-1] × filters[b]`
 * liaisons ; la frontière 0 (image → `conv1`) n'a pas de couche amont et est
 * omise. Le tuple de filtres est validé par `validateFilterCounts`.
 */
export function enumerateLinks(filters: MnistCnnFilters): readonly LinkEndpoint[] {
  validateFilterCounts(filters);
  const links: LinkEndpoint[] = [];
  for (let boundary = 1; boundary < filters.length; boundary += 1) {
    const inputChannels = filters[boundary - 1];
    const outputChannels = filters[boundary];
    for (let inputChannel = 0; inputChannel < inputChannels; inputChannel += 1) {
      for (let outputChannel = 0; outputChannel < outputChannels; outputChannel += 1) {
        links.push({ boundary, inputChannel, outputChannel });
      }
    }
  }
  return links;
}

/** Rectangle en coordonnées de conteneur (comme `DOMRect`). */
export interface RectLike {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
}

/**
 * Vrai seulement si la largeur et la hauteur sont strictement positives et si
 * les quatre champs sont finis. Un rectangle non utilisable n'a pas de centre
 * calculable.
 */
export function isUsableRect(rect: RectLike): boolean {
  return (
    Number.isFinite(rect.left) &&
    Number.isFinite(rect.top) &&
    Number.isFinite(rect.width) &&
    Number.isFinite(rect.height) &&
    rect.width > 0 &&
    rect.height > 0
  );
}

/** Segment à tracer, en coordonnées relatives au conteneur. */
export interface LineGeometry {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
}

/**
 * Segment de la liaison : du bas-centre du rectangle amont au haut-centre du
 * rectangle aval, exprimé en coordonnées relatives au conteneur (son coin
 * supérieur gauche est l'origine). Lève une `RangeError` quand le conteneur
 * n'est pas utilisable (voir `isUsableRect`).
 */
export function linkGeometry(
  source: RectLike,
  target: RectLike,
  container: RectLike,
): LineGeometry {
  if (!isUsableRect(container)) {
    throw new RangeError(
      `linkGeometry: container must be a finite rectangle with positive size, received ${JSON.stringify(container)}.`,
    );
  }
  return {
    x1: source.left + source.width / 2 - container.left,
    y1: source.top + source.height - container.top,
    x2: target.left + target.width / 2 - container.left,
    y2: target.top - container.top,
  };
}

/**
 * Sélecteur DOM stable de la toile de filtre d'une couche et d'un canal, par
 * exemple `canvas.drl-canvas--filter[data-layer="1"][data-channel="2"]`. Le
 * gabarit pose ces deux attributs de données. Lève une `RangeError` quand
 * `layer` ou `channel` n'est pas un entier positif ou nul.
 */
export function filterCanvasSelector(layer: number, channel: number): string {
  if (!Number.isInteger(layer) || layer < 0) {
    throw new RangeError(
      `filterCanvasSelector: layer must be a non-negative integer, received ${String(layer)}.`,
    );
  }
  if (!Number.isInteger(channel) || channel < 0) {
    throw new RangeError(
      `filterCanvasSelector: channel must be a non-negative integer, received ${String(channel)}.`,
    );
  }
  return `canvas.drl-canvas--filter[data-layer="${String(layer)}"][data-channel="${String(channel)}"]`;
}
