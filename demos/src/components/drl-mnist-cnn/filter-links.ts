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
 * les quatre champs sont finis. Le coin inférieur droit `left + width`,
 * `top + height` doit aussi être fini : un dépassement vers l'infini rend le
 * centre incalculable. Un rectangle non utilisable n'a pas de centre calculable.
 */
export function isUsableRect(rect: RectLike): boolean {
  return (
    Number.isFinite(rect.left) &&
    Number.isFinite(rect.top) &&
    Number.isFinite(rect.width) &&
    Number.isFinite(rect.height) &&
    Number.isFinite(rect.left + rect.width) &&
    Number.isFinite(rect.top + rect.height) &&
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
 * supérieur gauche est l'origine). Lève une `RangeError` nommant le rectangle
 * (`source`, `target` ou `container`) qui n'est pas utilisable (voir
 * `isUsableRect`).
 */
export function linkGeometry(
  source: RectLike,
  target: RectLike,
  container: RectLike,
): LineGeometry {
  if (!isUsableRect(source)) {
    throw new RangeError(
      `linkGeometry: source must be a finite rectangle with positive size, received ${JSON.stringify(source)}.`,
    );
  }
  if (!isUsableRect(target)) {
    throw new RangeError(
      `linkGeometry: target must be a finite rectangle with positive size, received ${JSON.stringify(target)}.`,
    );
  }
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
 * Raccourcit un segment de `distance` à chaque extrémité, le long du segment
 * lui-même : les deux extrémités se rapprochent de son milieu, que la cible soit
 * à droite ou à gauche de la source. Sert à écarter une zone de clic des bords
 * des toiles qu'elle relie, sans déplacer la ligne visible. Un segment plus
 * court que deux fois `distance` — y compris un point — devient un segment nul
 * en son milieu. Renvoie un nouveau segment : `geometry` n'est pas modifié.
 */
export function insetSegment(geometry: LineGeometry, distance: number): LineGeometry {
  const dx = geometry.x2 - geometry.x1;
  const dy = geometry.y2 - geometry.y1;
  const length = Math.hypot(dx, dy);
  // `!(... >= ...)` couvre aussi une longueur infinie ou NaN.
  if (!(length >= 2 * distance)) {
    const midX = (geometry.x1 + geometry.x2) / 2;
    const midY = (geometry.y1 + geometry.y2) / 2;
    return { x1: midX, y1: midY, x2: midX, y2: midY };
  }
  const ratio = distance / length;
  return {
    x1: geometry.x1 + dx * ratio,
    y1: geometry.y1 + dy * ratio,
    x2: geometry.x2 - dx * ratio,
    y2: geometry.y2 - dy * ratio,
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
