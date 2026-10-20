import type { TemplateResult } from 'lit';
import { css, html, LitElement, nothing, svg } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import '../../ui/drl-slider.js';
import {
  CONV_KERNEL_SIZE,
  CONV_LAYER_NAMES,
  DEFAULT_FILTER_COUNTS,
  forwardMnistCnn,
  initializeMnistCnnWeights,
  MAX_FILTER_COUNT,
  MIN_FILTER_COUNT,
  MNIST_INPUT_LENGTH,
  MNIST_INPUT_SIZE,
  type MnistCnnFeatureMap,
  type MnistCnnFilters,
  type MnistCnnForwardResult,
  type MnistCnnLayerWeights,
  type MnistCnnWeights,
  mnistCnnParameterCount,
} from '../../core/mnist-cnn.js';
import { type MnistCnnAssets, parseMnistCnnManifestJson } from '../../core/mnist-cnn-assets.js';
import {
  createMnistCnnTfjsTrainer,
  type MnistTrainingProgress,
} from '../../core/mnist-cnn-tfjs.js';
import {
  MNIST_CLASS_COUNT,
  type MnistSample,
  selectStratifiedIndices,
} from '../../core/mnist-data.js';
import { createIdxTrainSource } from '../../core/mnist-train-source.js';
import { drlTokens } from '../../styles/tokens.js';
import {
  badgeText,
  type DemoStatusInput,
  formatPercent,
  statusLine,
  weightsOriginText,
} from './demo-status.js';
import {
  enumerateLinks,
  filterCanvasSelector,
  insetSegment,
  isUsableRect,
  type LineGeometry,
  type LinkEndpoint,
  linkGeometry,
  type RectLike,
} from './filter-links.js';
import { fetchMnistCnnManifest, loadMnistGallery, loadMnistSplits } from './mnist-assets.js';
import {
  accuracyPoints,
  appendHistory,
  CHART_BOX,
  type ChartLayout,
  type ChartSeries,
  emptyHistory,
  epochPoint,
  formatValue,
  layoutChart,
  layoutMultiChart,
  lossPoints,
  type MultiChartLayout,
  type TrainingHistory,
  trainAccuracyPoints,
} from './training-chart.js';
import { type MnistRunRequest, MnistTrainingController } from './training-controller.js';
import { isTrainingActive, type TrainingState } from './training-state.js';
import { applyWeightDraft, weightEditorHint, weightEditorView } from './weight-editor.js';
import {
  applyWeightEditorEvent,
  FEATURE_MAP_SIZES,
  type FilterView,
  filterCanvasInputChannel,
  filterKernelMean,
  filterKernelSlice,
  filterViewMode,
  type WeightEditorEvent,
  type WeightSelection,
} from './weight-selection.js';

export type { FilterView, WeightEditorEvent, WeightSelection } from './weight-selection.js';
export {
  applyWeightEditorEvent,
  convLayerGeometry,
  describeWeightSelection,
  filterCanvasInputChannel,
  filterKernelMean,
  filterKernelSlice,
  filterViewMode,
  parseWeightInput,
  selectedWeightValue,
} from './weight-selection.js';

/**
 * Démonstration interactive du CNN MNIST de référence (`tf-playground/mnist/train.py`) :
 * quatre convolutions 3×3 + sigmoïde, aplatissement, dense 8 + sigmoïde, dix logits.
 *
 * L'inférence est le passage avant pur TypeScript de `core/mnist-cnn.ts` ; le rendu
 * est un assemblage de canvas 2D stylés avec les jetons `drlTokens`.
 *
 * Poids : au premier affichage, le manifeste pré-entraîné `mnist/model-v1.json`
 * est téléchargé et appliqué (badge « Pré-entraîné ») ; en cas d'échec, des
 * poids déterministes (graine fixe) sont utilisés (badge « Poids déterministes »)
 * et la page reste entièrement fonctionnelle. Le manifeste ne s'applique qu'avant
 * la première run et jamais après une édition manuelle : une réponse tardive
 * n'écrase jamais des poids affichés.
 *
 * Entraînement : « Entraîner 1 époque » charge TensorFlow.js paresseusement
 * (aucun code TFJS avant ce clic), télécharge les jeux IDX locaux, sélectionne
 * deux sous-ensembles stratifiés (curseurs de pourcentage) et entraîne le modèle
 * par lots bornés. Chaque run repart toujours des poids affichés — pré-entraînés,
 * aléatoires, édités à la main ou issus d'une run précédente. « Entraîner en
 * boucle » enchaîne les époques jusqu'à « Pause ». « Pause » arrête la boucle au
 * lot suivant ; « Réinitialiser les poids » restaure les poids initiaux ;
 * « Poids aléatoires » tire une graine nouvelle et remplace les poids affichés.
 * Les poids d'inférence TypeScript sont rafraîchis depuis le modèle au plus
 * toutes les 250 ms, et toujours à la pause et en fin de course. Les courbes
 * affichent la perte d'entraînement et deux précisions — d'entraînement et de
 * validation — sur une échelle partagée, un point par époque achevée.
 *
 * Mise en page : la colonne de gauche réunit le dessin, la galerie et le panneau
 * d'entraînement ; la colonne de droite réunit les probabilités par classe et les
 * deux courbes. Les curseurs de nombre de filtres vivent dans une section
 * repliable au-dessus des panneaux de filtres et de cartes de caractéristiques,
 * qui passent côte à côte au-delà de 920 px.
 *
 * La machine à états de l'entraînement vit dans `training-state.ts` : toute
 * transition qui met fin à une run fait retomber le badge sur un état inactif.
 * La sélection de l'éditeur de poids — geste, vue des toiles, libellé, valeur du
 * poids, saisie au clavier — vit dans le module pur `weight-selection.ts`, que ce
 * composant réexporte sans le modifier.
 * Badge, origine des poids et ligne d'état sont calculés par le module pur
 * `demo-status.ts` à partir de l'état courant : le badge affiche « Modifié » dès
 * qu'un poids a été édité à la main sans run active.
 * La galerie et les jeux IDX ne sont téléchargés ni à l'affichage ni au
 * chargement : seule la première ouverture du sélecteur d'exemples ou le premier
 * clic sur « Précédent » ou « Suivant » les déclenche. Chaque chiffre propose
 * dix exemples du jeu de test, parcourus avec « Précédent » et « Suivant ».
 *
 * Éditeur de poids : un clic sur une liaison sélectionne le couple (canal d'entrée,
 * canal de sortie) à une frontière entre couches ; un second clic sur la même
 * liaison désélectionne. La toile de filtre du couple montre le noyau 3×3 exact
 * plutôt que la moyenne des canaux ; un clic sur une case de cette toile ouvre
 * le formulaire sur le poids exact : libellé, saisie au clavier pré-remplie,
 * boutons « Appliquer » et « Annuler » — Entrée applique, Échap annule. Une
 * liaison sélectionnée sans case affiche le conseil de geste du module
 * `weight-editor.ts` à la place du formulaire. Les liaisons se pilotent aussi
 * au clavier (Entrée et Espace, avec focus visible) ; les toiles, elles,
 * restent au pointeur.
 */

/** Graine de l'initialisation déterministe des poids et des sélections. */
const SEED = 42;

/** Échelle d'affichage de la surface de dessin : 28 cellules × 10 px = 280 px. */
const DRAW_SCALE = 10;
const INPUT_CANVAS_SIZE = MNIST_INPUT_SIZE * DRAW_SCALE;

/** Toiles de filtres : 3 × 3 cellules à 20 px. */
const FILTER_CELL_SIZE = 20;
const FILTER_CANVAS_SIZE = CONV_KERNEL_SIZE * FILTER_CELL_SIZE;

/** Largeur maximale d'une toile de carte de caractéristiques (les tailles
 * réelles 26, 13, 11 et 5 sont dérivées du noyau). */
const MAP_CANVAS_MAX = 192;

/** Rayon du pinceau de dessin, en cellules 28×28. */
const BRUSH_RADIUS = 1.6;

/** Retrait de la zone de clic transparente à chaque bout d'une liaison : la
 * ligne visible garde ses extrémités sur les toiles, la zone de clic s'arrête
 * avant leur bord pour ne pas capter les clics sur les toiles elles-mêmes. */
const HIT_LINE_INSET = 6;

const WHITE_RGB: readonly [number, number, number] = [255, 255, 255];

/** Taille des lots par défaut, même valeur que `tf-playground/mnist/train.py`. */
const DEFAULT_BATCH_SIZE = 64;

/** Bornes du curseur de taille des lots (pas de 8). */
const MIN_BATCH_SIZE = 8;
const MAX_BATCH_SIZE = 256;

/** Nombre d'époques par entraînement. */
const TRAIN_EPOCHS = 1;

/** Époques d'un entraînement en boucle : assez grand pour ne jamais s'arrêter seul. */
const TRAIN_LOOP_EPOCHS = Number.MAX_SAFE_INTEGER;

/** Effectifs officiels des jeux IDX, pour estimer les sélections avant téléchargement. */
const MNIST_TRAIN_SPLIT_SIZE = 60000;
const MNIST_TEST_SPLIT_SIZE = 10000;

/** Bornes des curseurs de pourcentage (1 à 100, pas de 1). */
const MIN_PERCENT = 1;
const MAX_PERCENT = 100;

/** Bornes du curseur de taux d'apprentissage (pas de 0,0005). */
const MIN_LEARNING_RATE = 0.0005;
const MAX_LEARNING_RATE = 0.02;
const DEFAULT_LEARNING_RATE = 0.003;

const MAP_CELL_SIZES: readonly number[] = FEATURE_MAP_SIZES.map((size) =>
  Math.max(4, Math.floor(MAP_CANVAS_MAX / size)),
);

/** Dimensions des toiles de cartes de caractéristiques, par couche. */
const MAP_CANVAS_SIZES: readonly number[] = FEATURE_MAP_SIZES.map(
  (size, layer) => size * MAP_CELL_SIZES[layer],
);

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Force du pinceau à une distance `distance` du centre du tracé (1 au centre). */
export function brushStrength(distance: number): number {
  return distance < BRUSH_RADIUS ? 1 - distance / BRUSH_RADIUS : 0;
}

/**
 * Peint un disque flou dans `pixels` (784 valeurs en [0,1], ligne-major) autour du
 * point `(gx, gy)` en coordonnées 28×28. La valeur max est conservée : un passage
 * plus sombre ne peut pas effacer un pixel déjà foncé.
 */
export function paintDisc(pixels: Float32Array, gx: number, gy: number): void {
  const minX = Math.max(0, Math.floor(gx - BRUSH_RADIUS));
  const maxX = Math.min(MNIST_INPUT_SIZE - 1, Math.ceil(gx + BRUSH_RADIUS));
  const minY = Math.max(0, Math.floor(gy - BRUSH_RADIUS));
  const maxY = Math.min(MNIST_INPUT_SIZE - 1, Math.ceil(gy + BRUSH_RADIUS));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const strength = brushStrength(Math.hypot(x + 0.5 - gx, y + 0.5 - gy));
      if (strength <= 0) {
        continue;
      }
      const index = y * MNIST_INPUT_SIZE + x;
      if (strength > pixels[index]) {
        pixels[index] = strength;
      }
    }
  }
}

/**
 * Convertit un événement pointeur en coordonnées de grille `size`×`size`, ou null
 * si la toile est vide. Les coordonnées sont fractionnaires et peuvent dépasser
 * [0, `size`] : le bornage est laissé à l'appelant.
 */
export function pointerToGrid(
  canvas: HTMLCanvasElement,
  event: PointerEvent,
  size: number,
): [number, number] | null {
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) {
    return null;
  }
  const gx = ((event.clientX - rect.left) / rect.width) * size;
  const gy = ((event.clientY - rect.top) / rect.height) * size;
  return [gx, gy];
}

/** Convertit un événement pointeur en coordonnées 28×28, ou null si la toile est vide. */
export function pointerToCell(
  canvas: HTMLCanvasElement,
  event: PointerEvent,
): [number, number] | null {
  return pointerToGrid(canvas, event, MNIST_INPUT_SIZE);
}

/**
 * Noyau 3×3 affiché par une toile de filtre pour un mode de vue donné : la tranche
 * du canal d'entrée pour `kernel`, la moyenne des canaux d'entrée pour `mean`. Un
 * canal d'entrée hors bornes — sélection plus ancienne que l'architecture
 * courante — retombe sur la moyenne.
 */
export function filterKernelValues(
  view: FilterView,
  kernel: Float32Array,
  inputChannels: number,
  outputChannel: number,
): Float32Array {
  const outputChannels = kernel.length / (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * inputChannels);
  const sliced =
    view.mode !== 'mean' &&
    Number.isInteger(outputChannels) &&
    outputChannels > 0 &&
    Number.isInteger(outputChannel) &&
    outputChannel >= 0 &&
    outputChannel < outputChannels &&
    view.inputChannel >= 0 &&
    view.inputChannel < inputChannels;
  return sliced
    ? filterKernelSlice(kernel, inputChannels, outputChannel, view.inputChannel)
    : filterKernelMean(kernel, inputChannels, outputChannel);
}

/**
 * État d'une ligne de liaison : `selected` pour la liaison sélectionnée (une
 * sélection de poids l'est aussi, sur la même frontière et les mêmes canaux),
 * `dimmed` pour les autres lignes dès qu'une sélection existe, `normal` quand
 * rien n'est sélectionné.
 */
export function linkLineState(
  selection: WeightSelection,
  boundary: number,
  inputChannel: number,
  outputChannel: number,
): 'selected' | 'dimmed' | 'normal' {
  if (selection === null) {
    return 'normal';
  }
  if (
    (selection.kind === 'link' || selection.kind === 'weight') &&
    selection.boundary === boundary &&
    selection.inputChannel === inputChannel &&
    selection.outputChannel === outputChannel
  ) {
    return 'selected';
  }
  return 'dimmed';
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
 * Libellé français d'une ligne de liaison : le nom du couple (couche, canal
 * d'entrée, canal de sortie) quand rien ne la restreint, suivi du nombre de poids
 * qui comptent quand une sélection la restreint — le poids unique d'une sélection
 * de poids.
 */
export function describeLinkLabel(
  selection: WeightSelection,
  boundary: number,
  inputChannel: number,
  outputChannel: number,
  filters: MnistCnnFilters,
): string {
  const name = `conv${boundary + 1} — entrée ${inputChannel + 1}, sortie ${outputChannel + 1}`;
  if (selection === null || !lineInArchitecture(boundary, inputChannel, outputChannel, filters)) {
    return name;
  }
  if (
    selection.kind === 'weight' &&
    selection.boundary === boundary &&
    selection.inputChannel === inputChannel &&
    selection.outputChannel === outputChannel
  ) {
    return `${name} · 1 poids sur ${CONV_KERNEL_SIZE * CONV_KERNEL_SIZE}`;
  }
  return name;
}

/**
 * Sous-titre du panneau des filtres : la moyenne des canaux d'entrée pour
 * chaque canal de sortie. Dès qu'un couple est sélectionné, sa toile de
 * filtre aval montre le noyau 3×3 exact plutôt que la moyenne : une phrase
 * le signale, sinon l'affichage contredirait le sous-titre.
 */
export function filterPanelSubtitle(selection: WeightSelection): string {
  const mean = `Moyenne des canaux d'entrée pour chaque canal de sortie ; bleu = positif, violet = négatif.`;
  return selection === null
    ? mean
    : `${mean} La toile du couple sélectionné montre le noyau 3×3 exact.`;
}

/**
 * Sélecteurs des deux toiles de filtres reliées par une liaison : la toile amont
 * est le canal de sortie `inputChannel` de la couche précédente, la toile aval le
 * canal d'entrée `outputChannel` de la couche de la frontière.
 * `filterCanvasSelector` lève une `RangeError` pour des indices négatifs : la
 * frontière 0, sans couche amont, n'énumère aucune liaison, et c'est
 * `linkGroupIndices` qui la refuse avant tout appel.
 */
export function linkCanvasSelectors(link: LinkEndpoint): { source: string; target: string } {
  return {
    source: filterCanvasSelector(link.boundary - 1, link.inputChannel),
    target: filterCanvasSelector(link.boundary, link.outputChannel),
  };
}

/**
 * Indices de la liaison portée par un `<g>` de ligne, lus depuis ses attributs de
 * données, ou null s'ils ne désignent pas une frontière et deux canaux valides.
 * Une frontière plus basse que 1 est refusée : la frontière 0 relie l'image
 * d'entrée à `conv1` et n'a pas de toile amont, donc `linkCanvasSelectors` lèverait
 * une `RangeError` en dehors de toute protection. `filterCanvasSelector` lève
 * aussi pour des indices négatifs ou non entiers, d'où ce filtrage préalable.
 */
export function linkGroupIndices(group: Element): LinkEndpoint | null {
  const read = (name: string): number | null => {
    const raw = group.getAttribute(name);
    if (raw === null || raw === '') {
      return null;
    }
    const value = Number(raw);
    return Number.isInteger(value) && value >= 0 ? value : null;
  };
  const boundary = read('data-boundary');
  const inputChannel = read('data-input-channel');
  const outputChannel = read('data-output-channel');
  if (boundary === null || boundary < 1 || inputChannel === null || outputChannel === null) {
    return null;
  }
  return { boundary, inputChannel, outputChannel };
}

/**
 * Segment d'une liaison entre deux toiles de filtres, exprimé en coordonnées
 * relatives au panneau, ou null si l'un des rectangles n'est pas utilisable :
 * la ligne est alors laissée en l'état plutôt qu tracée à une position absurde.
 * Ne lève jamais, y compris si `linkGeometry` refuse un rectangle.
 */
export function filterLinkGeometry(
  source: RectLike,
  target: RectLike,
  container: RectLike,
): LineGeometry | null {
  if (!isUsableRect(source) || !isUsableRect(target) || !isUsableRect(container)) {
    return null;
  }
  try {
    return linkGeometry(source, target, container);
  } catch {
    return null;
  }
}

/** Index du maximum d'un tableau (premier index en cas d'égalité). */
export function argmaxIndex(values: Float32Array): number {
  let best = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      best = index;
    }
  }
  return best;
}

export function hexToRgb(hex: string): readonly [number, number, number] {
  const match = /^#([0-9a-f]{6})$/i.exec(hex.trim());
  if (!match) {
    return [37, 99, 235];
  }
  const value = Number.parseInt(match[1], 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export function mixRgb(
  from: readonly [number, number, number],
  to: readonly [number, number, number],
  t: number,
): readonly [number, number, number] {
  return [
    Math.round(from[0] + (to[0] - from[0]) * t),
    Math.round(from[1] + (to[1] - from[1]) * t),
    Math.round(from[2] + (to[2] - from[2]) * t),
  ];
}

/**
 * Estimation déterministe du nombre d'exemples d'une sélection stratifiée (voir
 * `selectStratifiedIndices`) pour un jeu de `total` exemples répartis en dix
 * classes équilibrées : chaque classe fournit `max(1, round(total ÷ 10 × fraction))`
 * exemples. Sert à afficher les effectifs avant le téléchargement des jeux IDX.
 */
export function stratifiedExampleCountEstimate(total: number, fraction: number): number {
  if (!Number.isInteger(total) || total < MNIST_CLASS_COUNT) {
    throw new RangeError(
      `stratifiedExampleCountEstimate: total must be an integer of at least ${MNIST_CLASS_COUNT}, received ${String(total)}.`,
    );
  }
  if (typeof fraction !== 'number' || !Number.isFinite(fraction) || fraction <= 0 || fraction > 1) {
    throw new RangeError(
      `stratifiedExampleCountEstimate: fraction must be a finite number in (0, 1], received ${String(fraction)}.`,
    );
  }
  const perClass = total / MNIST_CLASS_COUNT;
  return MNIST_CLASS_COUNT * Math.max(1, Math.round(perClass * fraction));
}

/** Vrai si deux tuples de filtres décrivent la même architecture. */
function sameFilters(a: MnistCnnFilters, b: MnistCnnFilters): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
}

/** Données d'entrée d'une requête de run d'entraînement. */
export interface TrainingRunInput {
  readonly filters: MnistCnnFilters;
  readonly learningRate: number;
  readonly batchSize: number;
  readonly trainPercent: number;
  readonly validationPercent: number;
  readonly epochs: number;
  readonly weights: MnistCnnWeights;
}

/**
 * Construit la requête de run soumise à `MnistTrainingController.startRun` :
 * `initialWeights` porte toujours les poids affichés, donc chaque run repart des
 * poids affichés, qu'ils soient pré-entraînés, aléatoires, édités à la main ou
 * issus d'une run précédente.
 */
export function trainingRunRequest(input: TrainingRunInput): MnistRunRequest {
  return {
    filters: input.filters,
    learningRate: input.learningRate,
    batchSize: input.batchSize,
    seed: SEED,
    epochs: input.epochs,
    trainPercent: input.trainPercent,
    validationPercent: input.validationPercent,
    initialWeights: input.weights,
  };
}

/**
 * Fait avancer l'index d'exemple de la galerie avec bouclage : `delta` `1` passe
 * à l'exemple suivant, `-1` au précédent, et les deux bouts se rejoignent.
 * Retourne `0` quand `total` n'est pas positif.
 */
export function stepGalleryIndex(index: number, total: number, delta: number): number {
  if (total <= 0) {
    return 0;
  }
  return (((index + delta) % total) + total) % total;
}

@customElement('drl-mnist-cnn')
export class MnistCnn extends LitElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: block;
        color: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: var(--drl-font-size);
      }

      .drl-header {
        margin-bottom: var(--drl-space-3);
      }

      .drl-content {
        margin-top: var(--drl-space-3);
      }

      .drl-status {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-2);
        min-height: 1.5rem;
      }

      .drl-badge {
        background: var(--drl-accent);
        color: var(--drl-surface);
        border-radius: calc(var(--drl-radius) / 2);
        padding: 0 var(--drl-space-2);
        font-size: 0.85em;
      }

      .drl-status-text {
        color: var(--drl-muted);
      }

      .drl-mnist-layout {
        display: grid;
        grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
        gap: var(--drl-space-3);
      }

      @media (max-width: 920px) {
        .drl-mnist-layout {
          grid-template-columns: minmax(0, 1fr);
        }
      }

      .drl-mnist-left {
        display: grid;
        gap: var(--drl-space-3);
        align-content: start;
        min-width: 0;
      }

      .drl-filter-counts {
        margin-top: var(--drl-space-3);
        padding: var(--drl-space-3);
        background: var(--drl-bg);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
      }

      .drl-filter-counts > summary {
        display: flex;
        align-items: center;
        gap: var(--drl-space-2);
        cursor: pointer;
        list-style: none;
        padding: 0;
      }

      .drl-filter-counts > summary::-webkit-details-marker {
        display: none;
      }

      .drl-filter-counts > summary::before {
        content: '▸';
        display: inline-block;
        color: var(--drl-muted);
        transition: transform 120ms ease;
      }

      .drl-filter-counts[open] > summary {
        margin-bottom: var(--drl-space-2);
      }

      .drl-filter-counts[open] > summary::before {
        transform: rotate(90deg);
      }

      .drl-filter-counts-body {
        display: grid;
        gap: var(--drl-space-2);
        align-content: start;
      }

      .drl-panels-row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: var(--drl-space-3);
        margin-top: var(--drl-space-3);
      }

      @media (max-width: 920px) {
        .drl-panels-row {
          grid-template-columns: minmax(0, 1fr);
        }
      }

      .drl-panel {
        display: grid;
        gap: var(--drl-space-2);
        align-content: start;
        padding: var(--drl-space-3);
        background: var(--drl-bg);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
      }

      .drl-panel--filters {
        position: relative;
      }

      .drl-panel-title {
        margin: 0;
        font-size: 0.95em;
      }

      .drl-panel-subtitle {
        margin: 0;
        color: var(--drl-muted);
        font-size: 0.85em;
      }

      canvas.drl-canvas {
        display: block;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
      }

      .drl-canvas--input {
        width: 100%;
        max-width: ${INPUT_CANVAS_SIZE}px;
        height: auto;
        cursor: crosshair;
        touch-action: none;
      }

      .drl-canvas--filter {
        width: auto;
        height: auto;
      }

      /* La toile fait la taille réelle de la carte en pixels :
         image-rendering pixelated garde les cellules nettes. */
      .drl-canvas--map {
        image-rendering: pixelated;
      }

      .drl-hint {
        margin: 0;
        color: var(--drl-muted);
        font-size: 0.85em;
      }

      /* Éditeur de poids : ligne compacte (libellé, saisie, deux boutons), puis
          ligne d'erreur à l'accent secondaire. */
      .drl-weight-editor {
        display: grid;
        gap: var(--drl-space-1);
      }

      .drl-weight-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-2);
      }

      .drl-weight-label {
        font-size: 0.85em;
      }

      .drl-weight-input {
        font: inherit;
        width: 8em;
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-bg);
        color: var(--drl-fg);
        font-variant-numeric: tabular-nums;
      }

      .drl-weight-error {
        margin: 0;
        color: var(--drl-accent-2);
        font-size: 0.85em;
      }

      .drl-field {
        display: grid;
        gap: var(--drl-space-1);
      }

      .drl-field-label {
        color: var(--drl-muted);
        font-size: 0.85em;
      }

      select {
        font: inherit;
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-bg);
        color: var(--drl-fg);
      }

      .drl-toolbar {
        display: flex;
        flex-wrap: wrap;
        gap: var(--drl-space-2);
      }

      .drl-button {
        font: inherit;
        display: inline-flex;
        align-items: center;
        gap: var(--drl-space-1);
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-bg);
        color: var(--drl-fg);
        cursor: pointer;
      }

      .drl-button:hover:not(:disabled) {
        border-color: var(--drl-accent);
      }

      .drl-button:disabled {
        color: var(--drl-muted);
        cursor: not-allowed;
      }

      .drl-button.drl-primary {
        background: var(--drl-accent);
        border-color: var(--drl-accent);
        color: var(--drl-surface);
      }

      .drl-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, minmax(0, 1fr)));
        gap: var(--drl-space-2);
      }

      .drl-charts {
        display: grid;
        gap: var(--drl-space-2);
      }

      .drl-chart {
        display: grid;
        gap: var(--drl-space-1);
        margin: 0;
      }

      .drl-chart-title {
        margin: 0;
        color: var(--drl-muted);
        font-size: 0.8em;
      }

      .drl-chart-svg {
        display: block;
        width: 100%;
        height: auto;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
      }

      .drl-chart-gridline {
        stroke: var(--drl-border);
        stroke-width: 1;
      }

      .drl-chart-tick {
        fill: var(--drl-muted);
        font-size: 8px;
      }

      .drl-chart-axis {
        stroke: var(--drl-border);
        stroke-width: 1;
      }

      .drl-chart-line--loss {
        stroke: var(--drl-accent);
        stroke-width: 2;
        fill: none;
      }

      .drl-chart-line--accuracy {
        stroke: var(--drl-accent-2);
        stroke-width: 2;
        fill: none;
      }

      .drl-chart-empty {
        fill: var(--drl-muted);
        font-size: 9px;
      }

      /* Précision d'entraînement : pointillés, à distinguer de la précision de
         validation (trait plein) sans la couleur. */
      .drl-chart-line--accuracy-train {
        stroke: var(--drl-accent);
        stroke-width: 2;
        stroke-dasharray: 4 3;
        fill: none;
      }

      .drl-chart-legend {
        display: flex;
        flex-wrap: wrap;
        gap: var(--drl-space-2);
        margin: 0;
        padding: 0;
        list-style: none;
        color: var(--drl-muted);
        font-size: 0.75em;
      }

      .drl-chart-legend-item {
        display: inline-flex;
        align-items: center;
        gap: var(--drl-space-1);
      }

      .drl-chart-legend-swatch {
        display: block;
        overflow: visible;
      }

      .drl-chart-legend-line {
        stroke-width: 2;
      }

      .drl-gallery-counter {
        font-variant-numeric: tabular-nums;
        color: var(--drl-muted);
        font-size: 0.85em;
      }

      .drl-stat {
        background: var(--drl-bg);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        padding: var(--drl-space-2);
      }

      .drl-stat-label {
        margin: 0 0 2px;
        color: var(--drl-muted);
        font-size: 0.8em;
      }

      .drl-stat-value {
        margin: 0;
        font-variant-numeric: tabular-nums;
      }

      .drl-probs {
        display: grid;
        gap: var(--drl-space-1);
      }

      .drl-prob-row {
        display: grid;
        grid-template-columns: 1.4em minmax(0, 1fr) 4.5em;
        align-items: center;
        gap: var(--drl-space-2);
      }

      .drl-prob-digit {
        font-weight: 600;
      }

      .drl-prob-bar-wrap {
        height: 0.8em;
        overflow: hidden;
        background: var(--drl-bg);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
      }

      .drl-prob-bar {
        display: block;
        height: 100%;
        background: var(--drl-accent);
      }

      .drl-prob-row--max .drl-prob-bar {
        background: var(--drl-accent-2);
      }

      .drl-prob-value {
        color: var(--drl-muted);
        font-size: 0.85em;
        font-variant-numeric: tabular-nums;
        text-align: right;
      }

      .drl-sliders {
        display: grid;
        gap: var(--drl-space-1);
      }

      .drl-layer-block {
        display: grid;
        gap: var(--drl-space-1);
      }

      .drl-layer-title {
        margin: 0;
        color: var(--drl-muted);
        font-size: 0.85em;
      }

      .drl-cells-grid {
        display: flex;
        flex-wrap: wrap;
        gap: var(--drl-space-1);
      }

      .drl-cell {
        display: grid;
        gap: 2px;
        justify-items: center;
        margin: 0;
      }

      .drl-cell figcaption {
        color: var(--drl-muted);
        font-size: 0.75em;
      }

      /* Lignes de liaison entre toiles de filtres : le SVG couvre tout le panneau
         mais ne capture les clics que sur les lignes, pour ne jamais masquer une
         toile ni la surface de dessin. */
      .drl-filter-links {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        overflow: visible;
        pointer-events: none;
      }

      .drl-filter-link-group {
        pointer-events: auto;
        cursor: pointer;
      }

      .drl-filter-link {
        fill: none;
        stroke: var(--drl-muted);
        stroke-width: 1;
        opacity: 0.7;
        transition:
          stroke 120ms ease,
          stroke-width 120ms ease,
          opacity 120ms ease;
      }

      /* Sélection : lisible sans concurrencer les toiles. */
      .drl-filter-link--selected {
        stroke: var(--drl-accent);
        stroke-width: 2;
        opacity: 1;
      }

      /* Une sélection atténue toutes les autres lignes. */
      .drl-filter-link--dimmed {
        stroke: var(--drl-muted);
        stroke-width: 1;
        opacity: 0.35;
      }

      /* Survol : la ligne prend l'accent secondaire, y compris si elle est
         atténue — c'est l'état « highlighted ». */
      .drl-filter-link-group:hover .drl-filter-link--normal,
      .drl-filter-link-group:hover .drl-filter-link--dimmed {
        stroke: var(--drl-accent-2);
        stroke-width: 1.5;
        opacity: 0.85;
      }

      /* Focus clavier : la ligne visible du groupe prend l'accent secondaire,
          sans cadre autour du groupe SVG. */
      .drl-filter-link-group:focus-visible .drl-filter-link {
        stroke: var(--drl-accent-2);
        stroke-width: 2;
        opacity: 1;
      }

      /* Zone de clic large mais transparente : le contour du segment reste la
         seule partie sensible, jamais le panneau entier. */
      .drl-filter-link-hit {
        fill: none;
        stroke: transparent;
        stroke-width: 10;
        pointer-events: auto;
      }
    `,
  ];

  @state()
  private filterCounts: MnistCnnFilters = DEFAULT_FILTER_COUNTS;

  @state()
  private weights: MnistCnnWeights = initializeMnistCnnWeights(
    { filters: DEFAULT_FILTER_COUNTS },
    SEED,
  );

  @state()
  private result: MnistCnnForwardResult | null = null;

  @state()
  private inputSource = 'Dessin';

  private drawing = false;

  /** Manifeste pré-entraîné chargé (null avant la fin du téléchargement). */
  @state()
  private pretrainedAssets: MnistCnnAssets | null = null;

  /** Raison de l'échec du manifeste, affichée dans la ligne d'état. */
  @state()
  private manifestError: string | null = null;

  /** Galerie de dix exemples par chiffre, extraits du jeu de test (vide avant la fin du chargement). */
  @state()
  private gallerySample: MnistSample | null = null;

  /** Chiffre affiché par la galerie, ou null avant le premier chargement. */
  @state()
  private galleryClass: number | null = null;

  /** Index de l'exemple courant dans la classe affichée. */
  @state()
  private galleryIndex = 0;

  @state()
  private galleryLoading = false;

  /** Raison de l'échec du chargement de la galerie, affichée sous le sélecteur. */
  @state()
  private galleryError: string | null = null;

  /** Libellé du chiffre attendu quand l'entrée est un exemple de la galerie. */
  private galleryLabel: number | null = null;

  @state()
  private trainingState: TrainingState = 'deterministic';

  /** Dernière erreur d'entraînement, affichée dans le panneau d'entraînement. */
  @state()
  private trainingError: string | null = null;

  @state()
  private batchesProcessed = 0;

  @state()
  private loss: number | null = null;

  @state()
  private validationAccuracy: number | null = null;

  @state()
  private elapsedMs = 0;

  /** Historique des points de courbe, un par époque achevée (plafonné à 50). */
  @state()
  private history: TrainingHistory = emptyHistory();

  /** Dernière époque déjà reportée dans l'historique des courbes. */
  private lastEmittedEpoch = 0;

  /** Vrai entre le clic sur un bouton d'entraînement et la fin de la préparation. */
  private startingTraining = false;

  /** Vrai dès le premier évènement de progression : les métriques sont affichées. */
  @state()
  private metricsStarted = false;

  @state()
  private trainPercent = 10;

  @state()
  private validationPercent = 10;

  @state()
  private learningRate = DEFAULT_LEARNING_RATE;

  /** Taille des lots de la prochaine run, de 8 à 256. */
  @state()
  private batchSize = DEFAULT_BATCH_SIZE;

  @state()
  private trainExampleCount = stratifiedExampleCountEstimate(MNIST_TRAIN_SPLIT_SIZE, 10 / 100);

  @state()
  private validationExampleCount = stratifiedExampleCountEstimate(MNIST_TEST_SPLIT_SIZE, 10 / 100);

  /** Dessin des toiles reporté après la prochaine mise à jour du DOM Lit. */
  private canvasesDirty = true;

  /** Observe le panneau des filtres pour repositionner lignes et toiles ensemble. */
  private filterPanelObserver: ResizeObserver | null = null;

  /** Sélection de l'éditeur de poids : liaison ou poids exact. */
  @state()
  private selection: WeightSelection = null;

  /** Vrai tant qu'un poids a été édité à la main depuis le dernier remplacement. */
  @state()
  private weightsEdited = false;

  /** Valeur saisie au clavier pour le poids sélectionné (remplie par la suite). */
  @state()
  private weightDraft = '';

  /** Erreur de saisie du poids, ou null (remplie par la suite). */
  @state()
  private weightError: string | null = null;

  /**
   * Chaîne de notifications d'édition de poids : deux applications rapprochées
   * pendant une run active sont sérialisées, car `notifyWeightsEdited` attend
   * l'arrêt de l'entraîneur — sans file d'attente, la seconde édition serait
   * perdue par la course. La chaîne reste vivante après un échec.
   */
  private weightsNotifyChain: Promise<void> = Promise.resolve();

  /** Image courante : 784 valeurs d'encre en [0,1], ligne-major. */
  private readonly inputPixels = new Float32Array(MNIST_INPUT_LENGTH);

  /** Canvas hors-écran de 28×28 réutilisé pour l'image d'entrée. */
  private readonly inputCells: HTMLCanvasElement = document.createElement('canvas');

  /** Canvas hors-écran réutilisé pour les petits rendus mis à l'échelle. */
  private readonly scratch: HTMLCanvasElement = document.createElement('canvas');

  private paletteCache: {
    positive: readonly [number, number, number];
    negative: readonly [number, number, number];
  } | null = null;

  /** Contrôleur du cycle de vie de l'entraînement (pur, sans DOM ni TFJS). */
  private readonly controller = new MnistTrainingController({
    loadSplits: async () => {
      const splits = await loadMnistSplits();
      this.splits = splits;
      return splits;
    },
    createTrainer: (config) => createMnistCnnTfjsTrainer(config),
    createSource: (dataset, indices) => createIdxTrainSource(dataset, indices),
    selectIndices: (dataset, fraction, seed) =>
      selectStratifiedIndices(dataset, { fraction, seed }),
    onStateChanged: (state) => {
      this.trainingState = state;
    },
    onProgress: (progress) => {
      this.applyProgress(progress);
    },
    onWeightsChanged: (weights) => {
      // Poids remplacés par l'entraînement : la sélection ne désigne plus rien.
      this.clearSelection();
      this.weights = weights;
      this.runInference();
    },
    onWeightsReset: (weights) => {
      // Une édition manuelle a déjà calculé le passage avant sur ce même
      // objet : relancer l'inférence ici doublerait le travail sans changer
      // le résultat. Les réinitialisations et les poids aléatoires passent
      // un objet neuf, donc recalculent bien l'inférence.
      if (weights !== this.weights) {
        this.weights = weights;
        this.runInference();
      }
    },
    onError: (message) => {
      this.trainingError = message;
    },
    onCountsChanged: (trainCount, validationCount) => {
      this.trainExampleCount = trainCount;
      this.validationExampleCount = validationCount;
    },
  });

  /** Jeux IDX chargés (mis en cache par le port `loadSplits`), ou null. */
  private splits: Awaited<ReturnType<typeof loadMnistSplits>> | null = null;

  private get predictedClass(): number | null {
    return this.result ? argmaxIndex(this.result.probabilities) : null;
  }

  /**
   * État courant assemblé pour le module pur `demo-status.ts` : le badge,
   * l'origine des poids et la ligne d'état en sont déduits, sans logique
   * propre au composant.
   */
  private get demoStatus(): DemoStatusInput {
    return {
      state: this.trainingState,
      weightsEdited: this.weightsEdited,
      manifestError: this.manifestError,
      batchesProcessed: this.batchesProcessed,
      loss: this.loss,
      validationAccuracy: this.validationAccuracy,
      elapsedMs: this.elapsedMs,
      metricsStarted: this.metricsStarted,
      trainExampleCount: this.trainExampleCount,
      validationExampleCount: this.validationExampleCount,
      parameters: mnistCnnParameterCount(this.filterCounts),
      predictedClass: this.predictedClass,
      inputSource: this.inputSource,
    };
  }

  /** Vrai si le manifeste chargé correspond à l'architecture courante. */
  private get manifestMatches(): boolean {
    return (
      this.pretrainedAssets !== null &&
      sameFilters(this.pretrainedAssets.config.filters, this.filterCounts)
    );
  }

  private get inputHint(): string {
    if (this.galleryLabel === null) {
      return "Dessinez un chiffre à la souris ou au doigt ; l'inférence se met à jour pendant le tracé.";
    }
    return `Exemple de la galerie MNIST : le chiffre attendu est un ${this.galleryLabel}.`;
  }

  /** Nombre d'exemples proposés par chiffre dans la galerie chargée. */
  private get galleryPerClass(): number {
    return this.gallerySample === null ? 0 : this.gallerySample.count / MNIST_CLASS_COUNT;
  }

  /** Vrai quand la galerie est chargée et qu'un chiffre est affiché. */
  private get galleryReady(): boolean {
    return this.gallerySample !== null && this.galleryClass !== null;
  }

  /** Compteur affiché sous le sélecteur : `n / 10`, ou `–` avant le chargement. */
  private get galleryCounterText(): string {
    if (!this.galleryReady) {
      return '–';
    }
    return `${this.galleryIndex + 1} / ${this.galleryPerClass}`;
  }

  /** Courbe de perte d'entraînement, un point par époque achevée. */
  private get lossLayout(): ChartLayout {
    return layoutChart("Perte d'entraînement", lossPoints(this.history), CHART_BOX, (value) =>
      formatValue(value, 3),
    );
  }

  /**
   * Courbe de précision : entraînement et validation partagent la même échelle
   * verticale, un point par époque achevée. La distinction sans couleur repose
   * sur le motif (pointillés pour l'entraînement, trait plein pour la validation)
   * et sur la légende textuelle.
   */
  private get accuracyLayout(): MultiChartLayout {
    const percent = (value: number): string => `${formatValue(value * 100, 1)} %`;
    const series: ChartSeries[] = [
      {
        name: 'Entraînement',
        points: trainAccuracyPoints(this.history),
        lineClass: 'drl-chart-line--accuracy-train',
        format: percent,
      },
      {
        name: 'Validation',
        points: accuracyPoints(this.history),
        lineClass: 'drl-chart-line--accuracy',
        format: percent,
      },
    ];
    return layoutMultiChart('Précision (entraînement et validation)', series, CHART_BOX);
  }

  /**
   * Pixels de l'exemple courant de la galerie : une vue sur
   * `gallerySample.images`, sans copie. Retourne `null` avant le chargement.
   */
  private currentGalleryPixels(): Float32Array | null {
    const sample = this.gallerySample;
    const classIndex = this.galleryClass;
    if (sample === null || classIndex === null) {
      return null;
    }
    const perClass = this.galleryPerClass;
    const offset = (classIndex * perClass + this.galleryIndex) * MNIST_INPUT_LENGTH;
    return sample.images.subarray(offset, offset + MNIST_INPUT_LENGTH);
  }

  /**
   * Charge l'exemple courant de la galerie dans le tampon de dessin : le
   * libellé attendu devient le chiffre affiché et l'inférence repart.
   */
  private applyGalleryExample(): void {
    const pixels = this.currentGalleryPixels();
    const classIndex = this.galleryClass;
    if (pixels === null || classIndex === null) {
      return;
    }
    this.inputPixels.set(pixels);
    this.galleryLabel = classIndex;
    this.inputSource = `Exemple MNIST (chiffre ${classIndex})`;
    this.runInference();
  }

  protected override firstUpdated(): void {
    this.runInference();
    void this.loadManifest();
    const panel = this.renderRoot.querySelector<HTMLElement>('.drl-panel--filters');
    // Le redimensionnement du panneau déplace les toiles : un `ResizeObserver`
    // repositionne les lignes, tracées en pixels relatifs au SVG qui couvre le
    // panneau.
    if (panel && typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(() => {
        this.positionFilterLinks();
      });
      observer.observe(panel);
      this.filterPanelObserver = observer;
    }
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.controller.notifyConnected();
  }

  override disconnectedCallback(): void {
    this.filterPanelObserver?.disconnect();
    this.filterPanelObserver = null;
    this.resetMetrics();
    super.disconnectedCallback();
    void this.controller.notifyDisconnected();
  }

  /**
   * Positionne les lignes de liaison : pour chaque `<g>`, lit les rectangles de la
   * toile amont et de la toile aval, relatifs au SVG, puis écrit `x1`, `y1`,
   * `x2`, `y2` sur ses deux lignes. Une ligne dont un rectangle n'est pas
   * utilisable — toile absente, panneau masqué — est laissée en l'état. Ne lève
   * jamais : le rendu ne dépend pas de cette méthode.
   *
   * Le conteneur est le SVG, pas le panneau : le SVG est positionné sur la boîte
   * de remplissage du panneau, pas sur sa boîte de bordure. Prendre le rectangle
   * du panneau décalerait chaque ligne de la largeur de sa bordure.
   */
  private positionFilterLinks(): void {
    const panel = this.renderRoot.querySelector<HTMLElement>('.drl-panel--filters');
    const svg = panel?.querySelector<SVGSVGElement>('svg.drl-filter-links');
    if (!panel || !svg) {
      return;
    }
    const container = svg.getBoundingClientRect();
    if (!isUsableRect(container)) {
      return;
    }
    for (const group of Array.from(svg.querySelectorAll<SVGGElement>('g.drl-filter-link-group'))) {
      try {
        const indices = linkGroupIndices(group);
        if (!indices) {
          continue;
        }
        // `linkGroupIndices` refuse désormais la frontière 0, donc cette appel ne
        // lève plus ; le `try` garde la promesse « ne lève jamais » de la méthode.
        const selectors = linkCanvasSelectors(indices);
        const source = panel.querySelector<HTMLElement>(selectors.source);
        const target = panel.querySelector<HTMLElement>(selectors.target);
        if (!source || !target) {
          continue;
        }
        const geometry = filterLinkGeometry(
          source.getBoundingClientRect(),
          target.getBoundingClientRect(),
          container,
        );
        if (!geometry) {
          continue;
        }
        for (const line of Array.from(group.querySelectorAll<SVGLineElement>('line'))) {
          // Seule la zone de clic est raccourcie : la ligne visible garde ses
          // extrémités exactes sur les bords des toiles.
          const segment = line.classList.contains('drl-filter-link-hit')
            ? insetSegment(geometry, HIT_LINE_INSET)
            : geometry;
          line.setAttribute('x1', String(segment.x1));
          line.setAttribute('y1', String(segment.y1));
          line.setAttribute('x2', String(segment.x2));
          line.setAttribute('y2', String(segment.y2));
        }
      } catch {
        // Une ligne qui ne peut pas être tracée reste dans son état courant.
      }
    }
  }

  /**
   * Télécharge le manifeste pré-entraîné et l'applique si l'architecture
   * courante correspond. Tout échec laisse des poids déterministes et un
   * message clair : la page reste fonctionnelle. Le manifeste ne s'applique
   * qu'avant la première run : une réponse tardive n'écrase jamais des poids
   * entraînés.
   */
  private async loadManifest(): Promise<void> {
    try {
      const text = await fetchMnistCnnManifest();
      const assets = parseMnistCnnManifestJson(text);
      this.pretrainedAssets = assets;
      this.manifestError = null;
      const matches = sameFilters(assets.config.filters, this.filterCounts);
      this.controller.notifyManifestChanged(matches);
      if (this.trainingState === 'pretrained') {
        this.applyPretrainedWeights();
      }
    } catch (error) {
      this.manifestError = (error as Error).message;
      this.controller.notifyManifestChanged(false);
    }
  }

  /**
   * Applique les poids du manifeste si le tuple de filtres correspond. Une
   * édition manuelle en cours arrête l'application : une réponse tardive du
   * manifeste ne doit jamais écraser des poids modifiés à la main.
   */
  private applyPretrainedWeights(): void {
    if (this.weightsEdited) {
      return;
    }
    if (this.pretrainedAssets === null) {
      return;
    }
    if (!sameFilters(this.pretrainedAssets.config.filters, this.filterCounts)) {
      return;
    }
    this.weights = this.pretrainedAssets.weights;
    this.clearSelection();
    this.runInference();
  }

  /**
   * Charge la galerie de dix exemples par chiffre et les jeux IDX, qui servent
   * aussi à afficher les effectifs exacts des sélections. Le premier exemple du
   * premier chiffre est appliqué à l'entrée. Tout échec laisse le sélecteur avec
   * l'option de recharger et un message : l'inférence reste utilisable.
   */
  private async loadGallery(): Promise<void> {
    if (this.galleryLoading) {
      return;
    }
    this.galleryLoading = true;
    try {
      this.splits = await loadMnistSplits();
      this.gallerySample = await loadMnistGallery();
      this.galleryClass = 0;
      this.galleryIndex = 0;
      this.galleryError = null;
      this.updateExampleCounts();
      this.applyGalleryExample();
    } catch (error) {
      this.gallerySample = null;
      this.galleryClass = null;
      this.galleryError = (error as Error).message;
    } finally {
      this.galleryLoading = false;
    }
  }

  /**
   * Recalcule les effectifs affichés : exactement depuis les jeux IDX quand ils
   * sont chargés, par estimation sur les effectifs officiels sinon.
   */
  private updateExampleCounts(): void {
    const trainFraction = this.trainPercent / 100;
    const validationFraction = this.validationPercent / 100;
    if (this.splits !== null) {
      this.trainExampleCount = selectStratifiedIndices(this.splits.train, {
        fraction: trainFraction,
        seed: SEED,
      }).length;
      this.validationExampleCount = selectStratifiedIndices(this.splits.test, {
        fraction: validationFraction,
        seed: SEED,
      }).length;
    } else {
      this.trainExampleCount = stratifiedExampleCountEstimate(
        MNIST_TRAIN_SPLIT_SIZE,
        trainFraction,
      );
      this.validationExampleCount = stratifiedExampleCountEstimate(
        MNIST_TEST_SPLIT_SIZE,
        validationFraction,
      );
    }
  }

  /** Lance une run depuis les poids affichés, pour le nombre d'époques demandé. */
  private startTraining(epochs: number): void {
    if (this.startingTraining || this.controller.isRunActive()) {
      return;
    }
    this.startingTraining = true;
    this.trainingError = null;
    this.lastEmittedEpoch = 0;
    const request = trainingRunRequest({
      filters: this.filterCounts,
      learningRate: this.learningRate,
      batchSize: this.batchSize,
      trainPercent: this.trainPercent,
      validationPercent: this.validationPercent,
      epochs,
      weights: this.weights,
    });
    void this.controller
      .startRun(request)
      .catch(() => undefined)
      .finally(() => {
        this.startingTraining = false;
      });
  }

  /** Une époque depuis les poids affichés. */
  private readonly handleStartTraining = (): void => {
    this.startTraining(TRAIN_EPOCHS);
  };

  /** Époques en boucle depuis les poids affichés, jusqu'à Pause. */
  private readonly handleStartLoopTraining = (): void => {
    this.startTraining(TRAIN_LOOP_EPOCHS);
  };

  /** Pause : la boucle s'arrête avant le lot suivant et les poids sont figés. */
  private readonly handlePauseTraining = (): void => {
    void this.controller.pause();
  };

  /** Réinitialise les poids : l'entraînement s'arrête et les poids sont restaurés. */
  private readonly handleResetWeights = (): void => {
    this.resetMetrics();
    this.clearSelection();
    void this.controller.resetWeights(this.weightsForCurrentFilters());
  };

  /**
   * Réinitialise les poids avec une graine nouvelle : l'entraînement s'arrête et
   * les poids tirés au hasard remplacent les poids affichés.
   */
  private readonly handleRandomResetWeights = (): void => {
    const seed = Math.floor(Math.random() * 0xffffffff);
    const weights = initializeMnistCnnWeights({ filters: this.filterCounts }, seed);
    this.resetMetrics();
    this.clearSelection();
    void this.controller.resetWeights(weights, 'random');
  };

  /** Poids à afficher pour l'architecture courante : pré-entraînés si possible. */
  private weightsForCurrentFilters(): MnistCnnWeights {
    if (
      this.pretrainedAssets !== null &&
      sameFilters(this.pretrainedAssets.config.filters, this.filterCounts)
    ) {
      return this.pretrainedAssets.weights;
    }
    return initializeMnistCnnWeights({ filters: this.filterCounts }, SEED);
  }

  private resetMetrics(): void {
    this.batchesProcessed = 0;
    this.loss = null;
    this.validationAccuracy = null;
    this.elapsedMs = 0;
    this.metricsStarted = false;
    this.history = emptyHistory();
    this.lastEmittedEpoch = 0;
  }

  private applyProgress(progress: MnistTrainingProgress): void {
    this.batchesProcessed = progress.batchesProcessed;
    this.loss = Number.isFinite(progress.loss) ? progress.loss : null;
    this.validationAccuracy = progress.validationAccuracy;
    this.elapsedMs = progress.elapsedMs;
    this.metricsStarted = true;
    if (progress.completedEpochs > this.lastEmittedEpoch) {
      const point = epochPoint(progress, this.lastEmittedEpoch, this.history.epochs + 1);
      if (point !== null) {
        this.history = appendHistory(this.history, point);
        this.lastEmittedEpoch = progress.completedEpochs;
      }
    }
  }

  /**
   * Change le nombre de filtres d'une couche : l'entraînement s'arrête et le
   * modèle est reconstruit (poids pré-entraînés si le tuple correspond au
   * manifeste, déterministes sinon). L'inférence est recalculée sur l'entrée
   * courante.
   */
  private setFilterCount(layer: number, count: number): void {
    const next: [number, number, number, number] = [
      this.filterCounts[0],
      this.filterCounts[1],
      this.filterCounts[2],
      this.filterCounts[3],
    ];
    next[layer] = clamp(Math.round(count), MIN_FILTER_COUNT, MAX_FILTER_COUNT);
    if (next[layer] === this.filterCounts[layer]) {
      return;
    }
    this.filterCounts = next;
    this.weights = this.weightsForCurrentFilters();
    this.clearSelection();
    this.runInference();
    this.resetMetrics();
    void this.controller.notifySliderChanged(this.manifestMatches, true);
  }

  /** Change la part du jeu d'entraînement : n'agit qu'à la prochaine run et
   * n'interrompt pas une run active. */
  private setTrainPercent(value: number): void {
    const percent = clamp(Math.round(value), MIN_PERCENT, MAX_PERCENT);
    if (percent === this.trainPercent) {
      return;
    }
    this.trainPercent = percent;
    if (!this.controller.isRunActive()) {
      this.updateExampleCounts();
    }
  }

  /** Change la part du jeu de validation : n'agit qu'à la prochaine run et
   * n'interrompt pas une run active. */
  private setValidationPercent(value: number): void {
    const percent = clamp(Math.round(value), MIN_PERCENT, MAX_PERCENT);
    if (percent === this.validationPercent) {
      return;
    }
    this.validationPercent = percent;
    if (!this.controller.isRunActive()) {
      this.updateExampleCounts();
    }
  }

  /** Change le taux d'apprentissage : n'agit qu'à la prochaine run et
   * n'interrompt pas une run active. */
  private setLearningRate(value: number): void {
    const rate = clamp(value, MIN_LEARNING_RATE, MAX_LEARNING_RATE);
    if (rate === this.learningRate) {
      return;
    }
    this.learningRate = rate;
  }

  /** Change la taille des lots : n'interrompt pas une run active et n'agit qu'à la prochaine run. */
  private setBatchSize(value: number): void {
    const size = clamp(Math.round(value), MIN_BATCH_SIZE, MAX_BATCH_SIZE);
    if (size === this.batchSize) {
      return;
    }
    this.batchSize = size;
  }

  /** Applique le chiffre choisi et remet la position au premier exemple. */
  private readonly handleGallerySelect = (event: Event): void => {
    const target = event.currentTarget;
    if (!(target instanceof HTMLSelectElement)) {
      return;
    }
    if (target.value === '') {
      void this.loadGallery();
      return;
    }
    const classValue = Number(target.value);
    if (!Number.isInteger(classValue) || classValue < 0 || classValue >= MNIST_CLASS_COUNT) {
      return;
    }
    this.galleryClass = classValue;
    this.galleryIndex = 0;
    this.applyGalleryExample();
  };

  /** Exemple précédent du chiffre courant, avec bouclage. */
  private readonly handleGalleryPrevious = (): void => {
    if (!this.galleryReady) {
      void this.loadGallery();
      return;
    }
    this.galleryIndex = stepGalleryIndex(this.galleryIndex, this.galleryPerClass, -1);
    this.applyGalleryExample();
  };

  /** Exemple suivant du chiffre courant, avec bouclage. */
  private readonly handleGalleryNext = (): void => {
    if (!this.galleryReady) {
      void this.loadGallery();
      return;
    }
    this.galleryIndex = stepGalleryIndex(this.galleryIndex, this.galleryPerClass, 1);
    this.applyGalleryExample();
  };

  /** Ouvrir le sélecteur déclenche le chargement de la galerie. */
  private readonly handleSampleSelectorOpen = (): void => {
    if (this.gallerySample === null && !this.galleryLoading) {
      void this.loadGallery();
    }
  };

  protected override updated(): void {
    if (this.canvasesDirty) {
      this.canvasesDirty = false;
      this.drawCanvases();
    }
  }

  private runInference(): void {
    this.result = forwardMnistCnn(this.inputPixels, this.weights, {
      filters: this.filterCounts,
    });
    this.canvasesDirty = true;
  }

  private readonly onDrawPointerDown = (event: PointerEvent): void => {
    const canvas = event.currentTarget;
    if (!(canvas instanceof HTMLCanvasElement)) {
      return;
    }
    event.preventDefault();
    this.drawing = true;
    canvas.setPointerCapture(event.pointerId);
    this.paint(canvas, event);
  };

  private readonly onDrawPointerMove = (event: PointerEvent): void => {
    if (!this.drawing) {
      return;
    }
    const canvas = event.currentTarget;
    if (canvas instanceof HTMLCanvasElement) {
      this.paint(canvas, event);
    }
  };

  private readonly onDrawPointerUp = (event: PointerEvent): void => {
    if (!this.drawing) {
      return;
    }
    this.drawing = false;
    const canvas = event.currentTarget;
    if (canvas instanceof HTMLCanvasElement && canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  };

  private paint(canvas: HTMLCanvasElement, event: PointerEvent): void {
    const cell = pointerToCell(canvas, event);
    if (!cell) {
      return;
    }
    paintDisc(this.inputPixels, cell[0], cell[1]);
    this.runInference();
  }

  private readonly handleClearDrawing = (): void => {
    this.inputPixels.fill(0);
    this.galleryLabel = null;
    this.inputSource = 'Dessin';
    this.runInference();
  };

  /**
   * Applique un événement de l'éditeur à la sélection courante : la sélection est
   * recalculée, la saisie est effacée — elle visait l'ancienne sélection — et les
   * toiles sont redessinées. Dès que la sélection devient un poids, la saisie est
   * pré-remplie avec la valeur courante formatée et le champ prend le focus une
   * fois le DOM à jour.
   */
  private applySelectionEvent(event: WeightEditorEvent): void {
    this.selection = applyWeightEditorEvent(this.selection, event);
    this.weightDraft = '';
    this.weightError = null;
    const view =
      this.selection !== null && this.selection.kind === 'weight'
        ? weightEditorView(this.selection, this.weights, this.filterCounts)
        : null;
    if (view !== null) {
      this.weightDraft = view.text;
      this.focusWeightInput();
    }
    this.canvasesDirty = true;
  }

  /**
   * Efface la sélection de l'éditeur et l'indicateur d'édition manuelle : appelée
   * dès que des poids sont remplacés par une source autre que l'édition manuelle,
   * une sélection plus ancienne ne désignant plus rien de valide.
   */
  private clearSelection(): void {
    this.cancelWeightEdit();
    this.weightsEdited = false;
  }

  /**
   * Efface la sélection, la saisie et l'erreur de l'éditeur, sans toucher à
   * l'indicateur d'édition manuelle : appelée par « Annuler » et Échap, le poids
   * déjà appliqué reste écrit dans les poids affichés.
   */
  private cancelWeightEdit(): void {
    this.selection = null;
    this.weightDraft = '';
    this.weightError = null;
    this.canvasesDirty = true;
  }

  /**
   * Clic sur une ligne de liaison : sélectionne la liaison entre les deux canaux
   * (ou la désélectionne). Le clic part du `<g>`, qui porte la zone de clic large :
   * seul le segment, visible ou non, est sensible.
   */
  private readonly handleLinkClick = (
    boundary: number,
    inputChannel: number,
    outputChannel: number,
  ): void => {
    this.applySelectionEvent({ kind: 'link', boundary, inputChannel, outputChannel });
  };

  /**
   * Clavier d'une ligne de liaison : Entrée et Espace sélectionnent le couple
   * comme le clic ; Espace est bloqué pour ne pas faire défiler la page.
   */
  private readonly handleLinkKeyDown = (
    event: KeyboardEvent,
    boundary: number,
    inputChannel: number,
    outputChannel: number,
  ): void => {
    if (event.key === 'Enter') {
      this.handleLinkClick(boundary, inputChannel, outputChannel);
    } else if (event.key === ' ') {
      event.preventDefault();
      this.handleLinkClick(boundary, inputChannel, outputChannel);
    }
  };

  /**
   * Clic sur une case d'une toile de filtres : ne produit une sélection de poids
   * que si la sélection courante est la liaison de la frontière immédiatement en
   * amont désignant cette toile. Sinon le clic est ignoré : le canal d'entrée
   * affiché par la toile est alors indéterminé.
   */
  private readonly handleFilterCellClick = (
    event: PointerEvent,
    canvas?: HTMLCanvasElement,
  ): void => {
    const target = (canvas ?? event.currentTarget) as HTMLCanvasElement | null;
    if (!(target instanceof HTMLCanvasElement)) {
      return;
    }
    const layer = Number(target.dataset.layer);
    const channel = Number(target.dataset.channel);
    if (!Number.isInteger(layer) || !Number.isInteger(channel)) {
      return;
    }
    const cell = pointerToGrid(target, event, CONV_KERNEL_SIZE);
    if (!cell) {
      return;
    }
    const kx = clamp(Math.floor(cell[0]), 0, CONV_KERNEL_SIZE - 1);
    const ky = clamp(Math.floor(cell[1]), 0, CONV_KERNEL_SIZE - 1);
    const inputChannel = filterCanvasInputChannel(this.selection, layer, channel);
    if (inputChannel === null) {
      return;
    }
    this.applySelectionEvent({
      kind: 'filterCell',
      layer,
      outputChannel: channel,
      inputChannel,
      ky,
      kx,
    });
  };

  /**
   * Saisie du brouillon : le texte du champ devient `weightDraft`, que
   * `handleApplyWeight` soumet à `applyWeightDraft`. Le message d'erreur est
   * effacé : l'utilisateur corrige la valeur rejetée.
   */
  private readonly handleWeightInput = (event: Event): void => {
    const target = event.currentTarget;
    if (target instanceof HTMLInputElement) {
      this.weightDraft = target.value;
      this.weightError = null;
    }
  };

  /**
   * Clavier du champ de saisie : Entrée applique le brouillon comme le bouton
   * « Appliquer », Échap l'annule comme le bouton « Annuler ».
   */
  private readonly handleWeightKeyDown = (event: KeyboardEvent): void => {
    if (event.key === 'Enter') {
      event.preventDefault();
      this.handleApplyWeight();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      this.handleCancelWeight();
    }
  };

  /**
   * Applique le brouillon au poids sélectionné : les poids écrits deviennent les
   * poids affichés, l'indicateur d'édition manuelle est levé, l'inférence repart
   * et le contrôleur est notifié. En cas d'erreur, le message s'affiche et le
   * formulaire reste ouvert ; sinon la sélection et le focus sont conservés pour
   * la modification suivante.
   */
  private readonly handleApplyWeight = (): void => {
    const result = applyWeightDraft(
      this.weightDraft,
      this.selection,
      this.weights,
      this.filterCounts,
    );
    if (result.kind === 'error') {
      this.weightError = result.message;
      return;
    }
    // Poids capturés avant la mise en file : la notification sérialisée envoie
    // la valeur de cette édition, pas celle d'une édition suivante.
    const edited = result.weights;
    this.weights = edited;
    this.weightsEdited = true;
    this.runInference();
    this.weightsNotifyChain = this.weightsNotifyChain
      .then(() => this.controller.notifyWeightsEdited(edited))
      .catch(() => undefined);
    this.weightError = null;
  };

  /**
   * Annule l'édition : sélection, saisie et erreur sont effacées — les toiles
   * retombent sur la vue en moyenne des canaux — mais les poids déjà écrits
   * restent édités, donc l'origine « Modifié » est conservée.
   */
  private readonly handleCancelWeight = (): void => {
    this.cancelWeightEdit();
  };

  /**
   * Donne le focus au champ de saisie après la mise à jour du DOM : le champ
   * n'existe qu'une fois la sélection de poids rendue par Lit.
   */
  private focusWeightInput(): void {
    void this.updateComplete.then(() => {
      this.renderRoot.querySelector<HTMLInputElement>('input.drl-weight-input')?.focus();
    });
  }

  private filterPalette(): {
    positive: readonly [number, number, number];
    negative: readonly [number, number, number];
  } {
    if (!this.paletteCache) {
      const style = getComputedStyle(this);
      const read = (name: string, fallback: string): string => {
        const value = style.getPropertyValue(name).trim();
        return value.length > 0 ? value : fallback;
      };
      this.paletteCache = {
        positive: hexToRgb(read('--drl-accent', '#2563eb')),
        negative: hexToRgb(read('--drl-accent-2', '#7c3aed')),
      };
    }
    return this.paletteCache;
  }

  private drawCanvases(): void {
    this.drawInputCanvas(
      this.renderRoot.querySelector<HTMLCanvasElement>('canvas.drl-canvas--input'),
    );
    const result = this.result;
    if (!result) {
      return;
    }
    for (const canvas of Array.from(
      this.renderRoot.querySelectorAll<HTMLCanvasElement>('canvas.drl-canvas--filter'),
    )) {
      const layer = Number(canvas.dataset.layer);
      const channel = Number(canvas.dataset.channel);
      if (Number.isInteger(layer) && Number.isInteger(channel)) {
        this.drawFilterCanvas(canvas, layer, channel);
      }
    }
    for (const canvas of Array.from(
      this.renderRoot.querySelectorAll<HTMLCanvasElement>('canvas.drl-canvas--map'),
    )) {
      const layer = Number(canvas.dataset.layer);
      const channel = Number(canvas.dataset.channel);
      const featureMap = result.featureMaps[layer];
      if (featureMap && channel < featureMap.channels) {
        this.drawFeatureMapCanvas(canvas, featureMap, channel);
      }
    }
    this.positionFilterLinks();
  }

  private drawInputCanvas(canvas: HTMLCanvasElement | null): void {
    if (!canvas) {
      return;
    }
    const context = canvas.getContext('2d');
    const cells = this.inputCells.getContext('2d');
    if (!context || !cells) {
      return;
    }
    if (this.inputCells.width !== MNIST_INPUT_SIZE || this.inputCells.height !== MNIST_INPUT_SIZE) {
      this.inputCells.width = MNIST_INPUT_SIZE;
      this.inputCells.height = MNIST_INPUT_SIZE;
    }
    const image = cells.createImageData(MNIST_INPUT_SIZE, MNIST_INPUT_SIZE);
    for (let index = 0; index < MNIST_INPUT_LENGTH; index += 1) {
      const shade = Math.round(255 * (1 - clamp(this.inputPixels[index], 0, 1)));
      image.data[index * 4] = shade;
      image.data[index * 4 + 1] = shade;
      image.data[index * 4 + 2] = shade;
      image.data[index * 4 + 3] = 255;
    }
    cells.putImageData(image, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(this.inputCells, 0, 0, INPUT_CANVAS_SIZE, INPUT_CANVAS_SIZE);
    context.strokeStyle = 'rgba(15, 23, 42, 0.08)';
    context.lineWidth = 1;
    context.beginPath();
    for (let line = 0; line <= MNIST_INPUT_SIZE; line += 1) {
      const position = line * DRAW_SCALE + 0.5;
      context.moveTo(position, 0);
      context.lineTo(position, INPUT_CANVAS_SIZE);
      context.moveTo(0, position);
      context.lineTo(INPUT_CANVAS_SIZE, position);
    }
    context.stroke();
  }

  private drawFilterCanvas(canvas: HTMLCanvasElement, layer: number, channel: number): void {
    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }
    const inputChannels = layer === 0 ? 1 : this.filterCounts[layer - 1];
    const convWeights: readonly MnistCnnLayerWeights[] = [
      this.weights.conv1,
      this.weights.conv2,
      this.weights.conv3,
      this.weights.conv4,
    ];
    const view = filterViewMode(this.selection, layer, channel);
    const values = filterKernelValues(view, convWeights[layer].kernel, inputChannels, channel);
    let maxAbs = 0;
    for (const value of values) {
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
    if (this.scratch.width !== CONV_KERNEL_SIZE || this.scratch.height !== CONV_KERNEL_SIZE) {
      this.scratch.width = CONV_KERNEL_SIZE;
      this.scratch.height = CONV_KERNEL_SIZE;
    }
    const scratch = this.scratch.getContext('2d');
    if (!scratch) {
      return;
    }
    const palette = this.filterPalette();
    const image = scratch.createImageData(CONV_KERNEL_SIZE, CONV_KERNEL_SIZE);
    for (let ky = 0; ky < CONV_KERNEL_SIZE; ky += 1) {
      for (let kx = 0; kx < CONV_KERNEL_SIZE; kx += 1) {
        const index = (ky * CONV_KERNEL_SIZE + kx) * 4;
        const value = values[ky * CONV_KERNEL_SIZE + kx];
        const t = maxAbs > 1e-9 ? clamp(value / maxAbs, -1, 1) : 0;
        const rgb =
          t >= 0 ? mixRgb(WHITE_RGB, palette.positive, t) : mixRgb(WHITE_RGB, palette.negative, -t);
        image.data[index] = rgb[0];
        image.data[index + 1] = rgb[1];
        image.data[index + 2] = rgb[2];
        image.data[index + 3] = 255;
      }
    }
    scratch.putImageData(image, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(this.scratch, 0, 0, FILTER_CANVAS_SIZE, FILTER_CANVAS_SIZE);
  }

  private drawFeatureMapCanvas(
    canvas: HTMLCanvasElement,
    featureMap: MnistCnnFeatureMap,
    channel: number,
  ): void {
    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }
    if (this.scratch.width !== featureMap.width || this.scratch.height !== featureMap.height) {
      this.scratch.width = featureMap.width;
      this.scratch.height = featureMap.height;
    }
    const scratch = this.scratch.getContext('2d');
    if (!scratch) {
      return;
    }
    const image = scratch.createImageData(featureMap.width, featureMap.height);
    for (let index = 0; index < featureMap.width * featureMap.height; index += 1) {
      const value = clamp(featureMap.values[index * featureMap.channels + channel], 0, 1);
      const shade = Math.round(255 * value);
      image.data[index * 4] = shade;
      image.data[index * 4 + 1] = shade;
      image.data[index * 4 + 2] = shade;
      image.data[index * 4 + 3] = 255;
    }
    scratch.putImageData(image, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(this.scratch, 0, 0, canvas.width, canvas.height);
  }

  protected renderHeader(): TemplateResult {
    return html`
      <div class="drl-status">
        <span class="drl-badge">${badgeText(this.demoStatus)}</span>
        <span class="drl-status-text" role="status">${statusLine(this.demoStatus)}</span>
      </div>
    `;
  }

  protected override render(): TemplateResult {
    return html`
      <div class="drl-header">${this.renderHeader()}</div>
      <div class="drl-content">${this.renderContent()}</div>
    `;
  }

  /**
   * Une courbe d'entraînement : titre, grille de graduations, axe horizontal,
   * puis le tracé ou le message d'attente. Les fragments SVG imbriqués
   * (`<line>`, `<text>`, `<path>`) utilisent le tag `svg` : un template
   * dont la racine est un élément SVG doit être parsé dans le namespace SVG,
   * sinon les nœuds existent dans le DOM mais ne sont jamais dessinés.
   */
  private renderTrainingChart(layout: ChartLayout, lineClass: string): TemplateResult {
    const box = layout.box;
    const right = box.width - box.padRight;
    return html`
      <figure class="drl-chart">
        <figcaption class="drl-chart-title">${layout.name}</figcaption>
        <svg
          class="drl-chart-svg"
          viewBox="0 0 ${box.width} ${box.height}"
          role="img"
          aria-label=${layout.ariaLabel}
        >
          <g class="drl-chart-grid">
            ${layout.ticks.map(
              (tick) => svg`
                <line
                  class="drl-chart-gridline"
                  x1=${box.padLeft}
                  x2=${right}
                  y1=${tick.y}
                  y2=${tick.y}
                ></line>
                <text
                  class="drl-chart-tick"
                  x=${box.padLeft - 4}
                  y=${tick.y}
                  text-anchor="end"
                  dominant-baseline="middle"
                  >${formatValue(tick.value, 2)}</text
                >
              `,
            )}
          </g>
          <line
            class="drl-chart-axis"
            x1=${box.padLeft}
            x2=${right}
            y1=${layout.axisY}
            y2=${layout.axisY}
          ></line>
          ${
            layout.empty
              ? svg`<text
                class="drl-chart-empty"
                x=${box.width / 2}
                y=${box.height / 2}
                text-anchor="middle"
                >En attente de deux époques</text
              >`
              : svg`<path class="drl-chart-line ${lineClass}" d=${layout.path} fill="none"></path>`
          }
        </svg>
      </figure>
    `;
  }

  /**
   * Graphique multi-séries : titre, grille de graduations et axe partagés, un
   * tracé par courbe, puis une légende textuelle. Les fragments SVG imbriqués
   * (`<line>`, `<text>`, `<path>` et le `<svg>` de la pastille de légende)
   * utilisent le tag ``svg`` : un template dont la racine est un élément SVG
   * doit être parsé dans le namespace SVG. Le texte de la légende porte le sens,
   * donc la pastille est décorative (`aria-hidden="true"`). Une courbe sans
   * point n'a pas de tracé, mais reste dans la légende.
   */
  private renderMultiTrainingChart(layout: MultiChartLayout): TemplateResult {
    const box = layout.box;
    const right = box.width - box.padRight;
    return html`
      <figure class="drl-chart">
        <figcaption class="drl-chart-title">${layout.name}</figcaption>
        <svg
          class="drl-chart-svg"
          viewBox="0 0 ${box.width} ${box.height}"
          role="img"
          aria-label=${layout.ariaLabel}
        >
          <g class="drl-chart-grid">
            ${layout.ticks.map(
              (tick) => svg`
                <line
                  class="drl-chart-gridline"
                  x1=${box.padLeft}
                  x2=${right}
                  y1=${tick.y}
                  y2=${tick.y}
                ></line>
                <text
                  class="drl-chart-tick"
                  x=${box.padLeft - 4}
                  y=${tick.y}
                  text-anchor="end"
                  dominant-baseline="middle"
                  >${formatValue(tick.value, 2)}</text
                >
              `,
            )}
          </g>
          <line
            class="drl-chart-axis"
            x1=${box.padLeft}
            x2=${right}
            y1=${layout.axisY}
            y2=${layout.axisY}
          ></line>
          ${
            layout.empty
              ? svg`<text
                class="drl-chart-empty"
                x=${box.width / 2}
                y=${box.height / 2}
                text-anchor="middle"
                >En attente de deux époques</text
              >`
              : layout.series
                  .filter((series) => series.points.length > 0)
                  .map(
                    (series) => svg`<path
                      class="drl-chart-line ${series.lineClass}"
                      d=${series.path}
                      fill="none"
                    ></path>`,
                  )
          }
        </svg>
        <ul class="drl-chart-legend">
          ${layout.series.map(
            (series) => html`
              <li class="drl-chart-legend-item">
                ${svg`
                  <svg
                    class="drl-chart-legend-swatch"
                    width="24"
                    height="8"
                    aria-hidden="true"
                  >
                    <line
                      class="drl-chart-legend-line ${series.lineClass}"
                      x1="0"
                      y1="4"
                      x2="24"
                      y2="4"
                    ></line>
                  </svg>
                `}
                ${series.name}
              </li>
            `,
          )}
        </ul>
      </figure>
    `;
  }

  protected renderContent(): TemplateResult {
    const result = this.result;
    return html`
      <div class="drl-mnist-layout">
        <div class="drl-mnist-left">
          <section class="drl-panel" aria-label="Zone de dessin et choix de l'entrée">
            <h4 class="drl-panel-title">Dessin 28×28</h4>
            <canvas
              class="drl-canvas drl-canvas--input"
              width=${INPUT_CANVAS_SIZE}
              height=${INPUT_CANVAS_SIZE}
              role="img"
              aria-label="Surface de dessin 28 par 28 pixels pour un chiffre MNIST"
              @pointerdown=${this.onDrawPointerDown}
              @pointermove=${this.onDrawPointerMove}
              @pointerup=${this.onDrawPointerUp}
              @pointercancel=${this.onDrawPointerUp}
            ></canvas>
            <p class="drl-hint">${this.inputHint}</p>
            <div class="drl-field">
              <label class="drl-field-label" for="drl-mnist-cnn-sample">Exemple MNIST</label>
              <select
                id="drl-mnist-cnn-sample"
                aria-label="Choisir un exemple MNIST de la galerie"
                @click=${this.handleSampleSelectorOpen}
                @change=${this.handleGallerySelect}
              >
                ${
                  this.galleryLoading
                    ? html`<option>Chargement de la galerie…</option>`
                    : this.gallerySample === null
                      ? html`<option value="">Charger la galerie MNIST…</option>`
                      : repeat(
                          Array.from({ length: MNIST_CLASS_COUNT }, (_unused, digit) => digit),
                          (digit) => digit,
                          (digit) =>
                            html`<option value=${digit} ?selected=${digit === this.galleryClass}>
                              Chiffre ${digit}
                            </option>`,
                        )
                }
              </select>
              ${
                this.galleryError !== null
                  ? html`<p class="drl-hint">Galerie indisponible : ${this.galleryError}</p>`
                  : nothing
              }
            </div>
            <div
              class="drl-toolbar"
              role="group"
              aria-label="Parcourir les exemples du chiffre choisi"
            >
              <button
                type="button"
                class="drl-button"
                ?disabled=${this.galleryLoading}
                @click=${this.handleGalleryPrevious}
              >
                ← Précédent
              </button>
              <span class="drl-gallery-counter">${this.galleryCounterText}</span>
              <button
                type="button"
                class="drl-button"
                ?disabled=${this.galleryLoading}
                @click=${this.handleGalleryNext}
              >
                Suivant →
              </button>
            </div>
            <div class="drl-toolbar">
              <button type="button" class="drl-button" @click=${this.handleClearDrawing}>
                Effacer
              </button>
            </div>
          </section>

          <section class="drl-panel" aria-label="Entraînement du modèle">
            <h4 class="drl-panel-title">Entraînement</h4>
            <p class="drl-panel-subtitle">
              TensorFlow.js est chargé au premier clic sur un bouton d'entraînement, ainsi que les
              exemples MNIST locaux. L'entraînement repart toujours des poids affichés :
              pré-entraînés, aléatoires ou modifiés à la main. « Entraîner 1 époque » fait une
              passe ; « Entraîner en boucle » enchaîne les époques jusqu'à Pause. La courbe en
              pointillés mesure la précision d'entraînement sur l'échantillon borné ; la courbe en
              trait plein mesure la précision de validation sur le sous-ensemble tenu à l'écart.
            </p>
            <div class="drl-toolbar" role="group" aria-label="Lancer et contrôler l'entraînement">
              <button
                type="button"
                class="drl-button drl-primary"
                ?disabled=${isTrainingActive(this.trainingState)}
                @click=${this.handleStartTraining}
              >
                Entraîner 1 époque
              </button>
              <button
                type="button"
                class="drl-button"
                ?disabled=${isTrainingActive(this.trainingState)}
                @click=${this.handleStartLoopTraining}
              >
                Entraîner en boucle
              </button>
              <button
                type="button"
                class="drl-button"
                ?disabled=${!isTrainingActive(this.trainingState)}
                @click=${this.handlePauseTraining}
              >
                Pause
              </button>
              <button type="button" class="drl-button" @click=${this.handleResetWeights}>
                Réinitialiser les poids
              </button>
              <button type="button" class="drl-button" @click=${this.handleRandomResetWeights}>
                Poids aléatoires
              </button>
            </div>
            <div
              class="drl-sliders"
              role="group"
              aria-label="Fraction des jeux d'entraînement et de validation, taux d'apprentissage et taille des lots"
            >
              <drl-slider
                label="Données d'entraînement"
                min=${MIN_PERCENT}
                max=${MAX_PERCENT}
                step="1"
                .value=${this.trainPercent}
                @drl-input=${(event: CustomEvent<number>) => this.setTrainPercent(event.detail)}
              ></drl-slider>
              <drl-slider
                label="Validation"
                min=${MIN_PERCENT}
                max=${MAX_PERCENT}
                step="1"
                .value=${this.validationPercent}
                @drl-input=${(event: CustomEvent<number>) => this.setValidationPercent(event.detail)}
              ></drl-slider>
              <drl-slider
                label="Taux d'apprentissage"
                min=${MIN_LEARNING_RATE}
                max=${MAX_LEARNING_RATE}
                step="0.0005"
                .value=${this.learningRate}
                @drl-input=${(event: CustomEvent<number>) => this.setLearningRate(event.detail)}
              ></drl-slider>
              <drl-slider
                label="Taille des lots"
                min=${MIN_BATCH_SIZE}
                max=${MAX_BATCH_SIZE}
                step="8"
                .value=${this.batchSize}
                @drl-input=${(event: CustomEvent<number>) => this.setBatchSize(event.detail)}
              ></drl-slider>
            </div>
            <p class="drl-hint">
              ${
                this.trainingError !== null
                  ? html`Erreur d'entraînement : ${this.trainingError}`
                  : "Les curseurs de pourcentage, de taux et de taille des lots ne prennent effet qu'au prochain entraînement, sans interrompre une run."
              }
            </p>
          </section>
        </div>

        <section class="drl-panel" aria-label="Statistiques, probabilités et courbes d'entraînement">
          <h4 class="drl-panel-title">Résultats</h4>
          <div class="drl-stats">
            <div class="drl-stat">
              <p class="drl-stat-label">Paramètres</p>
              <p class="drl-stat-value">${mnistCnnParameterCount(this.filterCounts)}</p>
            </div>
            <div class="drl-stat">
              <p class="drl-stat-label">Classe prédite</p>
              <p class="drl-stat-value">${this.predictedClass ?? '–'}</p>
            </div>
            <div class="drl-stat">
              <p class="drl-stat-label">Entrée</p>
              <p class="drl-stat-value">${this.inputSource}</p>
            </div>
            <div class="drl-stat">
              <p class="drl-stat-label">Poids</p>
              <p class="drl-stat-value">${weightsOriginText(this.demoStatus)}</p>
            </div>
          </div>

          <h4 class="drl-panel-title">Probabilités par classe</h4>
          <div class="drl-probs" role="list" aria-label="Probabilités des dix classes de chiffres">
            ${
              result
                ? repeat(
                    result.probabilities,
                    (_probability: number, digit: number) => digit,
                    (probability, digit) => html`
                    <div
                      class="${digit === this.predictedClass ? 'drl-prob-row drl-prob-row--max' : 'drl-prob-row'}"
                      role="listitem"
                    >
                      <span class="drl-prob-digit">${digit}</span>
                      <span class="drl-prob-bar-wrap"
                        ><span class="drl-prob-bar" style="width:${(probability * 100).toFixed(1)}%"
                        ></span
                      ></span>
                      <span class="drl-prob-value">${formatPercent(probability * 100)}</span>
                    </div>
                  `,
                  )
                : html`<p class="drl-hint">En attente du premier passage avant.</p>`
            }
          </div>

          <div class="drl-charts">
            ${this.renderTrainingChart(this.lossLayout, 'drl-chart-line--loss')}
            ${this.renderMultiTrainingChart(this.accuracyLayout)}
          </div>
        </section>
      </div>

      <details class="drl-filter-counts" open>
        <summary><h4 class="drl-panel-title">Filtres par couche</h4></summary>
        <div class="drl-filter-counts-body">
          <p class="drl-panel-subtitle">
            Chaque curseur arrête l'entraînement et reconstruit le modèle : poids pré-entraînés
            (manifeste) si le tuple correspond, poids déterministes (graine ${SEED}) sinon.
          </p>
          <div
            class="drl-sliders"
            role="group"
            aria-label="Nombre de filtres de chacune des quatre couches de convolution"
          >
            ${repeat(
              CONV_LAYER_NAMES,
              (name) => name,
              (name, layer) => html`
                <drl-slider
                  label="Filtres ${name}"
                  min=${MIN_FILTER_COUNT}
                  max=${MAX_FILTER_COUNT}
                  step="1"
                  .value=${this.filterCounts[layer]}
                  @drl-input=${(event: CustomEvent<number>) =>
                    this.setFilterCount(layer, event.detail)}
                ></drl-slider>
              `,
            )}
          </div>
        </div>
      </details>

      <div class="drl-panels-row">
        <section class="drl-panel drl-panel--filters" aria-label="Filtres de convolution">
          <h4 class="drl-panel-title">Filtres de convolution 3×3</h4>
          <p class="drl-panel-subtitle">${filterPanelSubtitle(this.selection)}</p>
          ${this.renderWeightEditor()}
          ${
            result
              ? repeat(
                  CONV_LAYER_NAMES,
                  (name) => name,
                  (name, layer) => html`
                  <div class="drl-layer-block">
                    <h5 class="drl-layer-title">${name} — ${this.filterCounts[layer]} canal(aux)</h5>
                    <div class="drl-cells-grid">
                      ${repeat(
                        Array.from({ length: this.filterCounts[layer] }, (_, channel) => channel),
                        (channel) => channel,
                        (channel) => html`
                          <figure class="drl-cell">
                            <canvas
                              class="drl-canvas drl-canvas--filter"
                              data-layer=${layer}
                              data-channel=${channel}
                              width=${FILTER_CANVAS_SIZE}
                              height=${FILTER_CANVAS_SIZE}
                              role="img"
                              aria-label="Filtre ${name}, canal de sortie ${channel + 1}"
                              @pointerdown=${this.handleFilterCellClick}
                            ></canvas>
                            <figcaption>sortie ${channel + 1}</figcaption>
                          </figure>
                        `,
                      )}
                    </div>
                  </div>
                `,
                )
              : html`<p class="drl-hint">En attente du premier passage avant.</p>`
          }
          ${this.renderFilterLinks()}
        </section>

        <section class="drl-panel" aria-label="Cartes de caractéristiques">
          <h4 class="drl-panel-title">Cartes de caractéristiques</h4>
          <p class="drl-panel-subtitle">
            Activations après sigmoïde ; blanc = activation forte. Tailles : 26, 13, 11 puis 5.
          </p>
          ${
            result
              ? repeat(
                  result.featureMaps,
                  (featureMap) => featureMap.name,
                  (featureMap, layer) => html`
                  <div class="drl-layer-block">
                    <h5 class="drl-layer-title">
                      ${featureMap.name} — ${featureMap.width}×${featureMap.height}×${featureMap.channels}
                    </h5>
                    <div class="drl-cells-grid">
                      ${repeat(
                        Array.from({ length: featureMap.channels }, (_, channel) => channel),
                        (channel) => channel,
                        (channel) => html`
                          <figure class="drl-cell">
                            <canvas
                              class="drl-canvas drl-canvas--map"
                              data-layer=${layer}
                              data-channel=${channel}
                              width=${MAP_CANVAS_SIZES[layer]}
                              height=${MAP_CANVAS_SIZES[layer]}
                              role="img"
                              aria-label="Carte de caractéristiques ${featureMap.name}, canal ${channel + 1}"
                            ></canvas>
                            <figcaption>canal ${channel + 1}</figcaption>
                          </figure>
                        `,
                      )}
                    </div>
                  </div>
                `,
                )
              : html`<p class="drl-hint">En attente du premier passage avant.</p>`
          }
        </section>
      </div>
    `;
  }

  /**
   * Formulaire d'édition du poids sélectionné, en haut du panneau des filtres :
   * libellé de la sélection, saisie au clavier pré-remplie, boutons « Appliquer »
   * et « Annuler », ligne d'erreur. Une liaison sélectionnée sans case affiche
   * le conseil de geste de `weight-editor.ts` à la place du formulaire.
   */
  private renderWeightEditor(): TemplateResult | typeof nothing {
    const selection = this.selection;
    if (selection !== null && selection.kind === 'link') {
      return html`<p class="drl-hint">${weightEditorHint()}</p>`;
    }
    if (selection === null) {
      return nothing;
    }
    const view = weightEditorView(selection, this.weights, this.filterCounts);
    if (view === null) {
      return nothing;
    }
    return html`
      <div class="drl-weight-editor">
        <div class="drl-weight-row">
          <label class="drl-weight-label" for="drl-mnist-cnn-weight">${view.label}</label>
          <input
            id="drl-mnist-cnn-weight"
            class="drl-weight-input"
            type="text"
            inputmode="decimal"
            autocomplete="off"
            .value=${this.weightDraft}
            @input=${this.handleWeightInput}
            @keydown=${this.handleWeightKeyDown}
          />
          <button type="button" class="drl-button drl-primary" @click=${this.handleApplyWeight}>
            Appliquer
          </button>
          <button type="button" class="drl-button" @click=${this.handleCancelWeight}>
            Annuler
          </button>
        </div>
        ${
          this.weightError !== null
            ? html`<p class="drl-weight-error" role="alert">${this.weightError}</p>`
            : nothing
        }
      </div>
    `;
  }

  /**
   * Couche SVG des lignes de liaison : un `<g>` par liaison entre couches de
   * convolution, positionné ensuite par `positionFilterLinks`. Elle reste dans le
   * panneau même sans passage avant, pour que le `ResizeObserver` la trouve.
   */
  private renderFilterLinks(): TemplateResult {
    return html`
      <svg class="drl-filter-links">
        ${repeat(
          enumerateLinks(this.filterCounts),
          (link) => `${link.boundary}:${link.inputChannel}:${link.outputChannel}`,
          (link) => this.renderFilterLink(link),
        )}
      </svg>
    `;
  }

  /**
   * Une ligne de liaison : une ligne visible à l'état courant, une zone de clic
   * large et transparente, et le libellé porté par le titre et par `aria-label`.
   */
  private renderFilterLink(link: LinkEndpoint): TemplateResult {
    const state = linkLineState(
      this.selection,
      link.boundary,
      link.inputChannel,
      link.outputChannel,
    );
    const label = describeLinkLabel(
      this.selection,
      link.boundary,
      link.inputChannel,
      link.outputChannel,
      this.filterCounts,
    );
    return html`
      <g
        class="drl-filter-link-group"
        role="button"
        tabindex="0"
        data-boundary=${link.boundary}
        data-input-channel=${link.inputChannel}
        data-output-channel=${link.outputChannel}
        aria-label=${label}
        @click=${() => this.handleLinkClick(link.boundary, link.inputChannel, link.outputChannel)}
        @keydown=${(event: KeyboardEvent) =>
          this.handleLinkKeyDown(event, link.boundary, link.inputChannel, link.outputChannel)}
      >
        <title>${label}</title>
        <line class="drl-filter-link drl-filter-link--${state}" x1="0" y1="0" x2="0" y2="0"></line>
        <line class="drl-filter-link-hit" x1="0" y1="0" x2="0" y2="0"></line>
      </g>
    `;
  }
}
