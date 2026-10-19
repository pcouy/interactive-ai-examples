import type { TemplateResult } from 'lit';
import { css, html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { DemoElement } from '../../base/demo-element.js';
import '../../ui/drl-slider.js';
import {
  CONV_KERNEL_SIZE,
  CONV_LAYER_NAMES,
  CONV_PADDINGS,
  CONV_STRIDES,
  convOutputLength,
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
import { drlTokens } from '../../styles/tokens.js';

/**
 * Démonstration interactive du CNN MNIST de référence (`tf-playground/mnist/train.py`) :
 * quatre convolutions 3×3 + sigmoïde, aplatissement, dense 8 + sigmoïde, dix logits.
 * L'inférence est le passage avant pur TypeScript de `core/mnist-cnn.ts` ; aucun
 * TensorFlow.js, aucun asset externe, aucune dépendance 3D. Le rendu est un
 * assemblage de canvas 2D stylés avec les jetons `drlTokens`.
 *
 * Poids : initialisation déterministe (graine fixe) via `initializeMnistCnnWeights`,
 * en attendant le modèle pré-entraîné ou l'adaptateur d'entraînement TFJS.
 */

/** Graine de l'initialisation déterministe des poids. */
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

const WHITE_RGB: readonly [number, number, number] = [255, 255, 255];

const FEATURE_MAP_SIZES: readonly number[] = (() => {
  const sizes: number[] = [];
  let size = MNIST_INPUT_SIZE;
  for (let layer = 0; layer < CONV_STRIDES.length; layer += 1) {
    size = convOutputLength(size, CONV_STRIDES[layer], CONV_PADDINGS[layer]);
    sizes.push(size);
  }
  return sizes;
})();

/** Taille de cellule entière d'une carte pour que la toile reste ≤ MAP_CANVAS_MAX. */
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

/** Convertit un événement pointeur en coordonnées 28×28, ou null si la toile est vide. */
export function pointerToCell(
  canvas: HTMLCanvasElement,
  event: PointerEvent,
): [number, number] | null {
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) {
    return null;
  }
  const gx = ((event.clientX - rect.left) / rect.width) * MNIST_INPUT_SIZE;
  const gy = ((event.clientY - rect.top) / rect.height) * MNIST_INPUT_SIZE;
  return [gx, gy];
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

/** Pourcentage à une décimale, format français (virgule). */
export function formatPercent(value: number): string {
  return `${value.toFixed(1).replace('.', ',')} %`;
}

@customElement('drl-mnist-cnn')
export class MnistCnn extends DemoElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: block;
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

      .drl-panels-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, minmax(0, 1fr)));
        gap: var(--drl-space-3);
        margin-top: var(--drl-space-3);
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

      .drl-canvas--filter,
      .drl-canvas--map {
        width: auto;
        height: auto;
      }

      .drl-hint {
        margin: 0;
        color: var(--drl-muted);
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

      .drl-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, minmax(0, 1fr)));
        gap: var(--drl-space-2);
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

  @state()
  private drawing = false;

  /** Dessin des toiles reporté après la prochaine mise à jour du DOM Lit. */
  private canvasesDirty = true;

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

  private get predictedClass(): number | null {
    return this.result ? argmaxIndex(this.result.probabilities) : null;
  }

  private get statusLine(): string {
    const predicted = this.predictedClass;
    const parameters = mnistCnnParameterCount(this.filterCounts);
    const label = predicted === null ? '–' : String(predicted);
    return `Entrée : ${this.inputSource} · ${parameters} paramètres · classe prédite : ${label}`;
  }

  protected override firstUpdated(): void {
    this.runInference();
  }

  override step(): void {
    this.runInference();
  }

  protected override onReset(): void {
    // Remise à zéro de l'état d'entraînement : les poids déterministes sont
    // régénérés avec la graine fixe et l'inférence est recalculée. Le dessin
    // de l'utilisateur est conservé.
    this.weights = initializeMnistCnnWeights({ filters: this.filterCounts }, SEED);
    this.runInference();
  }

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

  /**
   * Change le nombre de filtres d'une couche : le modèle est reconstruit avec
   * des poids déterministes et les métriques sont remises à zéro (inférence
   * recalculée sur l'entrée courante).
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
    this.weights = initializeMnistCnnWeights({ filters: this.filterCounts }, SEED);
    this.runInference();
  }

  private readonly onDrawPointerDown = (event: PointerEvent): void => {
    const canvas = event.currentTarget;
    if (!(canvas instanceof HTMLCanvasElement)) {
      return;
    }
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
    this.inputSource = 'Dessin';
    this.runInference();
  };

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
    const means = filterKernelMean(convWeights[layer].kernel, inputChannels, channel);
    let maxAbs = 0;
    for (const value of means) {
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
        const value = means[ky * CONV_KERNEL_SIZE + kx];
        const t = maxAbs > 1e-9 ? clamp(value / maxAbs, -1, 1) : 0;
        const rgb =
          t >= 0 ? mixRgb(WHITE_RGB, palette.positive, t) : mixRgb(WHITE_RGB, palette.negative, -t);
        const index = (ky * CONV_KERNEL_SIZE + kx) * 4;
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

  protected override renderHeader(): TemplateResult {
    return html`
      <div class="drl-status">
        <span class="drl-badge">Inférence MNIST</span>
        <span class="drl-status-text">${this.statusLine}</span>
      </div>
    `;
  }

  protected override renderContent(): TemplateResult {
    const result = this.result;
    return html`
      <div class="drl-mnist-layout">
        <section class="drl-panel" aria-label="Zone de dessin et choix de l'entrée">
          <h4 class="drl-panel-title">Dessin 28×28</h4>
          <canvas
            class="drl-canvas drl-canvas--input"
            width=${INPUT_CANVAS_SIZE}
            height=${INPUT_CANVAS_SIZE}
            aria-label="Surface de dessin 28 par 28 pixels pour un chiffre MNIST"
            @pointerdown=${this.onDrawPointerDown}
            @pointermove=${this.onDrawPointerMove}
            @pointerup=${this.onDrawPointerUp}
            @pointercancel=${this.onDrawPointerUp}
          ></canvas>
          <p class="drl-hint">
            Dessinez un chiffre à la souris ou au doigt ; l'inférence se met à jour pendant
            le tracé.
          </p>
          <div class="drl-field">
            <label class="drl-field-label" for="drl-mnist-cnn-sample">Exemple MNIST</label>
            <select
              id="drl-mnist-cnn-sample"
              disabled
              aria-label="Sélecteur d'exemples MNIST, bientôt disponible"
            >
              <option>Exemples locaux IDX — bientôt disponible</option>
            </select>
          </div>
          <div class="drl-toolbar">
            <button type="button" class="drl-button" @click=${this.handleClearDrawing}>
              Effacer
            </button>
          </div>
        </section>

        <section class="drl-panel" aria-label="Probabilités et réglages du modèle">
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
              <p class="drl-stat-value">Déterministes</p>
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

          <h4 class="drl-panel-title">Filtres par couche</h4>
          <p class="drl-panel-subtitle">
            Chaque curseur reconstruit le modèle (poids déterministes, graine ${SEED}) et remet
            les métriques à zéro.
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
        </section>
      </div>

      <div class="drl-panels-row">
        <section class="drl-panel" aria-label="Filtres de convolution">
          <h4 class="drl-panel-title">Filtres de convolution 3×3</h4>
          <p class="drl-panel-subtitle">
            Moyenne des canaux d'entrée pour chaque canal de sortie ; bleu = positif, violet =
            négatif.
          </p>
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
                              aria-label="Filtre ${name}, canal de sortie ${channel + 1}"
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
}
