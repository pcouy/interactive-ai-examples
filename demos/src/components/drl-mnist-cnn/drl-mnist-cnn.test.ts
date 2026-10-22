import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { getDataset } from '../../core/dataset-registry.js';
import type {
  MnistCnnFilters,
  MnistCnnLayerWeights,
  MnistCnnWeights,
} from '../../core/mnist-cnn.js';
import {
  buildSegmentPlan,
  type RoundEntry,
  SEGMENT_IMAGE_THRESHOLD,
  shouldUseSegments,
} from '../../core/mnist-cnn-segmented.js';
import type { MnistIdxDataset } from '../../core/mnist-data.js';
import { displayClass } from './class-selection.js';
import type { TrainingRunInput, WeightEditorEvent } from './drl-mnist-cnn.js';
import {
  activationRgb,
  activationScaleAriaLabel,
  activationScaleLabels,
  applyWeightEditorEvent,
  argmaxIndex,
  brushStrength,
  clampDenseSize,
  clampPercent,
  convLayerGeometry,
  denseCellLabel,
  denseLayerTitle,
  densePackColumns,
  denseSizeLabel,
  describeLinkLabel,
  describeWeightSelection,
  filterBiasCaption,
  filterCanvasInputChannel,
  filterKernelMean,
  filterKernelSlice,
  filterKernelValues,
  filterLinkGeometry,
  filterPanelSubtitle,
  filterViewMode,
  galleryClassLabel,
  galleryInputHint,
  galleryInputSource,
  groupEntriesByNetworkClass,
  hexToRgb,
  isReferenceDenseSizes,
  linkCanvasSelectors,
  linkGroupIndices,
  linkLineState,
  loadSplitsForDataset,
  MnistCnn,
  mapCanvasSize,
  mapCellSize,
  mixRgb,
  nextGalleryClass,
  outputCellLabel,
  paintDisc,
  parseWeightInput,
  pointerToCell,
  pointerToGrid,
  segmentRunNeeded,
  segmentSelectionCount,
  selectedWeightValue,
  stepGalleryIndex,
  stratifiedExampleCountEstimate,
  trainingRunRequest,
} from './drl-mnist-cnn.js';
import type { RectLike } from './filter-links.js';
import { linkGeometry } from './filter-links.js';
import { loadDatasetSplits, type MnistSplits } from './mnist-assets.js';
import {
  loadShardedSplits,
  SegmentStore,
  type ShardEntry,
  type ShardedIndex,
  type ShardedSplitIndex,
  selectShardEntries,
} from './sharded-assets.js';

vi.mock('./sharded-assets.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./sharded-assets.js')>();
  return { ...actual, loadShardedSplits: vi.fn() };
});

vi.mock('./mnist-assets.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./mnist-assets.js')>();
  return { ...actual, loadDatasetSplits: vi.fn() };
});

function blankPixels(): Float32Array {
  return new Float32Array(784);
}

/** Toile simulée avec un rectangle fixe, pour tester `pointerToCell` sans DOM. */
function mockCanvas(width: number, height: number, left = 0, top = 0): HTMLCanvasElement {
  return {
    getBoundingClientRect: () => ({
      left,
      top,
      width,
      height,
      right: left + width,
      bottom: top + height,
      x: left,
      y: top,
      toJSON: () => ({}),
    }),
  } as unknown as HTMLCanvasElement;
}

function pointer(clientX: number, clientY: number): PointerEvent {
  return { clientX, clientY } as PointerEvent;
}

/** Couche de convolution minimale : noyau plus biais de `outputChannels` zéros. */
function convLayer(kernel: Float32Array, outputChannels: number): MnistCnnLayerWeights {
  return { kernel, bias: new Float32Array(outputChannels) };
}

/** Poids minimaux dont seule `conv2` porte un noyau rempli, pour `selectedWeightValue`. */
function weightsWithConv2(kernel: Float32Array, outputChannels: number): MnistCnnWeights {
  return {
    conv: [
      convLayer(new Float32Array(9), 1),
      convLayer(kernel, outputChannels),
      convLayer(new Float32Array(9), 1),
      convLayer(new Float32Array(9), 1),
    ],
    dense: [convLayer(new Float32Array(9), 1), convLayer(new Float32Array(9), 1)],
  };
}

/** Géométrie d'une couche : `conv1` stride 1 sans remplissage, `conv2` stride 2 remplissage 1. */
const CONV1_GEOMETRY = { stride: 1, padding: 0, inputSize: 28 } as const;
const CONV2_GEOMETRY = { stride: 2, padding: 1, inputSize: 26 } as const;

/** Poids minimaux pour `trainingRunRequest` : la requête ne valide aucune forme. */
function minimalWeights(): MnistCnnWeights {
  const layer = (kernelLength: number, biasLength: number) => ({
    kernel: new Float32Array(kernelLength),
    bias: new Float32Array(biasLength),
  });
  return {
    conv: [layer(9, 1), layer(9, 1), layer(9, 1), layer(9, 1)],
    dense: [layer(200, 8), layer(80, 10)],
  };
}

/** Événement de case de noyau, avec le canal d'entrée de la liaison concernée. */
function filterCellEvent(
  layer: number,
  inputChannel: number,
  outputChannel: number,
  ky: number,
  kx: number,
): WeightEditorEvent {
  return { kind: 'filterCell', layer, inputChannel, outputChannel, ky, kx };
}

/**
 * Index synthétique de type QuickDraw : `shardsPerClass` fragments pleins par
 * classe et par fraction, numérotés dans l'ordre de l'index.
 */
function shardedIndexFixture(
  classCount: number,
  shardsPerClass: number,
  countPerShard = 1000,
): ShardedIndex {
  const split = (labelsFile: string): ShardedSplitIndex => {
    const shards: ShardEntry[] = [];
    let count = 0;
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      for (let shardIndex = 0; shardIndex < shardsPerClass; shardIndex += 1) {
        shards.push({
          file: `c${classIndex}-s${shardIndex}.idx.gz`,
          classIndex,
          start: count,
          count: countPerShard,
        });
        count += countPerShard;
      }
    }
    return { count, labelsFile, shards };
  };
  return {
    version: 2,
    dataset: 'quickdraw',
    classNames: Array.from({ length: classCount }, (_unused, index) => `classe-${index}`),
    shardSize: countPerShard,
    splits: {
      train: split('train-labels.idx1.gz'),
      test: split('test-labels.idx1.gz'),
    },
    galleries: {
      train: {
        imagesFile: 'train-gallery-images.idx3.gz',
        labelsFile: 'train-gallery-labels.idx1.gz',
        perClass: 10,
      },
      test: {
        imagesFile: 'test-gallery-images.idx3.gz',
        labelsFile: 'test-gallery-labels.idx1.gz',
        perClass: 10,
      },
    },
  };
}

describe('mapCanvasSize', () => {
  it('keeps the cell size at least four pixels and fits the maximum canvas', () => {
    expect(mapCellSize(26)).toBe(Math.floor(192 / 26));
    expect(mapCanvasSize(26)).toBe(26 * Math.floor(192 / 26));
    expect(mapCanvasSize(5)).toBe(5 * Math.floor(192 / 5));
    expect(mapCanvasSize(64)).toBe(64 * 4);
  });
});

describe('brushStrength', () => {
  it('is 1 at the stroke center and fades linearly to 0 at the brush radius', () => {
    expect(brushStrength(0)).toBe(1);
    expect(brushStrength(0.8)).toBeCloseTo(0.5);
    expect(brushStrength(1.6)).toBe(0);
    expect(brushStrength(3)).toBe(0);
  });
});

describe('paintDisc', () => {
  it('paints a soft disc around the stroke point', () => {
    const pixels = blankPixels();
    paintDisc(pixels, 10.5, 10.5);
    expect(pixels[10 * 28 + 10]).toBeGreaterThan(0.9);
    expect(pixels[10 * 28 + 9]).toBeGreaterThan(0);
    expect(pixels[5 * 28 + 5]).toBe(0);
  });

  it('keeps the darkest value when strokes overlap', () => {
    const pixels = blankPixels();
    paintDisc(pixels, 5.5, 5.5);
    const first = pixels[5 * 28 + 5];
    paintDisc(pixels, 6.5, 6.5);
    expect(pixels[5 * 28 + 5]).toBe(first);
  });

  it('stays inside the 28×28 grid on edge strokes', () => {
    const pixels = blankPixels();
    paintDisc(pixels, 27.9, 0.1);
    expect(pixels[0 * 28 + 27]).toBeGreaterThan(0);
    expect(pixels[3 * 28 + 27]).toBe(0);
  });
});

describe('filterKernelMean', () => {
  it('returns the 3×3 kernel when the layer has one input channel', () => {
    // HWIO flat layout: 9 positions × 1 input channel × 2 output channels.
    const kernel = new Float32Array(18);
    for (let position = 0; position < 9; position += 1) {
      kernel[position * 2] = position + 1;
      kernel[position * 2 + 1] = 100;
    }
    expect(Array.from(filterKernelMean(kernel, 1, 0))).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(Array.from(filterKernelMean(kernel, 1, 1))).toEqual([
      100, 100, 100, 100, 100, 100, 100, 100, 100,
    ]);
  });

  it('averages the input channels for multi-channel kernels', () => {
    // HWIO flat layout: 9 positions × 3 input channels × 1 output channel;
    // channel ic contributes (ic + 1) × 10 + position.
    const kernel = new Float32Array(27);
    for (let position = 0; position < 9; position += 1) {
      for (let ic = 0; ic < 3; ic += 1) {
        kernel[position * 3 + ic] = (ic + 1) * 10 + position;
      }
    }
    const means = filterKernelMean(kernel, 3, 0);
    expect(means[0]).toBeCloseTo((10 + 20 + 30) / 3);
    expect(means[8]).toBeCloseTo((18 + 28 + 38) / 3);
  });
});

describe('argmaxIndex', () => {
  it('returns the index of the largest value', () => {
    expect(argmaxIndex(new Float32Array([0.1, 0.4, 0.3, 0.2]))).toBe(1);
  });

  it('returns the first index on ties', () => {
    expect(argmaxIndex(new Float32Array([0.5, 0.5, 0.1]))).toBe(0);
  });
});

describe('hexToRgb', () => {
  it('parses six-digit hex colors', () => {
    expect(hexToRgb('#2563eb')).toEqual([37, 99, 235]);
  });

  it('falls back to the accent blue on malformed input', () => {
    expect(hexToRgb('blue')).toEqual([37, 99, 235]);
  });
});

describe('mixRgb', () => {
  it('interpolates between two colors', () => {
    expect(mixRgb([255, 255, 255], [0, 0, 0], 0.5)).toEqual([128, 128, 128]);
    expect(mixRgb([255, 255, 255], [255, 0, 0], 0)).toEqual([255, 255, 255]);
  });
});

describe('activationRgb', () => {
  it('maps 0 to red, 0.5 to neutral grey and 1 to blue', () => {
    expect(activationRgb(0)).toEqual([215, 48, 39]);
    expect(activationRgb(0.5)).toEqual([128, 128, 128]);
    expect(activationRgb(1)).toEqual([69, 117, 180]);
  });

  it('interpolates from each end to the neutral grey', () => {
    expect(activationRgb(0.25)).toEqual(mixRgb([215, 48, 39], [128, 128, 128], 0.5));
    expect(activationRgb(0.75)).toEqual(mixRgb([128, 128, 128], [69, 117, 180], 0.5));
  });

  it('clamps values outside the unit interval', () => {
    expect(activationRgb(-1)).toEqual([215, 48, 39]);
    expect(activationRgb(2)).toEqual([69, 117, 180]);
  });

  it('centres the tanh scale on zero', () => {
    expect(activationRgb(0, 'tanh')).toEqual([128, 128, 128]);
    expect(activationRgb(-1, 'tanh')).toEqual([215, 48, 39]);
    expect(activationRgb(1, 'tanh')).toEqual([69, 117, 180]);
  });

  it('interpolates the tanh scale on both sides of zero', () => {
    expect(activationRgb(-0.5, 'tanh')).toEqual(mixRgb([215, 48, 39], [128, 128, 128], 0.5));
    expect(activationRgb(0.5, 'tanh')).toEqual(mixRgb([128, 128, 128], [69, 117, 180], 0.5));
  });

  it('clamps a tanh value outside its domain', () => {
    expect(activationRgb(-3, 'tanh')).toEqual([215, 48, 39]);
    expect(activationRgb(3, 'tanh')).toEqual([69, 117, 180]);
  });
});

describe('activationScaleLabels', () => {
  it('names the three stops of each activation domain', () => {
    expect(activationScaleLabels('sigmoid')).toEqual(['0', '0,5', '1']);
    expect(activationScaleLabels('tanh')).toEqual(['-1', '0', '1']);
  });

  it('describes the scale for screen readers', () => {
    expect(activationScaleAriaLabel('sigmoid')).toBe(
      'Échelle des cartes : rouge à 0, gris à 0,5, bleu à 1',
    );
    expect(activationScaleAriaLabel('tanh')).toBe(
      'Échelle des cartes : rouge à -1, gris à 0, bleu à 1',
    );
  });
});

describe('filterBiasCaption', () => {
  it('formats the channel number and the bias with a decimal comma', () => {
    expect(filterBiasCaption(0, 0.1254)).toBe('sortie 1 · b = 0,125');
    expect(filterBiasCaption(2, -0.0432)).toBe('sortie 3 · b = -0,043');
    expect(filterBiasCaption(4, 0)).toBe('sortie 5 · b = 0,000');
  });
});

describe('dense activation labels', () => {
  it('names hidden layers and the output', () => {
    expect(denseLayerTitle(0, 2, 8)).toBe('Dense 1 — 8 unités');
    expect(denseLayerTitle(1, 2, 10)).toBe('Sortie — 10 classes');
  });

  it('labels a hidden unit with three decimals', () => {
    expect(denseCellLabel(2, 0.4234)).toBe('unité 3 : 0,423');
    expect(denseCellLabel(0, -0.5)).toBe('unité 1 : -0,500');
  });

  it('labels an output class with a percentage and the maximum', () => {
    expect(outputCellLabel(7, 0.9823, true)).toBe('classe 7 : 98,2 %, maximum');
    expect(outputCellLabel(0, 0.0012, false)).toBe('classe 0 : 0,1 %');
    expect(outputCellLabel(2, 0.5, false)).toBe('classe 2 : 50,0 %');
    expect(outputCellLabel(2, 0.5, false, 'chat')).toBe('classe 2 (chat) : 50,0 %');
    expect(outputCellLabel(2, 0.5, true, 'chat')).toBe('classe 2 (chat) : 50,0 %, maximum');
  });
});

describe('galleryClassLabel', () => {
  it('keeps plain digits for MNIST', () => {
    expect(galleryClassLabel(getDataset('mnist'), 3)).toBe('3');
  });

  it('pairs the index with the class name for named datasets', () => {
    expect(galleryClassLabel(getDataset('fashion-mnist'), 0)).toBe('0 — T-shirt');
    expect(galleryClassLabel(getDataset('quickdraw'), 2)).toBe('2 — chat');
  });

  it('falls back to the index when the class has no name', () => {
    expect(galleryClassLabel(getDataset('mnist'), 10)).toBe('10');
  });

  it('uses the English class names under the English locale', () => {
    expect(galleryClassLabel(getDataset('fashion-mnist'), 1, 'fr')).toContain('Pantalon');
    expect(galleryClassLabel(getDataset('fashion-mnist'), 1, 'en')).toContain('Trouser');
    expect(galleryClassLabel(getDataset('quickdraw'), 4, 'fr')).toContain('gâteau');
    expect(galleryClassLabel(getDataset('quickdraw'), 4, 'en')).toContain('cake');
  });

  it('keeps the registry class names when a dataset has no English names', () => {
    const { classNamesEn: _englishNames, ...withoutEnglishNames } = getDataset('fashion-mnist');
    expect(galleryClassLabel(withoutEnglishNames, 1, 'en')).toBe('1 — Pantalon');
  });
});

describe('galleryInputSource', () => {
  it('names the split, the dataset and the plain digit for MNIST', () => {
    expect(galleryInputSource(getDataset('mnist'), 3, 'test')).toBe(
      'Galerie de validation · Exemple MNIST (3)',
    );
    expect(galleryInputSource(getDataset('mnist'), 3, 'train')).toBe(
      "Galerie d'entraînement · Exemple MNIST (3)",
    );
  });

  it('names the dataset and the class for named datasets', () => {
    expect(galleryInputSource(getDataset('fashion-mnist'), 0, 'test')).toBe(
      'Galerie de validation · Exemple Fashion-MNIST (0 — T-shirt)',
    );
    expect(galleryInputSource(getDataset('quickdraw'), 2, 'test')).toBe(
      'Galerie de validation · Exemple QuickDraw (2 — chat)',
    );
    expect(galleryInputSource(getDataset('quickdraw'), 2, 'train')).toBe(
      "Galerie d'entraînement · Exemple QuickDraw (2 — chat)",
    );
  });
});

describe('galleryInputHint', () => {
  it('names the split, the dataset and the expected class', () => {
    expect(galleryInputHint(getDataset('mnist'), 3, 'test')).toBe(
      'Exemple de la galerie de validation MNIST : la classe attendue est « 3 ».',
    );
    expect(galleryInputHint(getDataset('mnist'), 3, 'train')).toBe(
      "Exemple de la galerie d'entraînement MNIST : la classe attendue est « 3 ».",
    );
    expect(galleryInputHint(getDataset('fashion-mnist'), 0, 'train')).toBe(
      "Exemple de la galerie d'entraînement Fashion-MNIST : la classe attendue est « T-shirt ».",
    );
    expect(galleryInputHint(getDataset('quickdraw'), 2, 'test')).toBe(
      'Exemple de la galerie de validation QuickDraw : la classe attendue est « chat ».',
    );
    expect(galleryInputHint(getDataset('quickdraw'), 2, 'train')).toBe(
      "Exemple de la galerie d'entraînement QuickDraw : la classe attendue est « chat ».",
    );
  });

  it('names the dataset in the drawing prompt when no class is chosen', () => {
    const hint = galleryInputHint(getDataset('quickdraw'), null, 'test');
    expect(hint).toContain('QuickDraw');
    expect(hint).toContain('Dessinez');
  });
});

describe('dense size labels', () => {
  it('names each hidden layer width control', () => {
    expect(denseSizeLabel(0)).toBe('Unités de la couche dense 1');
    expect(denseSizeLabel(1)).toBe('Unités de la couche dense 2');
  });

  it('shows the chosen width in the layer title', () => {
    expect(denseLayerTitle(0, 3, 7)).toBe('Dense 1 — 7 unités');
    expect(denseLayerTitle(2, 3, 10)).toBe('Sortie — 10 classes');
  });
});

describe('dense size bounds', () => {
  it('clamps and rounds a hidden width', () => {
    expect(clampDenseSize(0)).toBe(1);
    expect(clampDenseSize(129)).toBe(128);
    expect(clampDenseSize(7.6)).toBe(8);
    expect(clampDenseSize(32)).toBe(32);
  });

  it('recognises the reference hidden width only', () => {
    expect(isReferenceDenseSizes([8, 7])).toBe(true);
    expect(isReferenceDenseSizes([9, 8])).toBe(false);
    expect(isReferenceDenseSizes([128])).toBe(false);
  });
});

describe('dense pack columns', () => {
  it('scales the column count with the pack size', () => {
    expect(densePackColumns(1)).toBe(4);
    expect(densePackColumns(8)).toBe(4);
    expect(densePackColumns(32)).toBe(8);
    expect(densePackColumns(128)).toBe(16);
  });
});

describe('pointerToCell', () => {
  it('maps the canvas corners to the 28×28 grid corners', () => {
    const canvas = mockCanvas(280, 280, 10, 20);
    expect(pointerToCell(canvas, pointer(10, 20))).toEqual([0, 0]);
    expect(pointerToCell(canvas, pointer(290, 300))).toEqual([28, 28]);
  });

  it('maps the canvas center to the grid center', () => {
    const canvas = mockCanvas(280, 280, 10, 20);
    expect(pointerToCell(canvas, pointer(150, 160))).toEqual([14, 14]);
  });

  it('returns null for a zero-size canvas', () => {
    expect(pointerToCell(mockCanvas(0, 0), pointer(5, 5))).toBeNull();
    expect(pointerToCell(mockCanvas(0, 100), pointer(5, 5))).toBeNull();
    expect(pointerToCell(mockCanvas(100, 0), pointer(5, 5))).toBeNull();
  });
});

describe('stratifiedExampleCountEstimate', () => {
  it('estimates one tenth of the official split sizes', () => {
    expect(stratifiedExampleCountEstimate(60000, 0.1)).toBe(6000);
    expect(stratifiedExampleCountEstimate(10000, 0.1)).toBe(1000);
  });

  it('keeps at least one example per class for tiny fractions', () => {
    expect(stratifiedExampleCountEstimate(60000, 0.001)).toBe(60);
    expect(stratifiedExampleCountEstimate(10000, 0.001)).toBe(10);
    expect(stratifiedExampleCountEstimate(10000, 0.01)).toBe(100);
  });

  it('returns the full size for a fraction of one', () => {
    expect(stratifiedExampleCountEstimate(60000, 1)).toBe(60000);
    expect(stratifiedExampleCountEstimate(10000, 1)).toBe(10000);
  });

  it('rejects invalid totals and fractions', () => {
    expect(() => stratifiedExampleCountEstimate(5, 0.1)).toThrow(RangeError);
    expect(() => stratifiedExampleCountEstimate(1.5, 0.1)).toThrow(RangeError);
    expect(() => stratifiedExampleCountEstimate(60000, 0)).toThrow(RangeError);
    expect(() => stratifiedExampleCountEstimate(60000, 1.5)).toThrow(RangeError);
    expect(() => stratifiedExampleCountEstimate(60000, Number.NaN)).toThrow(RangeError);
  });

  it('uses the class count of the selected dataset', () => {
    // 5 classes : 60000 ÷ 5 × 0,1 = 1200 par classe, soit 6000 exemples.
    expect(stratifiedExampleCountEstimate(60000, 0.1, 5)).toBe(6000);
    // Le même total avec 10 classes donne un autre résultat sur 50 exemples.
    expect(stratifiedExampleCountEstimate(50, 0.1, 5)).toBe(5);
    expect(stratifiedExampleCountEstimate(50, 0.1)).toBe(10);
  });

  it('rejects a class count below two', () => {
    expect(() => stratifiedExampleCountEstimate(60000, 0.1, 1)).toThrow(RangeError);
    expect(() => stratifiedExampleCountEstimate(60000, 0.1, 2.5)).toThrow(RangeError);
  });
});

describe('clampPercent', () => {
  it('caps the selection percent at 100 and keeps the lower bound', () => {
    expect(clampPercent(150)).toBe(100);
    expect(clampPercent(100)).toBe(100);
    expect(clampPercent(0)).toBe(1);
    expect(clampPercent(42.4)).toBe(42);
  });
});

describe('segmentSelectionCount', () => {
  const allClasses = Array.from({ length: 15 }, (_unused, index) => index);

  it('sums exactly the shard entries that selectShardEntries keeps', () => {
    const index = shardedIndexFixture(15, 107);
    for (const percent of [1, 2, 10, 100]) {
      for (const classes of [[0, 1, 2, 3, 4], [2, 5], allClasses]) {
        const expected = selectShardEntries(index.splits.train, percent, classes).reduce(
          (total, entry) => total + entry.count,
          0,
        );
        expect(segmentSelectionCount(index, 'train', percent, classes)).toBe(expected);
      }
    }
  });

  it('reads the validation percent from the test split', () => {
    const index = shardedIndexFixture(15, 107);
    expect(segmentSelectionCount(index, 'test', 2, [0, 1, 2, 3, 4])).toBe(10_000);
    expect(segmentSelectionCount(index, 'test', 100, [0, 1, 2, 3, 4])).toBe(5 * 107 * 1000);
  });

  it('keeps a two-percent five-class selection on the whole path', () => {
    const index = shardedIndexFixture(15, 107);
    const small = segmentSelectionCount(index, 'train', 2, [0, 1, 2, 3, 4]);
    expect(small).toBe(10_000);
    expect(small).toBeLessThanOrEqual(SEGMENT_IMAGE_THRESHOLD);
    expect(shouldUseSegments(small)).toBe(false);
  });

  it('switches to segments at ten percent of all classes and at full size', () => {
    const index = shardedIndexFixture(15, 107);
    const tenPercent = segmentSelectionCount(index, 'train', 10, allClasses);
    const full = segmentSelectionCount(index, 'train', 100, allClasses);
    expect(tenPercent).toBe(15 * 11 * 1000);
    expect(full).toBe(15 * 107 * 1000);
    expect(shouldUseSegments(tenPercent)).toBe(true);
    expect(shouldUseSegments(full)).toBe(true);
  });
});

describe('segmentRunNeeded', () => {
  it('segments when the validation split alone exceeds the threshold', () => {
    expect(segmentRunNeeded(30_000, 100_000)).toBe(true);
  });

  it('keeps the whole path when both splits stay under the threshold', () => {
    expect(segmentRunNeeded(30_000, 15_000)).toBe(false);
    expect(segmentRunNeeded(60_000, 60_000)).toBe(false);
  });

  it('segments as soon as either split passes the threshold', () => {
    expect(segmentRunNeeded(60_000, 60_001)).toBe(true);
    expect(segmentRunNeeded(60_001, 60_000)).toBe(true);
  });
});

describe('groupEntriesByNetworkClass', () => {
  function entry(classIndex: number, file: string): RoundEntry {
    return { file, classIndex, start: 0, count: 10 };
  }

  it('groups entries by network position and keeps the pinned order', () => {
    const entries = [entry(2, 'c2-a'), entry(2, 'c2-b'), entry(5, 'c5-a')];
    const groups = groupEntriesByNetworkClass(entries, [2, 5]);
    expect(groups).toHaveLength(2);
    expect(groups[0].map((item) => item.file)).toEqual(['c2-a', 'c2-b']);
    expect(groups[1].map((item) => item.file)).toEqual(['c5-a']);
    expect(groups[0].every((item) => item.classIndex === 2)).toBe(true);
    expect(groups[1].every((item) => item.classIndex === 5)).toBe(true);
  });

  it('follows the order of the selected classes, not the original class order', () => {
    const groups = groupEntriesByNetworkClass([entry(2, 'c2-a'), entry(5, 'c5-a')], [5, 2]);
    expect(groups.map((group) => group[0].classIndex)).toEqual([5, 2]);
  });

  it('keeps an empty group for a selected class without entries', () => {
    const groups = groupEntriesByNetworkClass([entry(2, 'c2-a')], [2, 5]);
    expect(groups[0]).toHaveLength(1);
    expect(groups[1]).toEqual([]);
  });

  it('rejects a shard of a class that is not selected', () => {
    expect(() => groupEntriesByNetworkClass([entry(3, 'c3-a')], [2, 5])).toThrow(RangeError);
  });

  it('builds a plan whose total matches the pinned store on both splits', () => {
    const index = shardedIndexFixture(15, 107);
    const dataset = getDataset('quickdraw');
    if (dataset.kind !== 'sharded') {
      throw new Error('The QuickDraw dataset must be segmented.');
    }
    const classes = [0, 1, 2, 3, 4];
    for (const split of ['train', 'test'] as const) {
      const percent = split === 'train' ? 100 : 10;
      const entries = selectShardEntries(index.splits[split], percent, classes);
      const plan = buildSegmentPlan({
        shardsByClass: groupEntriesByNetworkClass(entries, classes),
        shardSize: index.shardSize,
        seed: 42,
      });
      const store = new SegmentStore({
        dataset,
        index,
        classIndices: classes,
        entries,
        cacheName: 'drl-datasets-v2-quickdraw',
        split,
      });
      expect(plan.totalCount).toBe(store.totalCount);
      expect(store.totalCount).toBe(segmentSelectionCount(index, split, percent, classes));
    }
  });
});

describe('displayClass', () => {
  it('maps a network output index to the original class index', () => {
    const enabled = [0, 2, 7];
    expect(displayClass(enabled, 0)).toBe(0);
    expect(displayClass(enabled, 1)).toBe(2);
    expect(displayClass(enabled, 2)).toBe(7);
  });
});

describe('pointerToGrid', () => {
  it('maps the canvas corners to the grid corners', () => {
    const canvas = mockCanvas(30, 30, 10, 20);
    expect(pointerToGrid(canvas, pointer(10, 20), 3)).toEqual([0, 0]);
    expect(pointerToGrid(canvas, pointer(40, 50), 3)).toEqual([3, 3]);
  });

  it('returns fractional coordinates at the canvas center', () => {
    const canvas = mockCanvas(30, 30, 10, 20);
    expect(pointerToGrid(canvas, pointer(25, 35), 3)).toEqual([1.5, 1.5]);
  });

  it('returns null for a zero-size canvas', () => {
    expect(pointerToGrid(mockCanvas(0, 0), pointer(5, 5), 3)).toBeNull();
    expect(pointerToGrid(mockCanvas(0, 10), pointer(5, 5), 3)).toBeNull();
    expect(pointerToGrid(mockCanvas(10, 0), pointer(5, 5), 3)).toBeNull();
  });

  it('backs pointerToCell, which keeps its 28×28 results', () => {
    const canvas = mockCanvas(280, 280, 10, 20);
    expect(pointerToCell(canvas, pointer(10, 20))).toEqual([0, 0]);
    expect(pointerToCell(canvas, pointer(290, 300))).toEqual([28, 28]);
    expect(pointerToCell(canvas, pointer(150, 160))).toEqual([14, 14]);
    expect(pointerToCell(mockCanvas(0, 0), pointer(5, 5))).toBeNull();
  });
});

describe('filterKernelSlice', () => {
  it('returns the exact 3×3 kernel of one input and output channel pair', () => {
    // HWIO flat layout: 9 positions × 2 input channels × 2 output channels.
    const kernel = new Float32Array(36);
    for (let position = 0; position < 9; position += 1) {
      for (let ic = 0; ic < 2; ic += 1) {
        for (let oc = 0; oc < 2; oc += 1) {
          kernel[(position * 2 + ic) * 2 + oc] = position * 10 + ic * 3 + oc;
        }
      }
    }
    expect(Array.from(filterKernelSlice(kernel, 2, 1, 0))).toEqual([
      1, 11, 21, 31, 41, 51, 61, 71, 81,
    ]);
    expect(Array.from(filterKernelSlice(kernel, 2, 0, 1))).toEqual([
      3, 13, 23, 33, 43, 53, 63, 73, 83,
    ]);
  });

  it('differs from filterKernelMean on a multi-channel layer', () => {
    // HWIO flat layout: 9 positions × 3 input channels × 1 output channel ;
    // channel ic contributes (ic + 1) × 10 + position.
    const kernel = new Float32Array(27);
    for (let position = 0; position < 9; position += 1) {
      for (let ic = 0; ic < 3; ic += 1) {
        kernel[position * 3 + ic] = (ic + 1) * 10 + position;
      }
    }
    expect(Array.from(filterKernelSlice(kernel, 3, 0, 0))).toEqual([
      10, 11, 12, 13, 14, 15, 16, 17, 18,
    ]);
    expect(Array.from(filterKernelMean(kernel, 3, 0))).not.toEqual(
      Array.from(filterKernelSlice(kernel, 3, 0, 0)),
    );
  });

  it('throws on inconsistent lengths and out-of-range indices', () => {
    const kernel = new Float32Array(27);
    expect(() => filterKernelSlice(kernel, 0, 0, 0)).toThrow(RangeError);
    expect(() => filterKernelSlice(kernel, 2, 0, 0)).toThrow(RangeError);
    expect(() => filterKernelSlice(kernel, 3, 1, 0)).toThrow(RangeError);
    expect(() => filterKernelSlice(kernel, 3, -1, 0)).toThrow(RangeError);
    expect(() => filterKernelSlice(kernel, 3, 0, 3)).toThrow(RangeError);
    expect(() => filterKernelSlice(kernel, 3, 0, -1)).toThrow(RangeError);
  });
});

describe('applyWeightEditorEvent', () => {
  it('selects a link and toggles it off on a second click', () => {
    const event = { kind: 'link', boundary: 1, inputChannel: 0, outputChannel: 2 } as const;
    const selection = applyWeightEditorEvent(null, event);
    expect(selection).toEqual(event);
    expect(applyWeightEditorEvent(selection, event)).toBeNull();
  });

  it('turns a filter cell click under a matching link into a weight selection', () => {
    const selection = applyWeightEditorEvent(null, {
      kind: 'link',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
    });
    expect(applyWeightEditorEvent(selection, filterCellEvent(1, 0, 2, 1, 0))).toEqual({
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 0,
    });
  });

  it('ignores a filter cell click without a link selection', () => {
    expect(applyWeightEditorEvent(null, filterCellEvent(1, 0, 2, 1, 0))).toBeNull();
  });

  it('ignores a filter cell click under a link of another boundary', () => {
    const selection = applyWeightEditorEvent(null, {
      kind: 'link',
      boundary: 0,
      inputChannel: 0,
      outputChannel: 2,
    });
    expect(applyWeightEditorEvent(selection, filterCellEvent(2, 0, 2, 1, 0))).toEqual(selection);
  });

  it('replaces a weight selection with a link selection', () => {
    const selection = applyWeightEditorEvent(
      { kind: 'link', boundary: 1, inputChannel: 0, outputChannel: 2 },
      filterCellEvent(1, 0, 2, 1, 0),
    );
    expect(selection).toEqual({
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 0,
    });
    expect(
      applyWeightEditorEvent(selection, {
        kind: 'link',
        boundary: 2,
        inputChannel: 2,
        outputChannel: 0,
      }),
    ).toEqual({ kind: 'link', boundary: 2, inputChannel: 2, outputChannel: 0 });
  });

  it('ignores a filter cell click under a weight selection', () => {
    const selection = applyWeightEditorEvent(
      { kind: 'link', boundary: 1, inputChannel: 0, outputChannel: 2 },
      filterCellEvent(1, 0, 2, 1, 0),
    );
    // La frontière 1 et le canal de sortie 2 correspondent : seul le genre de la
    // sélection (poids, pas liaison) fait ignorer le clic.
    expect(applyWeightEditorEvent(selection, filterCellEvent(1, 0, 2, 2, 2))).toEqual(selection);
  });
});

describe('filterViewMode', () => {
  it('is mean by default', () => {
    expect(filterViewMode(null, 1, 0)).toEqual({ mode: 'mean' });
    expect(
      filterViewMode(
        { kind: 'weight', boundary: 1, inputChannel: 0, outputChannel: 0, ky: 0, kx: 0 },
        2,
        0,
      ),
    ).toEqual({ mode: 'mean' });
  });

  it('is kernel only for the canvas matching a link selection', () => {
    const selection = { kind: 'link', boundary: 1, inputChannel: 2, outputChannel: 0 } as const;
    expect(filterViewMode(selection, 1, 0)).toEqual({ mode: 'kernel', inputChannel: 2 });
    expect(filterViewMode(selection, 2, 0)).toEqual({ mode: 'mean' });
    expect(filterViewMode(selection, 1, 1)).toEqual({ mode: 'mean' });
  });

  it('is kernel only for the canvas matching a weight selection', () => {
    const selection = {
      kind: 'weight',
      boundary: 0,
      inputChannel: 0,
      outputChannel: 1,
      ky: 2,
      kx: 2,
    } as const;
    expect(filterViewMode(selection, 0, 1)).toEqual({ mode: 'kernel', inputChannel: 0 });
    expect(filterViewMode(selection, 1, 1)).toEqual({ mode: 'mean' });
  });
});

describe('convLayerGeometry', () => {
  it('reads the stride, padding and input size of every layer', () => {
    expect(convLayerGeometry(0)).toEqual(CONV1_GEOMETRY);
    expect(convLayerGeometry(1)).toEqual(CONV2_GEOMETRY);
    expect(convLayerGeometry(2)).toEqual({ stride: 1, padding: 0, inputSize: 13 });
    expect(convLayerGeometry(3)).toEqual({ stride: 2, padding: 0, inputSize: 11 });
  });

  it('rejects a layer outside the architecture', () => {
    expect(() => convLayerGeometry(-1)).toThrow(RangeError);
    expect(() => convLayerGeometry(4)).toThrow(RangeError);
    expect(() => convLayerGeometry(1.5)).toThrow(RangeError);
  });
});

describe('filterCanvasInputChannel', () => {
  it('returns null when nothing is selected', () => {
    expect(filterCanvasInputChannel(null, 1, 2)).toBeNull();
  });

  it('returns the input channel of the link that designates the canvas', () => {
    const selection = { kind: 'link', boundary: 1, inputChannel: 0, outputChannel: 2 } as const;
    expect(filterCanvasInputChannel(selection, 1, 2)).toBe(0);
    expect(filterCanvasInputChannel(selection, 1, 1)).toBeNull();
    expect(filterCanvasInputChannel(selection, 2, 2)).toBeNull();
    expect(filterCanvasInputChannel(selection, 3, 2)).toBeNull();
  });

  it('returns null for a weight selection, which refines a link', () => {
    const selection = {
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 1,
    } as const;
    expect(filterCanvasInputChannel(selection, 1, 2)).toBeNull();
  });
});

describe('filterKernelValues', () => {
  /**
   * Noyau de `conv2` (2 canaux d'entrée, 3 canaux de sortie) où la valeur d'un
   * poids est `position × 10 + ic × 3 + oc` : la tranche d'un canal se lit
   * directement, la moyenne sur les canaux d'entrée d'un canal de sortie `oc`
   * vaut `position × 10 + oc + 1,5`.
   */
  function conv2Kernel(): Float32Array {
    const kernel = new Float32Array(54);
    for (let position = 0; position < 9; position += 1) {
      for (let ic = 0; ic < 2; ic += 1) {
        for (let oc = 0; oc < 3; oc += 1) {
          kernel[position * 6 + ic * 3 + oc] = position * 10 + ic * 3 + oc;
        }
      }
    }
    return kernel;
  }

  it('averages the input channels for a mean view', () => {
    const values = filterKernelValues({ mode: 'mean' }, conv2Kernel(), 2, 1);
    expect(Array.from(values)).toEqual([2.5, 12.5, 22.5, 32.5, 42.5, 52.5, 62.5, 72.5, 82.5]);
  });

  it('slices the displayed input channel for a kernel view', () => {
    const kernel = conv2Kernel();
    const slice = filterKernelValues({ mode: 'kernel', inputChannel: 0 }, kernel, 2, 1);
    expect(Array.from(slice)).toEqual([1, 11, 21, 31, 41, 51, 61, 71, 81]);
    const other = filterKernelValues({ mode: 'kernel', inputChannel: 1 }, kernel, 2, 2);
    expect(Array.from(other)).toEqual([5, 15, 25, 35, 45, 55, 65, 75, 85]);
  });

  it('falls back to the mean when the displayed input channel is out of range', () => {
    // 2 canaux d'entrée mais 5 canaux de sortie : la toile du canal de sortie 4
    // n'a pas de canal d'entrée 4.
    const kernel = new Float32Array(90);
    for (let index = 0; index < kernel.length; index += 1) {
      kernel[index] = index;
    }
    const values = filterKernelValues({ mode: 'kernel', inputChannel: 4 }, kernel, 2, 4);
    expect(Array.from(values)).toEqual(Array.from(filterKernelMean(kernel, 2, 4)));
  });

  it('falls back to the mean for a kernel view with an out-of-range input channel', () => {
    const kernel = conv2Kernel();
    const values = filterKernelValues({ mode: 'kernel', inputChannel: 2 }, kernel, 2, 1);
    expect(Array.from(values)).toEqual(Array.from(filterKernelMean(kernel, 2, 1)));
  });
});

describe('linkLineState', () => {
  it('is normal when nothing is selected', () => {
    expect(linkLineState(null, 0, 0, 1)).toBe('normal');
  });

  it('is selected for the selected link', () => {
    const selection = { kind: 'link', boundary: 1, inputChannel: 0, outputChannel: 2 } as const;
    expect(linkLineState(selection, 1, 0, 2)).toBe('selected');
    expect(linkLineState(selection, 1, 0, 1)).toBe('dimmed');
  });

  it('is selected for the line of a weight selection', () => {
    const selection = {
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 1,
    } as const;
    expect(linkLineState(selection, 1, 0, 2)).toBe('selected');
    expect(linkLineState(selection, 1, 0, 1)).toBe('dimmed');
    expect(linkLineState(selection, 1, 1, 2)).toBe('dimmed');
  });

  it('dims every link as soon as a selection exists', () => {
    const selection = { kind: 'link', boundary: 1, inputChannel: 0, outputChannel: 2 } as const;
    expect(linkLineState(selection, 0, 0, 1)).toBe('dimmed');
    expect(linkLineState(selection, 1, 0, 1)).toBe('dimmed');
    expect(linkLineState(selection, 2, 0, 1)).toBe('dimmed');
  });
});

describe('linkCanvasSelectors', () => {
  it('binds the upstream output channel to the downstream input channel', () => {
    expect(linkCanvasSelectors({ boundary: 1, inputChannel: 0, outputChannel: 2 })).toEqual({
      source: 'canvas.drl-canvas--filter[data-layer="0"][data-channel="0"]',
      target: 'canvas.drl-canvas--filter[data-layer="1"][data-channel="2"]',
    });
    expect(linkCanvasSelectors({ boundary: 3, inputChannel: 2, outputChannel: 3 })).toEqual({
      source: 'canvas.drl-canvas--filter[data-layer="2"][data-channel="2"]',
      target: 'canvas.drl-canvas--filter[data-layer="3"][data-channel="3"]',
    });
  });

  it('rejects the boundary 0, which has no upstream canvas', () => {
    expect(() => linkCanvasSelectors({ boundary: 0, inputChannel: 0, outputChannel: 0 })).toThrow(
      RangeError,
    );
  });
});

describe('filterLinkGeometry', () => {
  const CONTAINER: RectLike = { left: 5, top: 7, width: 500, height: 400 };

  it('returns the segment of linkGeometry for usable rectangles', () => {
    const source: RectLike = { left: 10, top: 20, width: 40, height: 30 };
    const target: RectLike = { left: 100, top: 200, width: 20, height: 10 };
    expect(filterLinkGeometry(source, target, CONTAINER)).toEqual(
      linkGeometry(source, target, CONTAINER),
    );
  });

  it('returns null instead of throwing for an unusable rectangle', () => {
    const rect: RectLike = { left: 0, top: 0, width: 10, height: 10 };
    // linkGeometry lève dans chacun de ces cas : la ligne est ignorée, pas tracée.
    expect(filterLinkGeometry({ ...rect, width: 0 }, rect, CONTAINER)).toBeNull();
    expect(filterLinkGeometry(rect, { ...rect, height: 0 }, CONTAINER)).toBeNull();
    expect(filterLinkGeometry(rect, rect, { ...CONTAINER, top: Number.NaN })).toBeNull();
    expect(
      filterLinkGeometry(rect, rect, { ...CONTAINER, left: Number.POSITIVE_INFINITY }),
    ).toBeNull();
  });
});

describe('linkGroupIndices', () => {
  /** `<g>` simulé par ses trois attributs de données. */
  function mockGroup(attributes: Record<string, string | null>): Element {
    return {
      getAttribute: (name: string) => attributes[name] ?? null,
    } as unknown as Element;
  }

  function validGroup(): Record<string, string | null> {
    return { 'data-boundary': '1', 'data-input-channel': '2', 'data-output-channel': '0' };
  }

  it('reads the boundary and the two channels from the data attributes', () => {
    expect(linkGroupIndices(mockGroup(validGroup()))).toEqual({
      boundary: 1,
      inputChannel: 2,
      outputChannel: 0,
    });
  });

  it('returns null for a missing attribute', () => {
    const missingChannel = validGroup();
    delete missingChannel['data-input-channel'];
    expect(linkGroupIndices(mockGroup(missingChannel))).toBeNull();
    expect(linkGroupIndices(mockGroup({}))).toBeNull();
  });

  it('returns null for an empty or non-integer attribute', () => {
    for (const value of ['', 'x', '1.5', '-1', 'NaN', 'Infinity']) {
      const group = validGroup();
      group['data-boundary'] = value;
      expect(linkGroupIndices(mockGroup(group))).toBeNull();
    }
  });

  it('returns null for the boundary 0, which has no upstream canvas', () => {
    // `linkCanvasSelectors(0, …)` lèverait une `RangeError` : un `<g>` qui
    // porterait `data-boundary="0"` doit être ignoré plutôt que tracé.
    const group = validGroup();
    group['data-boundary'] = '0';
    expect(linkGroupIndices(mockGroup(group))).toBeNull();
    // Les frontières 1 à 3 restent acceptées, avec leurs deux canaux.
    for (const boundary of [1, 2, 3]) {
      const valid = validGroup();
      valid['data-boundary'] = String(boundary);
      expect(linkGroupIndices(mockGroup(valid))).toEqual({
        boundary,
        inputChannel: 2,
        outputChannel: 0,
      });
    }
  });
});

describe('describeWeightSelection', () => {
  const filters = [3, 3, 3, 3] as MnistCnnFilters;

  it('describes a weight selection', () => {
    expect(
      describeWeightSelection(
        { kind: 'weight', boundary: 1, inputChannel: 0, outputChannel: 2, ky: 2, kx: 1 },
        { filters },
      ),
    ).toBe('conv2 — entrée 1, sortie 3, position (3, 2)');
  });

  it('describes a link selection', () => {
    expect(
      describeWeightSelection(
        { kind: 'link', boundary: 0, inputChannel: 0, outputChannel: 1 },
        { filters },
      ),
    ).toBe('conv1 — entrée 1, sortie 2');
  });

  it('returns null for no selection', () => {
    expect(describeWeightSelection(null, { filters })).toBeNull();
  });

  it('returns null for a selection outside the architecture', () => {
    expect(
      describeWeightSelection(
        { kind: 'link', boundary: 0, inputChannel: 0, outputChannel: 5 },
        { filters },
      ),
    ).toBeNull();
  });
});

describe('describeLinkLabel', () => {
  const filters = [3, 3, 3, 3] as MnistCnnFilters;

  it('returns the plain pair name when nothing scopes the line', () => {
    expect(describeLinkLabel(null, 1, 0, 2, filters, 4)).toBe('conv2 — entrée 1, sortie 3');
    expect(
      describeLinkLabel(
        { kind: 'link', boundary: 0, inputChannel: 0, outputChannel: 1 },
        1,
        0,
        2,
        filters,
        4,
      ),
    ).toBe('conv2 — entrée 1, sortie 3');
  });

  it('states a single weight for the line of a weight selection', () => {
    const selection = {
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 0,
    } as const;
    expect(describeLinkLabel(selection, 1, 0, 2, filters, 4)).toBe(
      'conv2 — entrée 1, sortie 3 · 1 poids sur 9',
    );
    expect(describeLinkLabel(selection, 1, 0, 1, filters, 4)).toBe('conv2 — entrée 1, sortie 2');
  });
});

describe('selectedWeightValue', () => {
  const filters = [3, 3, 3, 3] as MnistCnnFilters;

  it('reads the value of the selected weight', () => {
    // conv2 with 2 input channels and 3 output channels: 9 × 2 × 3 = 54 weights.
    const kernel = new Float32Array(54);
    const index = (2 * 3 + 0) * 2 * 3 + 1 * 3 + 2;
    kernel[index] = 0.5;
    const weights = weightsWithConv2(kernel, 3);
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 1,
          outputChannel: 2,
          ky: 2,
          kx: 0,
        },
        weights,
        { filters },
      ),
    ).toBe(0.5);
  });

  it('returns null for a non-weight selection', () => {
    const weights = weightsWithConv2(new Float32Array(54), 3);
    expect(selectedWeightValue(null, weights, { filters })).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'link',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 0,
        },
        weights,
        { filters },
      ),
    ).toBeNull();
  });

  it('returns null for out-of-range indices', () => {
    const weights = weightsWithConv2(new Float32Array(54), 3);
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 5,
          outputChannel: 0,
          ky: 0,
          kx: 0,
        },
        weights,
        { filters },
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 9,
          inputChannel: 0,
          outputChannel: 0,
          ky: 0,
          kx: 0,
        },
        weights,
        { filters },
      ),
    ).toBeNull();
  });
});

describe('parseWeightInput', () => {
  it('accepts digits with a dot or comma decimal separator and a leading sign', () => {
    expect(parseWeightInput('1.5')).toBe(1.5);
    expect(parseWeightInput('1,5')).toBe(1.5);
    expect(parseWeightInput('-0.25')).toBe(-0.25);
    expect(parseWeightInput('+2')).toBe(2);
    expect(parseWeightInput('0')).toBe(0);
    expect(parseWeightInput('-0')).toBe(0);
    expect(Object.is(parseWeightInput('-0'), -0)).toBe(false);
  });

  it('rejects anything else', () => {
    expect(parseWeightInput('abc')).toBeNull();
    expect(parseWeightInput('1.2.3')).toBeNull();
    expect(parseWeightInput('')).toBeNull();
    expect(parseWeightInput('NaN')).toBeNull();
    expect(parseWeightInput('1.')).toBeNull();
    expect(parseWeightInput('.5')).toBeNull();
    expect(parseWeightInput(' 1.5')).toBeNull();
    expect(parseWeightInput('1 5')).toBeNull();
    expect(parseWeightInput('1,')).toBeNull();
    expect(parseWeightInput(',')).toBeNull();
    expect(parseWeightInput(',5')).toBeNull();
    expect(parseWeightInput('-')).toBeNull();
  });
});

describe('trainingRunRequest', () => {
  const filters = [2, 3, 4, 5] as MnistCnnFilters;
  const weights = minimalWeights();

  function input(overrides: Partial<TrainingRunInput> = {}): TrainingRunInput {
    return {
      filters,
      convLayers: 4,
      denseLayers: 2,
      denseSizes: [8, 8],
      outputClasses: 10,
      enabledClasses: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      learningRate: 0.01,
      batchSize: 64,
      trainPercent: 25,
      validationPercent: 15,
      epochs: 4,
      weights,
      activation: 'sigmoid',
      ...overrides,
    };
  }

  it('returns the exact weights it was given as initialWeights', () => {
    expect(trainingRunRequest(input()).initialWeights).toBe(weights);
  });

  it('forwards the training parameters from the input', () => {
    const request = trainingRunRequest(
      input({
        epochs: 9,
        trainPercent: 40,
        validationPercent: 20,
        learningRate: 0.02,
        batchSize: 32,
      }),
    );
    expect(request.epochs).toBe(9);
    expect(request.trainPercent).toBe(40);
    expect(request.validationPercent).toBe(20);
    expect(request.learningRate).toBe(0.02);
    expect(request.batchSize).toBe(32);
    expect(request.filters).toBe(filters);
    expect(request.activation).toBe('sigmoid');
  });

  it('pins the seed', () => {
    const request = trainingRunRequest(input());
    expect(request.seed).toBe(42);
  });

  it('forwards the architecture depth from the input', () => {
    const request = trainingRunRequest(input({ convLayers: 2, denseLayers: 3 }));
    expect(request.convLayers).toBe(2);
    expect(request.denseLayers).toBe(3);
  });

  it('forwards the dense sizes from the input', () => {
    const request = trainingRunRequest(input({ denseSizes: [3, 7] }));
    expect(request.denseSizes).toEqual([3, 7]);
  });

  it('forwards the output class count from the input', () => {
    expect(trainingRunRequest(input()).outputClasses).toBe(10);
    expect(trainingRunRequest(input({ outputClasses: 20 })).outputClasses).toBe(20);
    // QuickDraw : quinze classes de sortie.
    expect(trainingRunRequest(input({ outputClasses: 5 })).outputClasses).toBe(5);
  });

  it('forwards the enabled classes and the checked output width', () => {
    const request = trainingRunRequest(input({ outputClasses: 3, enabledClasses: [0, 2, 7] }));
    expect(request.outputClasses).toBe(3);
    expect(request.enabledClasses).toEqual([0, 2, 7]);
  });

  it('forwards the batch size from the input', () => {
    const request = trainingRunRequest(input({ batchSize: 32 }));
    expect(request.batchSize).toBe(32);
  });

  it('forwards the activation from the input', () => {
    const request = trainingRunRequest(input({ activation: 'tanh' }));
    expect(request.activation).toBe('tanh');
  });

  it('forwards sigmoid from the input', () => {
    expect(trainingRunRequest(input()).activation).toBe('sigmoid');
  });
});

describe('loadSplitsForDataset', () => {
  function idxDataset(count: number): MnistIdxDataset {
    return {
      count,
      images: new Uint8Array(count * 784),
      labels: new Uint8Array(count),
    };
  }

  it('passes the enabled classes to the sharded loader', async () => {
    const splits: MnistSplits = { train: idxDataset(10), test: idxDataset(5) };
    const loader = vi.mocked(loadShardedSplits);
    loader.mockResolvedValueOnce(splits);
    const dataset = getDataset('quickdraw');
    const enabledClasses = [0, 2, 4];

    await expect(loadSplitsForDataset(dataset, 2, 3, enabledClasses)).resolves.toBe(splits);
    // Le cinquième argument est la sélection de classes : sans lui, les libellés
    // remontent jusqu'à 14 classes alors que la sortie en compte `enabledClasses.length`.
    expect(loader).toHaveBeenCalledWith(dataset, 2, 3, undefined, enabledClasses);
  });

  it('leaves the enabled classes out of the IDX loader call', async () => {
    const splits: MnistSplits = { train: idxDataset(10), test: idxDataset(5) };
    const loader = vi.mocked(loadDatasetSplits);
    loader.mockResolvedValueOnce(splits);
    const dataset = getDataset('mnist');

    await expect(loadSplitsForDataset(dataset, 2, 3, [0, 2])).resolves.toBe(splits);
    expect(loader).toHaveBeenCalledWith(dataset);
  });
});

describe('nextGalleryClass', () => {
  it('selects the first enabled class when no class was shown', () => {
    expect(nextGalleryClass([0, 1, 2], null)).toBe(0);
  });

  it('keeps the current class when it is still enabled', () => {
    expect(nextGalleryClass([0, 2, 4], 2)).toBe(2);
  });

  it('moves to the first enabled class when the current one is disabled', () => {
    expect(nextGalleryClass([0, 2, 4], 1)).toBe(0);
  });

  it('returns null when no class is enabled', () => {
    expect(nextGalleryClass([], null)).toBeNull();
    expect(nextGalleryClass([], 3)).toBeNull();
  });
});

describe('stepGalleryIndex', () => {
  it('steps forward and wraps past the last example', () => {
    expect(stepGalleryIndex(0, 10, 1)).toBe(1);
    expect(stepGalleryIndex(9, 10, 1)).toBe(0);
  });

  it('steps backward and wraps before the first example', () => {
    expect(stepGalleryIndex(0, 10, -1)).toBe(9);
    expect(stepGalleryIndex(2, 10, -1)).toBe(1);
  });

  it('keeps the index when the delta is zero', () => {
    expect(stepGalleryIndex(5, 10, 0)).toBe(5);
  });

  it('returns 0 when the gallery is empty', () => {
    expect(stepGalleryIndex(0, 0, 1)).toBe(0);
    expect(stepGalleryIndex(7, 0, -1)).toBe(0);
  });

  it('wraps an index past the end', () => {
    expect(stepGalleryIndex(12, 10, 1)).toBe(3);
  });

  it('clamps a negative index to the first example', () => {
    expect(stepGalleryIndex(-1, 10, 1)).toBe(0);
  });
});

/** Flattens a Lit template into text, including nested templates and values. */
function templateText(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'number') {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value.map(templateText).join('');
  }
  if (typeof value === 'object' && value !== null && 'strings' in value && 'values' in value) {
    const { strings, values } = value as {
      strings: readonly string[];
      values: readonly unknown[];
    };
    let text = '';
    for (let index = 0; index < strings.length; index += 1) {
      text += strings[index];
      if (index < values.length) {
        text += templateText(values[index]);
      }
    }
    return text;
  }
  return '';
}

/** Renders the component with the minimal document its canvases need. */
function renderMnistCnnText(locale: 'fr' | 'en'): string {
  const globalWithDocument = globalThis as { document?: unknown };
  const originalDocument = globalWithDocument.document;
  globalWithDocument.document = {
    createElement: () => ({ width: 0, height: 0, getContext: () => null }),
  };
  try {
    const element = new MnistCnn();
    const frame = element as unknown as {
      performUpdate(): void;
      render(): unknown;
    };
    frame.performUpdate = () => {};
    element.locale = locale;
    return templateText(frame.render());
  } finally {
    globalWithDocument.document = originalDocument;
  }
}

describe('English rendering', () => {
  it('renders the component labels in English', () => {
    const text = renderMnistCnnText('en');
    expect(text).toContain('Drawing 28×28');
    expect(text).toContain('Training loss');
    expect(text).toContain('Accuracy (training and validation)');
    expect(text).toContain('Filters per layer');
    expect(text).toContain('Convolution filters');
    expect(text).toContain('Probabilities per class');
    expect(text).toContain('Next →');
    expect(text).toContain('Clear');
    expect(text).not.toContain('Dessin');
    expect(text).not.toContain('Entraînement');
    expect(text).not.toContain('Paramètres');
    expect(text).not.toContain('Résultats');
    expect(text).not.toContain('Filtres par couche');
    expect(text).not.toContain('Précédent');
    expect(text).not.toContain('Effacer');
  });

  it('passes the locale to every slider', () => {
    for (const locale of ['en', 'fr'] as const) {
      const text = renderMnistCnnText(locale);
      const sliderCount = text.match(/<drl-slider/g)?.length ?? 0;
      const langCount = text.match(new RegExp(`lang=${locale}`, 'g'))?.length ?? 0;
      expect(sliderCount).toBeGreaterThan(0);
      expect(langCount).toBe(sliderCount);
    }
  });

  it('passes the locale to every slider in the component source', () => {
    const source = readFileSync(new URL('./drl-mnist-cnn.ts', import.meta.url), 'utf8');
    const tags = source.match(/<drl-slider[\s\S]*?>/g) ?? [];
    // Les deux modèles dans `repeat()` ne sont pas rendus dans ce test : la
    // source couvre les huit balises, y compris celles des listes répétées.
    expect(tags.length).toBeGreaterThanOrEqual(8);
    for (const tag of tags) {
      expect(tag).toMatch(/lang=\$\{this\.locale\}/);
    }
  });
});

describe('English labels', () => {
  const filters = [3, 3, 3, 3] as MnistCnnFilters;

  it('localizes the filter, dense and output labels', () => {
    expect(filterPanelSubtitle(null, 'en')).toBe(
      'Mean of the input channels for each output channel; blue = positive, red = negative.',
    );
    expect(filterBiasCaption(0, 0.1254, 'en')).toBe('output 1 · b = 0.125');
    expect(denseLayerTitle(0, 2, 8, 'en')).toBe('Dense 1 — 8 units');
    expect(denseLayerTitle(1, 2, 10, 'en')).toBe('Output — 10 classes');
    expect(denseSizeLabel(0, 'en')).toBe('Units of dense layer 1');
    expect(denseCellLabel(2, 0.4234, 'en')).toBe('unit 3: 0.423');
    expect(outputCellLabel(7, 0.9823, true, undefined, 'en')).toBe('class 7: 98.2%, maximum');
    expect(describeLinkLabel(null, 1, 0, 2, filters, 4, 'en')).toBe('conv2 — input 1, output 3');
  });

  it('localizes the gallery labels', () => {
    expect(galleryClassLabel(getDataset('fashion-mnist'), 0, 'en')).toBe('0 — T-shirt');
    expect(galleryInputSource(getDataset('mnist'), 3, 'test', 'en')).toBe(
      'Gallery validation · Example MNIST (3)',
    );
    expect(galleryInputHint(getDataset('mnist'), 3, 'test', 'en')).toBe(
      'Gallery validation example MNIST: the expected class is "3".',
    );
    expect(galleryInputHint(getDataset('mnist'), null, 'test', 'en')).toContain(
      'Draw an input with the mouse or a finger',
    );
  });

  it('localizes the activation scale', () => {
    expect(activationScaleLabels('sigmoid', 'en')).toEqual(['0', '0.5', '1']);
    expect(activationScaleAriaLabel('sigmoid', 'en')).toBe(
      'Map scale: red at 0, grey at 0.5, blue at 1',
    );
  });
});
