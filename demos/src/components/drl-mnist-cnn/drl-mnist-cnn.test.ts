import { describe, expect, it } from 'vitest';
import type {
  MnistCnnFilters,
  MnistCnnLayerWeights,
  MnistCnnWeights,
} from '../../core/mnist-cnn.js';
import type { WeightEditorEvent } from './drl-mnist-cnn.js';
import {
  applyWeightEditorEvent,
  argmaxIndex,
  brushStrength,
  convLayerGeometry,
  describeLinkLabel,
  describeWeightSelection,
  filterCanvasInputChannel,
  filterKernelMean,
  filterKernelSlice,
  filterKernelValues,
  filterViewMode,
  formatLoss,
  formatPercent,
  formatSeconds,
  hexToRgb,
  linkLineState,
  mixRgb,
  paintDisc,
  parseWeightInput,
  pointerToCell,
  pointerToGrid,
  selectedWeightValue,
  stratifiedExampleCountEstimate,
} from './drl-mnist-cnn.js';

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
    conv1: convLayer(new Float32Array(9), 1),
    conv2: convLayer(kernel, outputChannels),
    conv3: convLayer(new Float32Array(9), 1),
    conv4: convLayer(new Float32Array(9), 1),
    dense1: convLayer(new Float32Array(9), 1),
    dense2: convLayer(new Float32Array(9), 1),
  };
}

/** Géométrie d'une couche : `conv1` stride 1 sans remplissage, `conv2` stride 2 remplissage 1. */
const CONV1_GEOMETRY = { stride: 1, padding: 0, inputSize: 28 } as const;
const CONV2_GEOMETRY = { stride: 2, padding: 1, inputSize: 26 } as const;

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

describe('formatPercent', () => {
  it('formats with one decimal and a French comma', () => {
    expect(formatPercent(12.34)).toBe('12,3 %');
    expect(formatPercent(0)).toBe('0,0 %');
    expect(formatPercent(99.96)).toBe('100,0 %');
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

describe('formatLoss', () => {
  it('formats with three decimals and a French comma', () => {
    expect(formatLoss(0.34567)).toBe('0,346');
    expect(formatLoss(0)).toBe('0,000');
  });

  it('renders a dash for a null or non-finite loss', () => {
    expect(formatLoss(null)).toBe('–');
    expect(formatLoss(Number.NaN)).toBe('–');
    expect(formatLoss(Number.POSITIVE_INFINITY)).toBe('–');
  });
});

describe('formatSeconds', () => {
  it('formats durations with one decimal and a French comma', () => {
    expect(formatSeconds(0)).toBe('0,0 s');
    expect(formatSeconds(1500)).toBe('1,5 s');
    expect(formatSeconds(12345)).toBe('12,3 s');
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
    expect(applyWeightEditorEvent(selection, filterCellEvent(2, 0, 2, 1, 0))).toEqual({
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 0,
    });
  });

  it('ignores a filter cell click without a link selection', () => {
    expect(applyWeightEditorEvent(null, filterCellEvent(2, 0, 2, 1, 0))).toBeNull();
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
      filterCellEvent(2, 0, 2, 1, 0),
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
      filterCellEvent(2, 0, 2, 1, 0),
    );
    // La frontière 1 et le canal de sortie 2 correspondent : seul le genre de la
    // sélection (poids, pas liaison) fait ignorer le clic.
    expect(applyWeightEditorEvent(selection, filterCellEvent(2, 0, 2, 2, 2))).toEqual(selection);
  });
});

describe('filterViewMode', () => {
  it('is mean by default', () => {
    expect(filterViewMode(null, 1, 0)).toEqual({ mode: 'mean' });
    expect(
      filterViewMode(
        { kind: 'weight', boundary: 1, inputChannel: 0, outputChannel: 0, ky: 0, kx: 0 },
        1,
        0,
      ),
    ).toEqual({ mode: 'mean' });
  });

  it('is kernel only for the canvas matching a link selection', () => {
    const selection = { kind: 'link', boundary: 1, inputChannel: 2, outputChannel: 0 } as const;
    expect(filterViewMode(selection, 2, 0)).toEqual({ mode: 'kernel', inputChannel: 2 });
    expect(filterViewMode(selection, 1, 0)).toEqual({ mode: 'mean' });
    expect(filterViewMode(selection, 2, 1)).toEqual({ mode: 'mean' });
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
    expect(filterViewMode(selection, 1, 1)).toEqual({ mode: 'kernel', inputChannel: 0 });
    expect(filterViewMode(selection, 1, 0)).toEqual({ mode: 'mean' });
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
    expect(filterCanvasInputChannel(selection, 2, 2)).toBe(0);
    expect(filterCanvasInputChannel(selection, 2, 1)).toBeNull();
    expect(filterCanvasInputChannel(selection, 1, 2)).toBeNull();
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

describe('describeWeightSelection', () => {
  const filters = [3, 3, 3, 3] as MnistCnnFilters;

  it('describes a weight selection', () => {
    expect(
      describeWeightSelection(
        { kind: 'weight', boundary: 1, inputChannel: 0, outputChannel: 2, ky: 2, kx: 1 },
        filters,
      ),
    ).toBe('conv2 — entrée 1, sortie 3, position (3, 2)');
  });

  it('describes a link selection', () => {
    expect(
      describeWeightSelection(
        { kind: 'link', boundary: 0, inputChannel: 0, outputChannel: 1 },
        filters,
      ),
    ).toBe('conv1 — entrée 1, sortie 2');
  });

  it('returns null for no selection', () => {
    expect(describeWeightSelection(null, filters)).toBeNull();
  });

  it('returns null for a selection outside the architecture', () => {
    expect(
      describeWeightSelection(
        { kind: 'link', boundary: 0, inputChannel: 0, outputChannel: 5 },
        filters,
      ),
    ).toBeNull();
  });
});

describe('describeLinkLabel', () => {
  const filters = [3, 3, 3, 3] as MnistCnnFilters;

  it('returns the plain pair name when nothing scopes the line', () => {
    expect(describeLinkLabel(null, 1, 0, 2, filters)).toBe('conv2 — entrée 1, sortie 3');
    expect(
      describeLinkLabel(
        { kind: 'link', boundary: 0, inputChannel: 0, outputChannel: 1 },
        1,
        0,
        2,
        filters,
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
    expect(describeLinkLabel(selection, 1, 0, 2, filters)).toBe(
      'conv2 — entrée 1, sortie 3 · 1 poids sur 9',
    );
    expect(describeLinkLabel(selection, 1, 0, 1, filters)).toBe('conv2 — entrée 1, sortie 2');
  });
});

describe('selectedWeightValue', () => {
  it('reads the value of the selected weight', () => {
    // conv2 with 2 input channels and 3 output channels: 9 × 2 × 3 = 54 weights.
    const kernel = new Float32Array(54);
    const index = (2 * 3 + 0) * 2 * 3 + 1 * 3 + 2;
    kernel[index] = 0.5;
    const weights = weightsWithConv2(kernel, 3);
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 1,
        outputChannel: 2,
        ky: 2,
        kx: 0,
      }),
    ).toBe(0.5);
  });

  it('returns null for a non-weight selection', () => {
    const weights = weightsWithConv2(new Float32Array(54), 3);
    expect(selectedWeightValue(weights, null)).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'link',
        boundary: 1,
        inputChannel: 0,
        outputChannel: 0,
      }),
    ).toBeNull();
  });

  it('returns null for out-of-range indices', () => {
    const weights = weightsWithConv2(new Float32Array(54), 3);
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 5,
        outputChannel: 0,
        ky: 0,
        kx: 0,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 9,
        inputChannel: 0,
        outputChannel: 0,
        ky: 0,
        kx: 0,
      }),
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
