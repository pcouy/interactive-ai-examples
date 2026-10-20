import { describe, expect, it } from 'vitest';
import {
  DEFAULT_FILTER_COUNTS,
  type MnistCnnConfig,
  type MnistCnnLayerWeights,
  type MnistCnnWeights,
} from '../../core/mnist-cnn.js';
import {
  applyWeightEditorEvent,
  convLayerGeometry,
  describeWeightSelection,
  featureMapSizes,
  filterViewMode,
  selectedWeightValue,
  type WeightSelection,
} from './weight-selection.js';

/** Couche de convolution minimale : noyau plus biais de `outputChannels` zéros. */
function convLayer(kernel: Float32Array, outputChannels: number): MnistCnnLayerWeights {
  return { kernel, bias: new Float32Array(outputChannels) };
}

/** Architecture de référence, quatre convolutions de trois filtres. */
const CONFIG: MnistCnnConfig = { filters: DEFAULT_FILTER_COUNTS };

/** Poids minimaux dont une seule couche porte un noyau rempli, pour `selectedWeightValue`. */
function weightsWithLayerKernel(
  kernel: Float32Array,
  outputChannels: number,
  layerIndex: number = 1,
): MnistCnnWeights {
  const conv = [0, 1, 2, 3].map((index) =>
    index === layerIndex ? convLayer(kernel, outputChannels) : convLayer(new Float32Array(9), 1),
  );
  return {
    conv,
    dense: [convLayer(new Float32Array(9), 1), convLayer(new Float32Array(9), 1)],
  };
}

/** Noyau marqué à la position lue, 2 canaux d'entrée et 3 canaux de sortie. */
function markedConv2Kernel(position: number, value: number): Float32Array {
  const kernel = new Float32Array(54);
  kernel[position] = value;
  return kernel;
}

describe('selectedWeightValue', () => {
  it('reads the value of the selected weight', () => {
    // Position (ky=2, kx=0, canal d'entrée 1, canal de sortie 2) en HWIO : 41.
    const weights = weightsWithLayerKernel(markedConv2Kernel(41, 0.5), 3);
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
        CONFIG,
      ),
    ).toBe(0.5);
  });

  it('reads the value of the last active layer', () => {
    // Trois couches actives : la frontière 2 est la dernière. Position
    // (ky=1, kx=1, canal d'entrée 0, canal de sortie 0) en HWIO : 24.
    const weights = weightsWithLayerKernel(markedConv2Kernel(24, 0.25), 3, 2);
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 2,
          inputChannel: 0,
          outputChannel: 0,
          ky: 1,
          kx: 1,
        },
        weights,
        { filters: DEFAULT_FILTER_COUNTS, convLayers: 3 },
      ),
    ).toBe(0.25);
  });

  it('returns null for a boundary past the active layer count', () => {
    const weights = weightsWithLayerKernel(new Float32Array(54), 3);
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 3,
          inputChannel: 0,
          outputChannel: 0,
          ky: 1,
          kx: 1,
        },
        weights,
        { filters: DEFAULT_FILTER_COUNTS, convLayers: 3 },
      ),
    ).toBeNull();
  });

  it('returns null for a non-weight selection', () => {
    const weights = weightsWithLayerKernel(new Float32Array(54), 3);
    expect(selectedWeightValue(null, weights, CONFIG)).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'link',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 0,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
  });

  it('returns null for out-of-range indices', () => {
    const weights = weightsWithLayerKernel(new Float32Array(54), 3);
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
        CONFIG,
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
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 0,
          ky: 3,
          kx: 0,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 0,
          ky: 0,
          kx: -1,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 5,
          ky: 0,
          kx: 0,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: -1,
          outputChannel: 0,
          ky: 0,
          kx: 0,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
  });

  it('returns null instead of throwing for a fractional kernel index', () => {
    const weights = weightsWithLayerKernel(new Float32Array(54), 3);
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 2,
          ky: 1.5,
          kx: 1,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 2,
          ky: 1,
          kx: 0.25,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
  });

  it('returns null instead of throwing for a fractional boundary', () => {
    const weights = weightsWithLayerKernel(new Float32Array(54), 3);
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1.5,
          inputChannel: 0,
          outputChannel: 2,
          ky: 1,
          kx: 1,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
  });

  it('returns null instead of throwing for a fractional channel index', () => {
    const weights = weightsWithLayerKernel(new Float32Array(54), 3);
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0.5,
          outputChannel: 2,
          ky: 1,
          kx: 1,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 2.5,
          ky: 1,
          kx: 1,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
  });

  it('returns null instead of throwing for a NaN index', () => {
    const weights = weightsWithLayerKernel(new Float32Array(54), 3);
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: Number.NaN,
          inputChannel: 0,
          outputChannel: 2,
          ky: 1,
          kx: 1,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 2,
          ky: Number.NaN,
          kx: 1,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 2,
          ky: 1,
          kx: Number.NaN,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: Number.NaN,
          outputChannel: 2,
          ky: 1,
          kx: 1,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
    expect(
      selectedWeightValue(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: Number.NaN,
          ky: 1,
          kx: 1,
        },
        weights,
        CONFIG,
      ),
    ).toBeNull();
  });
});

describe('featureMapSizes', () => {
  it('returns the feature map size of each active convolution', () => {
    expect(featureMapSizes(4)).toEqual([26, 13, 11, 5]);
    expect(featureMapSizes(3)).toEqual([26, 13, 6]);
    expect(featureMapSizes(2)).toEqual([14, 6]);
    expect(featureMapSizes(1)).toEqual([14]);
    expect(featureMapSizes(0)).toEqual([]);
  });

  it('defaults to the four-layer reference architecture', () => {
    expect(featureMapSizes()).toEqual([26, 13, 11, 5]);
  });
});

describe('convLayerGeometry', () => {
  it('reads the stride, padding and input size of every default layer', () => {
    expect(convLayerGeometry(0)).toEqual({ stride: 1, padding: 0, inputSize: 28 });
    expect(convLayerGeometry(1)).toEqual({ stride: 2, padding: 1, inputSize: 26 });
    expect(convLayerGeometry(2)).toEqual({ stride: 1, padding: 0, inputSize: 13 });
    expect(convLayerGeometry(3)).toEqual({ stride: 2, padding: 0, inputSize: 11 });
  });

  it('reads the geometry of the active layers only', () => {
    expect(convLayerGeometry(0, 3)).toEqual({ stride: 1, padding: 0, inputSize: 28 });
    expect(convLayerGeometry(1, 3)).toEqual({ stride: 2, padding: 1, inputSize: 26 });
    expect(convLayerGeometry(2, 3)).toEqual({ stride: 2, padding: 0, inputSize: 13 });
    expect(convLayerGeometry(0, 1)).toEqual({ stride: 2, padding: 1, inputSize: 28 });
  });

  it('rejects a layer outside the active layer count', () => {
    expect(() => convLayerGeometry(-1)).toThrow(RangeError);
    expect(() => convLayerGeometry(4)).toThrow(RangeError);
    expect(() => convLayerGeometry(1.5)).toThrow(RangeError);
    // Trois couches actives : la couche 3 n'existe pas.
    expect(() => convLayerGeometry(3, 3)).toThrow(RangeError);
    // Aucune couche active : aucune géométrie.
    expect(() => convLayerGeometry(0, 0)).toThrow(RangeError);
  });
});

describe('describeWeightSelection', () => {
  it('labels a link and a weight selection in the active architecture', () => {
    expect(
      describeWeightSelection(
        { kind: 'link', boundary: 1, inputChannel: 0, outputChannel: 2 },
        CONFIG,
      ),
    ).toBe('conv2 — entrée 1, sortie 3');
    expect(
      describeWeightSelection(
        {
          kind: 'weight',
          boundary: 1,
          inputChannel: 0,
          outputChannel: 2,
          ky: 1,
          kx: 1,
        },
        CONFIG,
      ),
    ).toBe('conv2 — entrée 1, sortie 3, position (2, 2)');
  });

  it('returns null without a selection or outside the active layer count', () => {
    expect(describeWeightSelection(null, CONFIG)).toBeNull();
    expect(
      describeWeightSelection(
        { kind: 'weight', boundary: 3, inputChannel: 0, outputChannel: 2, ky: 1, kx: 1 },
        { filters: DEFAULT_FILTER_COUNTS, convLayers: 3 },
      ),
    ).toBeNull();
    expect(
      describeWeightSelection(
        { kind: 'link', boundary: 0, inputChannel: 0, outputChannel: 5 },
        CONFIG,
      ),
    ).toBeNull();
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
    expect(
      applyWeightEditorEvent(selection, {
        kind: 'filterCell',
        layer: 1,
        inputChannel: 0,
        outputChannel: 2,
        ky: 1,
        kx: 0,
      }),
    ).toEqual({
      kind: 'weight',
      boundary: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 0,
    });
  });

  it('ignores a filter cell click without a link selection', () => {
    const selection = applyWeightEditorEvent(null, {
      kind: 'filterCell',
      layer: 1,
      inputChannel: 0,
      outputChannel: 2,
      ky: 1,
      kx: 0,
    });
    expect(selection).toBeNull();
  });

  it('ignores a filter cell click under a link of another boundary', () => {
    const selection: WeightSelection = {
      kind: 'link',
      boundary: 0,
      inputChannel: 0,
      outputChannel: 2,
    };
    expect(
      applyWeightEditorEvent(selection, {
        kind: 'filterCell',
        layer: 2,
        inputChannel: 0,
        outputChannel: 2,
        ky: 1,
        kx: 0,
      }),
    ).toBe(selection);
  });

  it('replaces a weight selection with a link selection', () => {
    const weight = applyWeightEditorEvent(
      { kind: 'link', boundary: 1, inputChannel: 0, outputChannel: 2 },
      { kind: 'filterCell', layer: 1, inputChannel: 0, outputChannel: 2, ky: 1, kx: 0 },
    );
    expect(
      applyWeightEditorEvent(weight, {
        kind: 'link',
        boundary: 2,
        inputChannel: 2,
        outputChannel: 0,
      }),
    ).toEqual({ kind: 'link', boundary: 2, inputChannel: 2, outputChannel: 0 });
  });

  it('ignores a filter cell click under a weight selection', () => {
    const weight = applyWeightEditorEvent(
      { kind: 'link', boundary: 1, inputChannel: 0, outputChannel: 2 },
      { kind: 'filterCell', layer: 1, inputChannel: 0, outputChannel: 2, ky: 1, kx: 0 },
    );
    expect(
      applyWeightEditorEvent(weight, {
        kind: 'filterCell',
        layer: 1,
        inputChannel: 0,
        outputChannel: 2,
        ky: 2,
        kx: 2,
      }),
    ).toBe(weight);
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
    const selection: WeightSelection = {
      kind: 'weight',
      boundary: 0,
      inputChannel: 0,
      outputChannel: 1,
      ky: 2,
      kx: 2,
    };
    expect(filterViewMode(selection, 0, 1)).toEqual({ mode: 'kernel', inputChannel: 0 });
    expect(filterViewMode(selection, 1, 1)).toEqual({ mode: 'mean' });
  });

  it('shows the kernel of the downstream layer of the boundary, not the next one', () => {
    // La frontière 2 relie `conv2` à `conv3` : seule la toile de `conv3` montre le
    // noyau, les toiles de `conv2` et `conv4` restent à la moyenne.
    const selection = { kind: 'link', boundary: 2, inputChannel: 1, outputChannel: 2 } as const;
    expect(filterViewMode(selection, 1, 2)).toEqual({ mode: 'mean' });
    expect(filterViewMode(selection, 2, 2)).toEqual({ mode: 'kernel', inputChannel: 1 });
    expect(filterViewMode(selection, 3, 2)).toEqual({ mode: 'mean' });
  });
});
