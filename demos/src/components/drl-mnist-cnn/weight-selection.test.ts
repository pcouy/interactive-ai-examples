import { describe, expect, it } from 'vitest';
import type { MnistCnnLayerWeights, MnistCnnWeights } from '../../core/mnist-cnn.js';
import {
  applyWeightEditorEvent,
  filterViewMode,
  selectedWeightValue,
  type WeightSelection,
} from './weight-selection.js';

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

/** Noyau de `conv2` (2 canaux d'entrée, 3 canaux de sortie) marqué à la position lue. */
function markedConv2Kernel(position: number, value: number): Float32Array {
  const kernel = new Float32Array(54);
  kernel[position] = value;
  return kernel;
}

describe('selectedWeightValue', () => {
  it('reads the value of the selected weight', () => {
    // Position (ky=2, kx=0, canal d'entrée 1, canal de sortie 2) en HWIO : 41.
    const weights = weightsWithConv2(markedConv2Kernel(41, 0.5), 3);
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
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0,
        outputChannel: 0,
        ky: 3,
        kx: 0,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0,
        outputChannel: 0,
        ky: 0,
        kx: -1,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0,
        outputChannel: 5,
        ky: 0,
        kx: 0,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: -1,
        outputChannel: 0,
        ky: 0,
        kx: 0,
      }),
    ).toBeNull();
  });

  it('returns null instead of throwing for a fractional kernel index', () => {
    const weights = weightsWithConv2(new Float32Array(54), 3);
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0,
        outputChannel: 2,
        ky: 1.5,
        kx: 1,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0,
        outputChannel: 2,
        ky: 1,
        kx: 0.25,
      }),
    ).toBeNull();
  });

  it('returns null instead of throwing for a fractional boundary', () => {
    const weights = weightsWithConv2(new Float32Array(54), 3);
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1.5,
        inputChannel: 0,
        outputChannel: 2,
        ky: 1,
        kx: 1,
      }),
    ).toBeNull();
  });

  it('returns null instead of throwing for a fractional channel index', () => {
    const weights = weightsWithConv2(new Float32Array(54), 3);
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0.5,
        outputChannel: 2,
        ky: 1,
        kx: 1,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0,
        outputChannel: 2.5,
        ky: 1,
        kx: 1,
      }),
    ).toBeNull();
  });

  it('returns null instead of throwing for a NaN index', () => {
    const weights = weightsWithConv2(new Float32Array(54), 3);
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: Number.NaN,
        inputChannel: 0,
        outputChannel: 2,
        ky: 1,
        kx: 1,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0,
        outputChannel: 2,
        ky: Number.NaN,
        kx: 1,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0,
        outputChannel: 2,
        ky: 1,
        kx: Number.NaN,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: Number.NaN,
        outputChannel: 2,
        ky: 1,
        kx: 1,
      }),
    ).toBeNull();
    expect(
      selectedWeightValue(weights, {
        kind: 'weight',
        boundary: 1,
        inputChannel: 0,
        outputChannel: Number.NaN,
        ky: 1,
        kx: 1,
      }),
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
        layer: 2,
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
      layer: 2,
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
      { kind: 'filterCell', layer: 2, inputChannel: 0, outputChannel: 2, ky: 1, kx: 0 },
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
      { kind: 'filterCell', layer: 2, inputChannel: 0, outputChannel: 2, ky: 1, kx: 0 },
    );
    expect(
      applyWeightEditorEvent(weight, {
        kind: 'filterCell',
        layer: 2,
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
    const selection: WeightSelection = {
      kind: 'weight',
      boundary: 0,
      inputChannel: 0,
      outputChannel: 1,
      ky: 2,
      kx: 2,
    };
    expect(filterViewMode(selection, 1, 1)).toEqual({ mode: 'kernel', inputChannel: 0 });
    expect(filterViewMode(selection, 1, 0)).toEqual({ mode: 'mean' });
  });
});
