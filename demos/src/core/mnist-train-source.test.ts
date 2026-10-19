import { describe, expect, it } from 'vitest';
import type { MnistIdxDataset } from './mnist-data.js';
import { createIdxTrainSource, type IdxTrainSource } from './mnist-train-source.js';

const PIXELS = 784;

/** Jeu IDX de `count` images : l'image `i` est remplie de l'octet `(i * 37) % 256`. */
function idxDataset(count: number): MnistIdxDataset {
  const images = new Uint8Array(count * PIXELS);
  for (let index = 0; index < count; index += 1) {
    images.fill((index * 37) % 256, index * PIXELS, (index + 1) * PIXELS);
  }
  const labels = new Uint8Array(count);
  for (let index = 0; index < count; index += 1) {
    labels[index] = index % 10;
  }
  return { count, images, labels };
}

function snapshot(dataset: MnistIdxDataset): { images: number[]; labels: number[] } {
  return { images: Array.from(dataset.images), labels: Array.from(dataset.labels) };
}

describe('createIdxTrainSource', () => {
  it('exposes a count equal to the number of selected indices', () => {
    const source = createIdxTrainSource(idxDataset(10), [3, 1, 4, 1, 5]);
    expect(source.count).toBe(5);
  });

  it('materializes exactly the requested batch through the selected indices', () => {
    const dataset = idxDataset(6);
    const source = createIdxTrainSource(dataset, [3, 1, 4]);
    const batch = source.readBatch(0, 2);
    expect(batch.count).toBe(2);
    // Position 0 → indice 3, position 1 → indice 1.
    expect(batch.labels[0]).toBe(3 % 10);
    expect(batch.labels[1]).toBe(1 % 10);
    const expectedByte = (3 * 37) % 256;
    expect(batch.images[0]).toBeCloseTo(expectedByte / 255, 6);
    expect(batch.images[PIXELS - 1]).toBeCloseTo(expectedByte / 255, 6);
    expect(batch.images[PIXELS]).toBeCloseTo(((1 * 37) % 256) / 255, 6);
  });

  it('rejects invalid batch ranges', () => {
    const source = createIdxTrainSource(idxDataset(10), [0, 1, 2]);
    expect(() => source.readBatch(-1, 1)).toThrow(RangeError);
    expect(() => source.readBatch(2, 1)).toThrow(RangeError);
    expect(() => source.readBatch(0, 4)).toThrow(RangeError);
    expect(() => source.readBatch(1.5, 2)).toThrow(RangeError);
    expect(() => source.readBatch(0, 1.5)).toThrow(RangeError);
  });

  it('reshuffles deterministically for a given seed', () => {
    const dataset = idxDataset(40);
    const indices = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    const first: IdxTrainSource = createIdxTrainSource(dataset, indices);
    const second: IdxTrainSource = createIdxTrainSource(dataset, indices);
    first.reshuffle(123);
    second.reshuffle(123);
    const readLabels = (source: {
      readBatch(start: number, end: number): { labels: Uint8Array };
    }) => Array.from(source.readBatch(0, indices.length).labels);
    expect(readLabels(first)).toEqual(readLabels(second));
    // L'ordre a bien changé (le mélange n'est pas l'identité sur 20 indices).
    expect(readLabels(first)).not.toEqual(indices.map((index) => index % 10));
  });

  it('produces a different order for a different seed', () => {
    const dataset = idxDataset(40);
    const indices = Array.from({ length: 30 }, (_unused, index) => index);
    const first = createIdxTrainSource(dataset, indices);
    const second = createIdxTrainSource(dataset, indices);
    first.reshuffle(1);
    second.reshuffle(2);
    const readLabels = (source: {
      readBatch(start: number, end: number): { labels: Uint8Array };
    }) => Array.from(source.readBatch(0, indices.length).labels);
    expect(readLabels(first)).not.toEqual(readLabels(second));
  });

  it('leaves the dataset bytes unchanged after reads and reshuffles', () => {
    const dataset = idxDataset(12);
    const before = snapshot(dataset);
    const source = createIdxTrainSource(dataset, [2, 5, 8, 0]);
    source.readBatch(0, 4);
    source.reshuffle(42);
    source.readBatch(1, 3);
    expect(snapshot(dataset)).toEqual(before);
  });

  it('rejects an empty index list and out-of-range indices', () => {
    expect(() => createIdxTrainSource(idxDataset(10), [])).toThrow(RangeError);
    expect(() => createIdxTrainSource(idxDataset(10), [0, 10])).toThrow(RangeError);
    expect(() => createIdxTrainSource(idxDataset(10), [-1])).toThrow(RangeError);
    expect(() => createIdxTrainSource(idxDataset(10), [1.5])).toThrow(RangeError);
  });

  it('rejects an invalid reshuffle seed', () => {
    const source = createIdxTrainSource(idxDataset(10), [0, 1]);
    expect(() => source.reshuffle(-1)).toThrow(RangeError);
    expect(() => source.reshuffle(1.5)).toThrow(RangeError);
  });

  it('copies the indices so later external mutation does not affect the source', () => {
    const indices = new Uint32Array([4, 2, 7]);
    const source = createIdxTrainSource(idxDataset(10), indices);
    indices[0] = 0;
    const batch = source.readBatch(0, 1);
    expect(batch.labels[0]).toBe(4 % 10);
  });
});
