import { describe, expect, it } from 'vitest';
import {
  loadIdxDataset,
  loadMnistDataset,
  MNIST_PIXELS_PER_IMAGE,
  type MnistDataset,
  type MnistIdxDataset,
  materializeBatch,
  materializeSample,
  parseIdxImages,
  parseIdxLabels,
  sampleBalanced,
  selectStratifiedIndices,
} from './mnist-data.js';

const IMAGE_MAGIC = 2051;
const LABEL_MAGIC = 2049;

function u32Bytes(value: number): number[] {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

function buildImageFile(
  count: number,
  rows = 28,
  cols = 28,
  pixel: (image: number, offset: number) => number = (image, offset) => (image * 7 + offset) % 256,
  extraBytes = 0,
): Uint8Array {
  const bytes = new Uint8Array(16 + count * rows * cols + extraBytes);
  bytes.set([...u32Bytes(IMAGE_MAGIC), ...u32Bytes(count), ...u32Bytes(rows), ...u32Bytes(cols)]);
  for (let image = 0; image < count; image += 1) {
    for (let offset = 0; offset < rows * cols; offset += 1) {
      bytes[16 + image * rows * cols + offset] = pixel(image, offset) & 0xff;
    }
  }
  return bytes;
}

function buildLabelFile(labels: number[], extraBytes = 0): Uint8Array {
  const bytes = new Uint8Array(8 + labels.length + extraBytes);
  bytes.set([...u32Bytes(LABEL_MAGIC), ...u32Bytes(labels.length)]);
  for (let index = 0; index < labels.length; index += 1) {
    bytes[8 + index] = labels[index];
  }
  return bytes;
}

/** Jeu synthétique dont le premier pixel de l'image i vaut `labels[i] * 25`. */
function buildDataset(labels: number[]): MnistDataset {
  const imageFile = buildImageFile(labels.length, 28, 28, (image, offset) =>
    offset === 0 ? labels[image] * 25 : (image * 7 + offset) % 256,
  );
  return loadMnistDataset(imageFile, buildLabelFile(labels));
}

/** Jeu IDX brut équivalent à `buildDataset`, plus les buffers d'origine. */
function buildRawIdxDataset(labels: number[]): {
  dataset: MnistIdxDataset;
  imageFile: Uint8Array;
  labelFile: Uint8Array;
} {
  const imageFile = buildImageFile(labels.length, 28, 28, (image, offset) =>
    offset === 0 ? labels[image] * 25 : (image * 7 + offset) % 256,
  );
  const labelFile = buildLabelFile(labels);
  return { dataset: loadIdxDataset(imageFile, labelFile), imageFile, labelFile };
}

function balancedLabels(repeats: number): number[] {
  const labels: number[] = [];
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    for (let label = 0; label <= 9; label += 1) {
      labels.push(label);
    }
  }
  return labels;
}

describe('parseIdxImages', () => {
  it('decodes a valid image file into normalized float32 pixels', () => {
    const file = buildImageFile(2);
    const parsed = parseIdxImages(file);
    expect(parsed.count).toBe(2);
    expect(parsed.rows).toBe(28);
    expect(parsed.cols).toBe(28);
    expect(parsed.images).toBeInstanceOf(Float32Array);
    expect(parsed.images).toHaveLength(2 * MNIST_PIXELS_PER_IMAGE);
    const expected = new Float32Array(2 * MNIST_PIXELS_PER_IMAGE);
    for (let image = 0; image < 2; image += 1) {
      for (let offset = 0; offset < MNIST_PIXELS_PER_IMAGE; offset += 1) {
        expected[image * MNIST_PIXELS_PER_IMAGE + offset] =
          file[16 + image * MNIST_PIXELS_PER_IMAGE + offset] / 255;
      }
    }
    expect(parsed.images).toEqual(expected);
  });

  it('normalizes the extreme byte values to exactly 0 and 1', () => {
    const parsed = parseIdxImages(buildImageFile(3, 28, 28, (image) => [0, 255, 128][image]));
    expect(parsed.count).toBe(3);
    for (let offset = 0; offset < MNIST_PIXELS_PER_IMAGE; offset += 1) {
      expect(parsed.images[offset]).toBe(0);
      expect(parsed.images[MNIST_PIXELS_PER_IMAGE + offset]).toBe(1);
      expect(parsed.images[2 * MNIST_PIXELS_PER_IMAGE + offset]).toBe(Math.fround(128 / 255));
    }
  });

  it('keeps every normalized pixel inside [0, 1]', () => {
    const parsed = parseIdxImages(buildImageFile(4));
    for (const value of parsed.images) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('accepts an ArrayBuffer and a shifted Uint8Array view', () => {
    const file = buildImageFile(1);
    const fromBuffer = parseIdxImages(file.buffer as ArrayBuffer);
    expect(fromBuffer.count).toBe(1);
    const padded = new Uint8Array(5 + file.length);
    padded.set(file, 5);
    const fromView = parseIdxImages(padded.subarray(5));
    expect(fromView.count).toBe(1);
    expect(fromView.images).toEqual(parseIdxImages(file).images);
  });

  it('parses an empty image file with count 0', () => {
    const parsed = parseIdxImages(buildImageFile(0));
    expect(parsed.count).toBe(0);
    expect(parsed.images).toHaveLength(0);
  });

  it('rejects a wrong image magic', () => {
    const wrong = buildLabelFile([3, 4, 5, 6, 7, 8, 9, 0, 1]);
    expect(() => parseIdxImages(wrong)).toThrow(RangeError);
    expect(() => parseIdxImages(wrong)).toThrow(/magic 2049/);
  });

  it('rejects a truncated image header and payload', () => {
    expect(() => parseIdxImages(new Uint8Array(15))).toThrow(/truncated image header/);
    const full = buildImageFile(2);
    expect(() => parseIdxImages(full.subarray(0, full.length - 1))).toThrow(
      /truncated image payload/,
    );
    expect(() => parseIdxImages(full.subarray(0, 16))).toThrow(/truncated image payload/);
  });

  it('rejects extra bytes after the image payload', () => {
    expect(() => parseIdxImages(buildImageFile(1, 28, 28, undefined, 1))).toThrow(
      /extra bytes after image payload/,
    );
    expect(() => parseIdxImages(buildImageFile(1, 28, 28, undefined, 100))).toThrow(
      /extra bytes after image payload/,
    );
  });

  it('rejects image dimensions other than 28x28', () => {
    expect(() => parseIdxImages(buildImageFile(1, 27, 28))).toThrow(/27x28/);
    expect(() => parseIdxImages(buildImageFile(1, 28, 30))).toThrow(/28x30/);
  });
});

describe('parseIdxLabels', () => {
  it('decodes a valid label file', () => {
    const labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const parsed = parseIdxLabels(buildLabelFile(labels));
    expect(parsed).toBeInstanceOf(Uint8Array);
    expect(parsed).toEqual(new Uint8Array(labels));
  });

  it('rejects a wrong label magic', () => {
    const wrong = buildImageFile(1);
    expect(() => parseIdxLabels(wrong)).toThrow(RangeError);
    expect(() => parseIdxLabels(wrong)).toThrow(/magic 2051/);
  });

  it('rejects a truncated label header and payload', () => {
    expect(() => parseIdxLabels(new Uint8Array(7))).toThrow(/truncated label header/);
    const full = buildLabelFile([1, 2, 3]);
    expect(() => parseIdxLabels(full.subarray(0, full.length - 1))).toThrow(
      /truncated label payload/,
    );
    expect(() => parseIdxLabels(full.subarray(0, 8))).toThrow(/truncated label payload/);
  });

  it('rejects extra bytes after the label payload', () => {
    expect(() => parseIdxLabels(buildLabelFile([1, 2], 1))).toThrow(
      /extra bytes after label payload/,
    );
  });

  it('rejects labels outside 0 to 9 with the offending index', () => {
    expect(() => parseIdxLabels(buildLabelFile([0, 1, 10]))).toThrow(/invalid label 10 at index 2/);
    expect(() => parseIdxLabels(buildLabelFile([255]))).toThrow(/invalid label 255 at index 0/);
  });
});

describe('loadMnistDataset', () => {
  it('loads a consistent image and label pair', () => {
    const labels = [9, 0, 4];
    const dataset = loadMnistDataset(buildImageFile(3), buildLabelFile(labels));
    expect(dataset.count).toBe(3);
    expect(dataset.labels).toEqual(new Uint8Array(labels));
    expect(dataset.images).toHaveLength(3 * MNIST_PIXELS_PER_IMAGE);
  });

  it('rejects mismatched image and label counts', () => {
    const images = buildImageFile(2);
    const labels = buildLabelFile([1]);
    expect(() => loadMnistDataset(images, labels)).toThrow(/2 images vs 1 labels/);
  });

  it('loads an empty dataset with zero counts', () => {
    const dataset = loadMnistDataset(buildImageFile(0), buildLabelFile([]));
    expect(dataset.count).toBe(0);
    expect(dataset.labels).toHaveLength(0);
    expect(dataset.images).toHaveLength(0);
  });
});

describe('sampleBalanced', () => {
  const dataset = buildDataset(balancedLabels(3));

  it('picks the same number of examples from every class', () => {
    const sample = sampleBalanced(dataset, { perClass: 2, seed: 7 });
    expect(sample.count).toBe(20);
    expect(sample.images).toHaveLength(20 * MNIST_PIXELS_PER_IMAGE);
    const counts = new Array<number>(10).fill(0);
    for (const label of sample.labels) {
      counts[label] += 1;
    }
    expect(counts).toEqual(new Array<number>(10).fill(2));
  });

  it('keeps each image aligned with its label', () => {
    const sample = sampleBalanced(dataset, { perClass: 3, seed: 11 });
    for (let position = 0; position < sample.count; position += 1) {
      const firstPixel = sample.images[position * MNIST_PIXELS_PER_IMAGE];
      const encodedLabel = Math.round((firstPixel * 255) / 25);
      expect(encodedLabel).toBe(sample.labels[position]);
    }
  });

  it('is deterministic for a given seed and varies across seeds', () => {
    const first = sampleBalanced(dataset, { perClass: 2, seed: 7 });
    const second = sampleBalanced(dataset, { perClass: 2, seed: 7 });
    expect(first.labels).toEqual(second.labels);
    expect(first.images).toEqual(second.images);
    const other = sampleBalanced(dataset, { perClass: 2, seed: 8 });
    expect(other.labels).not.toEqual(first.labels);
    expect(other.images).not.toEqual(first.images);
  });

  it('interleaves classes instead of returning class blocks', () => {
    const sample = sampleBalanced(dataset, { perClass: 2, seed: 7 });
    const sorted = [...sample.labels].sort((a, b) => a - b);
    expect(Array.from(sample.labels)).not.toEqual(sorted);
  });

  it('allows taking all available examples of each class', () => {
    const sample = sampleBalanced(dataset, { perClass: 3, seed: 1 });
    expect(sample.count).toBe(30);
  });

  it('rejects perClass values that no class can satisfy', () => {
    expect(() => sampleBalanced(dataset, { perClass: 4, seed: 7 })).toThrow(
      /class 0 has only 3 examples, need 4/,
    );
    const sparse = buildDataset([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
    expect(() => sampleBalanced(sparse, { perClass: 1, seed: 7 })).toThrow(
      /class 5 has only 0 examples, need 1/,
    );
  });

  it('rejects invalid seeds', () => {
    for (const seed of [
      -1,
      1.5,
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.MAX_SAFE_INTEGER + 1,
    ]) {
      expect(() => sampleBalanced(dataset, { perClass: 2, seed })).toThrow(/invalid seed/);
    }
  });

  it('rejects invalid perClass sizes', () => {
    for (const perClass of [0, -2, 2.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => sampleBalanced(dataset, { perClass, seed: 7 })).toThrow(/invalid perClass/);
    }
  });

  it('rejects labels outside 0 to 9 in the source dataset', () => {
    const corrupted: MnistDataset = {
      count: 1,
      images: new Float32Array(MNIST_PIXELS_PER_IMAGE),
      labels: new Uint8Array([10]),
    };
    expect(() => sampleBalanced(corrupted, { perClass: 1, seed: 7 })).toThrow(
      /outside the 0 to 9 range/,
    );
  });

  it('rejects a dataset whose shape disagrees with its labels', () => {
    const wrongCount: MnistDataset = {
      count: 1,
      images: new Float32Array(2 * MNIST_PIXELS_PER_IMAGE),
      labels: new Uint8Array([0, 1]),
    };
    expect(() => sampleBalanced(wrongCount, { perClass: 1, seed: 7 })).toThrow(
      /count 1 does not match label count 2/,
    );

    const missingPixels: MnistDataset = {
      count: 10,
      images: new Float32Array(10 * MNIST_PIXELS_PER_IMAGE - 1),
      labels: new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    };
    expect(() => sampleBalanced(missingPixels, { perClass: 1, seed: 7 })).toThrow(
      /image length .* does not match label count/,
    );
  });
});

describe('mnist-data input immutability', () => {
  it('leaves the IDX input buffers byte-identical', () => {
    const imageFile = buildImageFile(10);
    const labelFile = buildLabelFile(balancedLabels(1));
    const imageCopy = new Uint8Array(imageFile);
    const labelCopy = new Uint8Array(labelFile);
    const dataset = loadMnistDataset(imageFile, labelFile);
    sampleBalanced(dataset, { perClass: 1, seed: 5 });
    expect(new Uint8Array(imageFile)).toEqual(imageCopy);
    expect(new Uint8Array(labelFile)).toEqual(labelCopy);
  });

  it('leaves the loaded dataset untouched by sampling and returns copies', () => {
    const dataset = buildDataset(balancedLabels(2));
    const imagesSnapshot = new Float32Array(dataset.images);
    const labelsSnapshot = new Uint8Array(dataset.labels);
    const sample = sampleBalanced(dataset, { perClass: 2, seed: 3 });
    const sampleImages = new Float32Array(sample.images);
    sample.images[0] = -1;
    expect(new Float32Array(dataset.images)).toEqual(imagesSnapshot);
    expect(new Uint8Array(dataset.labels)).toEqual(labelsSnapshot);
    const again = sampleBalanced(dataset, { perClass: 2, seed: 3 });
    expect(again.images).toEqual(sampleImages);
    expect(again.images[0]).toBeGreaterThanOrEqual(0);
  });
});

describe('loadIdxDataset', () => {
  it('keeps raw IDX bytes without converting pixels to float32', () => {
    const labels = balancedLabels(2);
    const { dataset, imageFile } = buildRawIdxDataset(labels);
    expect(dataset.count).toBe(labels.length);
    expect(dataset.images).toBeInstanceOf(Uint8Array);
    expect(dataset.labels).toBeInstanceOf(Uint8Array);
    expect(dataset.images).toHaveLength(labels.length * MNIST_PIXELS_PER_IMAGE);
    expect(dataset.labels).toEqual(new Uint8Array(labels));
    expect(Array.from(dataset.images.subarray(0, 5))).toEqual(
      Array.from(imageFile.subarray(16, 21)),
    );
    expect(dataset.images[0]).toBe(labels[0] * 25);
  });

  it('loads an empty pair with zero counts', () => {
    const { dataset } = buildRawIdxDataset([]);
    expect(dataset.count).toBe(0);
    expect(dataset.images).toHaveLength(0);
    expect(dataset.labels).toHaveLength(0);
  });

  it('rejects mismatched image and label counts', () => {
    expect(() => loadIdxDataset(buildImageFile(2), buildLabelFile([1]))).toThrow(
      /2 images vs 1 labels/,
    );
  });

  it('rejects a wrong image magic or dimensions', () => {
    expect(() => loadIdxDataset(buildLabelFile(balancedLabels(2)), buildLabelFile([1, 2]))).toThrow(
      /magic 2049/,
    );
    expect(() => loadIdxDataset(buildImageFile(1, 27, 28), buildLabelFile([1]))).toThrow(/27x28/);
  });

  it('rejects a truncated or over-long image payload', () => {
    const full = buildImageFile(2);
    expect(() => loadIdxDataset(full.subarray(0, full.length - 1), buildLabelFile([1, 2]))).toThrow(
      /truncated image payload/,
    );
    expect(() =>
      loadIdxDataset(buildImageFile(1, 28, 28, undefined, 1), buildLabelFile([1])),
    ).toThrow(/extra bytes after image payload/);
  });

  it('rejects a truncated label payload and out-of-range labels', () => {
    const full = buildLabelFile([1, 2, 3]);
    expect(() => loadIdxDataset(buildImageFile(3), full.subarray(0, full.length - 1))).toThrow(
      /truncated label payload/,
    );
    expect(() => loadIdxDataset(buildImageFile(3), buildLabelFile([0, 1, 10]))).toThrow(
      /invalid label 10 at index 2/,
    );
  });

  it('leaves the IDX input buffers byte-identical', () => {
    const { dataset, imageFile, labelFile } = buildRawIdxDataset(balancedLabels(2));
    const imageCopy = new Uint8Array(imageFile);
    const labelCopy = new Uint8Array(labelFile);
    selectStratifiedIndices(dataset, { fraction: 0.5, seed: 5 });
    materializeSample(dataset, Uint32Array.from([0, 1, 2]));
    expect(new Uint8Array(imageFile)).toEqual(imageCopy);
    expect(new Uint8Array(labelFile)).toEqual(labelCopy);
  });
});

describe('selectStratifiedIndices', () => {
  const { dataset } = buildRawIdxDataset(balancedLabels(3)); // 3 examples per class

  function classCounts(indices: Uint32Array, source: MnistIdxDataset): number[] {
    const counts = new Array<number>(10).fill(0);
    for (const index of indices) {
      counts[source.labels[index]] += 1;
    }
    return counts;
  }

  it('returns every example exactly once for fraction 1', () => {
    const indices = selectStratifiedIndices(dataset, { fraction: 1, seed: 7 });
    expect(indices).toBeInstanceOf(Uint32Array);
    expect(indices).toHaveLength(30);
    expect([...indices].sort((a, b) => a - b)).toEqual([...Array(30).keys()]);
  });

  it('keeps the per-class balance for a small fraction', () => {
    const indices = selectStratifiedIndices(dataset, { fraction: 0.1, seed: 7 });
    // round(3 × 0.1) = 0, ramené à 1 par classe : 10 exemples au total.
    expect(indices).toHaveLength(10);
    expect(classCounts(indices, dataset)).toEqual(new Array<number>(10).fill(1));
    const half = selectStratifiedIndices(dataset, { fraction: 2 / 3, seed: 7 });
    expect(half).toHaveLength(20);
    expect(classCounts(half, dataset)).toEqual(new Array<number>(10).fill(2));
  });

  it('keeps every returned index inside the dataset bounds', () => {
    const indices = selectStratifiedIndices(dataset, { fraction: 0.5, seed: 3 });
    for (const index of indices) {
      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThan(dataset.count);
    }
  });

  it('is deterministic for equal seeds and varies across seeds', () => {
    const first = selectStratifiedIndices(dataset, { fraction: 0.5, seed: 7 });
    const second = selectStratifiedIndices(dataset, { fraction: 0.5, seed: 7 });
    expect(first).toEqual(second);
    const other = selectStratifiedIndices(dataset, { fraction: 0.5, seed: 8 });
    expect(other).not.toEqual(first);
  });

  it('interleaves classes instead of returning class blocks', () => {
    const indices = selectStratifiedIndices(dataset, { fraction: 1, seed: 7 });
    const labels = [...indices].map((index) => dataset.labels[index]);
    const sorted = [...labels].sort((a, b) => a - b);
    expect(labels).not.toEqual(sorted);
  });

  it('rejects invalid fractions', () => {
    for (const fraction of [0, -0.5, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => selectStratifiedIndices(dataset, { fraction, seed: 7 })).toThrow(
        /invalid fraction/,
      );
    }
  });

  it('rejects invalid seeds', () => {
    for (const seed of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => selectStratifiedIndices(dataset, { fraction: 0.5, seed })).toThrow(
        /invalid seed/,
      );
    }
  });

  it('rejects a class without any example', () => {
    const sparse = buildRawIdxDataset([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]).dataset;
    expect(() => selectStratifiedIndices(sparse, { fraction: 0.5, seed: 7 })).toThrow(
      /class 5 has no examples/,
    );
  });

  it('rejects a dataset whose shape disagrees with its labels', () => {
    const wrongCount: MnistIdxDataset = {
      count: 1,
      images: new Uint8Array(MNIST_PIXELS_PER_IMAGE),
      labels: new Uint8Array([0, 1]),
    };
    expect(() => selectStratifiedIndices(wrongCount, { fraction: 0.5, seed: 7 })).toThrow(
      /count 1 does not match label count 2/,
    );
    const wrongBytes: MnistIdxDataset = {
      count: 2,
      images: new Uint8Array(MNIST_PIXELS_PER_IMAGE),
      labels: new Uint8Array([0, 1]),
    };
    expect(() => selectStratifiedIndices(wrongBytes, { fraction: 0.5, seed: 7 })).toThrow(
      /image bytes .* do not match label count/,
    );
  });
});

describe('materializeSample and materializeBatch', () => {
  const { dataset } = buildRawIdxDataset(balancedLabels(3));

  it('converts only the selected images and keeps labels aligned', () => {
    const indices = selectStratifiedIndices(dataset, { fraction: 0.5, seed: 11 });
    const sample = materializeSample(dataset, indices);
    expect(sample.count).toBe(indices.length);
    expect(sample.images).toBeInstanceOf(Float32Array);
    expect(sample.images).toHaveLength(indices.length * MNIST_PIXELS_PER_IMAGE);
    expect(sample.labels).toHaveLength(indices.length);
    for (let position = 0; position < sample.count; position += 1) {
      const firstPixel = sample.images[position * MNIST_PIXELS_PER_IMAGE];
      expect(Math.round((firstPixel * 255) / 25)).toBe(sample.labels[position]);
      expect(firstPixel).toBeGreaterThanOrEqual(0);
      expect(firstPixel).toBeLessThanOrEqual(1);
    }
  });

  it('materializes only the requested batch slice', () => {
    const indices = selectStratifiedIndices(dataset, { fraction: 1, seed: 7 });
    const batch = materializeBatch(dataset, indices, 5, 9);
    expect(batch.count).toBe(4);
    expect(batch.images).toHaveLength(4 * MNIST_PIXELS_PER_IMAGE);
    const expected = materializeSample(dataset, indices.slice(5, 9));
    expect(batch.images).toEqual(expected.images);
    expect(batch.labels).toEqual(expected.labels);
  });

  it('converts only a middle range from the raw bytes at 1/255', () => {
    const indices = selectStratifiedIndices(dataset, { fraction: 1, seed: 7 });
    const start = 6;
    const end = 13;
    const bytesBefore = new Uint8Array(dataset.images);
    const labelsBefore = new Uint8Array(dataset.labels);

    const batch = materializeBatch(dataset, indices, start, end);

    // Seule la tranche demandée est convertie.
    expect(batch.count).toBe(end - start);
    expect(batch.images).toHaveLength((end - start) * MNIST_PIXELS_PER_IMAGE);
    expect(batch.labels).toHaveLength(end - start);
    for (let position = 0; position < batch.count; position += 1) {
      const sourceIndex = indices[start + position];
      const source = sourceIndex * MNIST_PIXELS_PER_IMAGE;
      const target = position * MNIST_PIXELS_PER_IMAGE;
      for (let pixel = 0; pixel < MNIST_PIXELS_PER_IMAGE; pixel += 1) {
        expect(batch.images[target + pixel]).toBe(
          Math.fround(dataset.images[source + pixel] / 255),
        );
      }
      expect(batch.labels[position]).toBe(dataset.labels[sourceIndex]);
    }

    // Le jeu source reste inchangé.
    expect(new Uint8Array(dataset.images)).toEqual(bytesBefore);
    expect(new Uint8Array(dataset.labels)).toEqual(labelsBefore);
  });

  it('rejects indices outside the dataset bounds', () => {
    expect(() => materializeSample(dataset, Uint32Array.from([dataset.count]))).toThrow(
      /outside 0 to/,
    );
    expect(() => materializeSample(dataset, [-1])).toThrow(/outside 0 to/);
    expect(() => materializeSample(dataset, [1.5])).toThrow(/outside 0 to/);
  });

  it('rejects invalid batch ranges', () => {
    const indices = Uint32Array.from([0, 1, 2, 3]);
    expect(() => materializeBatch(dataset, indices, 3, 2)).toThrow(/invalid batch range/);
    expect(() => materializeBatch(dataset, indices, -1, 2)).toThrow(/invalid batch range/);
    expect(() => materializeBatch(dataset, indices, 0, 5)).toThrow(/invalid batch range/);
    expect(() => materializeBatch(dataset, indices, 0.5, 2)).toThrow(/invalid batch range/);
  });

  it('does not mutate the raw dataset', () => {
    const imagesSnapshot = new Uint8Array(dataset.images);
    const labelsSnapshot = new Uint8Array(dataset.labels);
    const indices = selectStratifiedIndices(dataset, { fraction: 1, seed: 4 });
    const sample = materializeSample(dataset, indices);
    sample.images[0] = -1;
    expect(new Uint8Array(dataset.images)).toEqual(imagesSnapshot);
    expect(new Uint8Array(dataset.labels)).toEqual(labelsSnapshot);
    const again = materializeSample(dataset, indices);
    expect(again.images[0]).toBeGreaterThanOrEqual(0);
  });
});
