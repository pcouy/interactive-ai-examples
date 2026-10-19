import { describe, expect, it } from 'vitest';
import {
  loadMnistDataset,
  MNIST_PIXELS_PER_IMAGE,
  type MnistDataset,
  parseIdxImages,
  parseIdxLabels,
  sampleBalanced,
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
