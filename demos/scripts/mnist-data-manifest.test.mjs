import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { MNIST_BASE_URL, MNIST_FILES } from './mnist-data-manifest.mjs';

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const README = await readFile(
  resolve(SCRIPT_DIRECTORY, '..', 'public', 'mnist', 'README.md'),
  'utf8',
);
const README_COMPACT = README.replace(/\s+/g, '');

describe('mnist-data-manifest', () => {
  it('lists exactly the four expected archives', () => {
    expect(MNIST_FILES.map((file) => file.name)).toEqual([
      'train-images-idx3-ubyte.gz',
      'train-labels-idx1-ubyte.gz',
      't10k-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz',
    ]);
  });

  it('uses the S3 mirror documented in the README', () => {
    expect(MNIST_BASE_URL).toBe('https://ossci-datasets.s3.amazonaws.com/mnist/');
    expect(README).toContain(MNIST_BASE_URL);
  });

  it('matches the sizes and MD5 values documented in the README', () => {
    for (const file of MNIST_FILES) {
      expect(README).toContain(file.name);
      expect(README_COMPACT).toContain(String(file.size));
      expect(README).toContain(file.md5);
    }
  });
});
