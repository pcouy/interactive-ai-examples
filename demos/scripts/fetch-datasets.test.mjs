import { createHash } from 'node:crypto';
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  DATASET_SOURCE_IDS,
  DATASET_SOURCES,
  MEDMNIST_BASE_URL,
  MEDMNIST_FILES,
  QUICKDRAW_FILES,
} from './dataset-sources.mjs';
import { buildTargetPath, describePlan, main, parseArgs } from './fetch-datasets.mjs';

const MD5_PATTERN = /^[0-9a-f]{32}$/;
const EXPECTED_FILES = [
  'train-images-idx3-ubyte.gz',
  'train-labels-idx1-ubyte.gz',
  't10k-images-idx3-ubyte.gz',
  't10k-labels-idx1-ubyte.gz',
];

describe('parseArgs', () => {
  it('yields no filters, no dry-run and no force by default', () => {
    expect(parseArgs([])).toEqual({ only: [], dryRun: false, force: false, help: false });
  });

  it('collects repeated ids and sets dry-run', () => {
    const options = parseArgs(['--only', 'mnist', '--only', 'fashion-mnist', '--dry-run']);
    expect(options.only).toEqual(['mnist', 'fashion-mnist']);
    expect(options.dryRun).toBe(true);
  });

  it('splits a comma-separated --only value and sets force', () => {
    const options = parseArgs(['--only', 'mnist,fashion-mnist', '--force']);
    expect(options.only).toEqual(['mnist', 'fashion-mnist']);
    expect(options.force).toBe(true);
  });

  it('rejects an unknown id with a clear error', () => {
    expect(() => parseArgs(['--only', 'quickdraw'])).toThrow(/unknown dataset "quickdraw"/);
  });

  it('rejects a prototype member id such as "constructor"', () => {
    expect(() => parseArgs(['--only', 'constructor'])).toThrow(/unknown dataset/i);
  });

  it('rejects an unknown argument', () => {
    expect(() => parseArgs(['--nope'])).toThrow(/unknown argument "--nope"/);
  });
});

describe('buildTargetPath', () => {
  it('builds a path under public/datasets/<id>/', () => {
    const target = buildTargetPath('fashion-mnist', 'train-images-idx3-ubyte.gz');
    expect(target).toMatch(
      /public[\\/]datasets[\\/]fashion-mnist[\\/]train-images-idx3-ubyte\.gz$/,
    );
  });

  it('rejects an id or name that escapes the target directory', () => {
    expect(() => buildTargetPath('sub/dir', 'x.gz')).toThrow(/invalid dataset id/);
    expect(() => buildTargetPath('mnist', '../x.gz')).toThrow(/invalid file name/);
  });
});

describe('describePlan', () => {
  it('lists the files of the selected datasets', () => {
    const plan = describePlan(parseArgs(['--only', 'fashion-mnist']));
    expect(plan).toMatch(/fashion-mnist/);
    expect(plan).toMatch(/train-images-idx3-ubyte\.gz/);
    expect(plan).not.toMatch(/# mnist/);
  });
});

describe('QUICKDRAW_FILES', () => {
  it('lists the fifteen QuickDraw files with unique names, positive sizes and 32-hex MD5s', () => {
    expect(QUICKDRAW_FILES).toHaveLength(15);
    const names = QUICKDRAW_FILES.map((file) => file.name);
    expect(new Set(names).size).toBe(names.length);
    expect(names).toEqual([
      'book.npy',
      'moon.npy',
      'cat.npy',
      'umbrella.npy',
      'cake.npy',
      'dog.npy',
      'fish.npy',
      'flower.npy',
      'star.npy',
      'house.npy',
      'tree.npy',
      'sun.npy',
      'bicycle.npy',
      'pizza.npy',
      'apple.npy',
    ]);
    for (const file of QUICKDRAW_FILES) {
      expect(file.size).toBeGreaterThan(0);
      expect(file.md5).toMatch(MD5_PATTERN);
    }
  });

  it('is not part of DATASET_SOURCES', () => {
    expect(Object.hasOwn(DATASET_SOURCES, 'quickdraw')).toBe(false);
    expect(DATASET_SOURCE_IDS).toEqual(['mnist', 'fashion-mnist']);
  });
});

describe('MEDMNIST_FILES', () => {
  it('lists the seven MedMNIST files with unique ids, names, positive sizes and 32-hex MD5s', () => {
    expect(MEDMNIST_FILES).toHaveLength(7);
    const ids = MEDMNIST_FILES.map((file) => file.dataset);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toEqual([
      'octmnist',
      'pneumoniamnist',
      'breastmnist',
      'tissuemnist',
      'organamnist',
      'organcmnist',
      'organsmnist',
    ]);
    const names = MEDMNIST_FILES.map((file) => file.name);
    expect(new Set(names).size).toBe(names.length);
    for (const file of MEDMNIST_FILES) {
      expect(file.size).toBeGreaterThan(0);
      expect(file.md5).toMatch(MD5_PATTERN);
    }
  });

  it('serves the `.npz` archives from the Zenodo record', () => {
    expect(MEDMNIST_BASE_URL).toBe('https://zenodo.org/records/10519652/files/');
    for (const file of MEDMNIST_FILES) {
      expect(file.name.endsWith('.npz')).toBe(true);
      expect(`${MEDMNIST_BASE_URL}${file.name}`).toMatch(/^https:\/\/zenodo\.org\//);
    }
  });

  it('is not part of DATASET_SOURCES', () => {
    for (const file of MEDMNIST_FILES) {
      expect(Object.hasOwn(DATASET_SOURCES, file.dataset)).toBe(false);
      expect(DATASET_SOURCE_IDS).not.toContain(file.dataset);
    }
  });
});

describe('DATASET_SOURCES', () => {
  it('has exactly the two dataset ids', () => {
    expect(DATASET_SOURCE_IDS).toEqual(['mnist', 'fashion-mnist']);
  });

  it('gives each dataset four files with positive sizes and 32-hex MD5s', () => {
    for (const id of DATASET_SOURCE_IDS) {
      const { files } = DATASET_SOURCES[id];
      expect(files.length).toBe(4);
      expect(files.map((file) => file.name)).toEqual(EXPECTED_FILES);
      for (const file of files) {
        expect(file.size).toBeGreaterThan(0);
        expect(file.md5).toMatch(MD5_PATTERN);
      }
    }
  });
});

/** Répertoires temporaires créés par les tests, nettoyés après chaque test. */
const tempDirs = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function makeTempDir() {
  const dir = await mkdtemp(join(tmpdir(), 'fetch-datasets-'));
  tempDirs.push(dir);
  return dir;
}

/** Jeu de données minimal : un seul fichier `x.bin` avec sa taille et son MD5. */
function fakeSources(bytes, md5 = createHash('md5').update(bytes).digest('hex')) {
  return {
    fake: {
      files: [
        {
          name: 'x.bin',
          size: bytes.length,
          md5,
          url: 'https://example.test/x.bin',
        },
      ],
    },
  };
}

describe('main with injected sources', () => {
  it('writes a verified download and leaves no temporary file', async () => {
    const targetRoot = await makeTempDir();
    const bytes = new Uint8Array([1, 2, 3, 4, 5]);
    const sources = fakeSources(bytes);
    const fetchImpl = async () => new Response(bytes);

    expect(await main(['--only', 'fake'], { sources, targetRoot, fetchImpl })).toBe(0);

    const written = await readFile(join(targetRoot, 'fake', 'x.bin'));
    expect([...written]).toEqual([...bytes]);
    expect((await readdir(join(targetRoot, 'fake'))).sort()).toEqual(['CACHEDIR.TAG', 'x.bin']);
  });

  it('fails on a checksum mismatch and leaves no target or temporary file', async () => {
    const targetRoot = await makeTempDir();
    const bytes = new Uint8Array([1, 2, 3, 4, 5]);
    const sources = fakeSources(bytes, '0'.repeat(32));
    const fetchImpl = async () => new Response(bytes);

    expect(await main(['--only', 'fake'], { sources, targetRoot, fetchImpl })).toBe(1);

    const entries = await readdir(join(targetRoot, 'fake'));
    expect(entries).toEqual(['CACHEDIR.TAG']);
  });
});

describe('main', () => {
  it('prints the usage and returns 1 on an unknown argument', async () => {
    expect(await main(['--nope'])).toBe(1);
  });

  it('returns 0 for a dry-run', async () => {
    expect(await main(['--dry-run'])).toBe(0);
  });
});
