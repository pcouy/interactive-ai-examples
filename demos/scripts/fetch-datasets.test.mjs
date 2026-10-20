import { describe, expect, it } from 'vitest';
import { DATASET_SOURCE_IDS, DATASET_SOURCES } from './dataset-sources.mjs';
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

describe('main', () => {
  it('prints the usage and returns 1 on an unknown argument', async () => {
    expect(await main(['--nope'])).toBe(1);
  });

  it('returns 0 for a dry-run', async () => {
    expect(await main(['--dry-run'])).toBe(0);
  });
});
