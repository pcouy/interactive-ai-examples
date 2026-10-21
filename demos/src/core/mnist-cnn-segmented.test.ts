import { describe, expect, it } from 'vitest';
import {
  buildSegmentPlan,
  type RoundEntry,
  SEGMENT_IMAGE_THRESHOLD,
  shouldUseSegments,
} from './mnist-cnn-segmented.js';

/** Fragment synthétique : le nom de fichier suffit à l'identifier. */
function shard(classIndex: number, shardIndex: number, count = 2): RoundEntry {
  return {
    file: `c${classIndex}-s${shardIndex}.idx.gz`,
    classIndex,
    start: shardIndex * count,
    count,
  };
}

/** Fragments réguliers : `shardsPerClass` fragments pour chaque classe. */
function regularShards(classCount: number, shardsPerClass: number, count = 2): RoundEntry[][] {
  return Array.from({ length: classCount }, (_, classIndex) =>
    Array.from({ length: shardsPerClass }, (_, shardIndex) => shard(classIndex, shardIndex, count)),
  );
}

function fingerprint(plan: ReturnType<typeof buildSegmentPlan>, epoch: number): string {
  return JSON.stringify(plan.roundsForEpoch(epoch));
}

describe('shouldUseSegments', () => {
  it('switches strictly above the threshold', () => {
    expect(SEGMENT_IMAGE_THRESHOLD).toBe(60_000);
    expect(shouldUseSegments(SEGMENT_IMAGE_THRESHOLD - 1)).toBe(false);
    expect(shouldUseSegments(SEGMENT_IMAGE_THRESHOLD)).toBe(false);
    expect(shouldUseSegments(SEGMENT_IMAGE_THRESHOLD + 1)).toBe(true);
  });
});

describe('buildSegmentPlan', () => {
  it('builds one round per shard for equal shard counts', () => {
    const shardsByClass = regularShards(15, 4);
    const plan = buildSegmentPlan({ shardsByClass, shardSize: 1000, seed: 7 });
    expect(plan.classes).toBe(15);
    expect(plan.shardSize).toBe(1000);
    expect(plan.totalCount).toBe(15 * 4 * 2);
    expect(plan.roundsPerEpoch).toBe(4);

    const rounds = plan.roundsForEpoch(0);
    expect(rounds).toHaveLength(4);
    expect(new Set(rounds.map((round) => round.index))).toEqual(new Set([0, 1, 2, 3]));
    for (const round of rounds) {
      expect(round.entries).toHaveLength(15);
    }
    // Chaque fragment sélectionné apparaît une seule fois dans l'époque.
    const files = rounds.flatMap((round) => round.entries.map((entry) => entry.file));
    expect(files).toHaveLength(60);
    expect(new Set(files).size).toBe(60);
  });

  it('is deterministic for the same seed and differs for another seed', () => {
    const shardsByClass = regularShards(15, 4);
    const first = buildSegmentPlan({ shardsByClass, shardSize: 1000, seed: 7 });
    const second = buildSegmentPlan({ shardsByClass, shardSize: 1000, seed: 7 });
    const other = buildSegmentPlan({ shardsByClass, shardSize: 1000, seed: 8 });
    expect(fingerprint(first, 0)).toBe(fingerprint(second, 0));
    expect(fingerprint(first, 1)).toBe(fingerprint(second, 1));
    expect(fingerprint(first, 0)).not.toBe(fingerprint(other, 0));
    // Même plan, autre époque : l'ordre change, la couverture reste complète.
    expect(fingerprint(first, 0)).not.toBe(fingerprint(first, 1));
  });

  it('cycles shorter classes when shard counts differ', () => {
    const shardsByClass: RoundEntry[][] = [];
    for (let classIndex = 0; classIndex < 8; classIndex += 1) {
      shardsByClass.push(regularShards(8, classIndex < 4 ? 3 : 7)[classIndex]);
    }
    const plan = buildSegmentPlan({ shardsByClass, shardSize: 1000, seed: 3 });
    expect(plan.roundsPerEpoch).toBe(7);
    expect(plan.totalCount).toBe((4 * 3 + 4 * 7) * 2);

    const rounds = plan.roundsForEpoch(0);
    expect(rounds).toHaveLength(7);
    const ordered = [...rounds].sort((left, right) => left.index - right.index);
    for (let classIndex = 0; classIndex < 8; classIndex += 1) {
      const sequence = ordered.map((round) => round.entries[classIndex].file);
      expect(sequence).toHaveLength(7);
      const expectedShards = classIndex < 4 ? 3 : 7;
      expect(new Set(sequence).size).toBe(expectedShards);
    }
  });

  it('copies the input fragments', () => {
    const shardsByClass = regularShards(2, 2);
    const plan = buildSegmentPlan({ shardsByClass, shardSize: 1000, seed: 1 });
    shardsByClass[0][0] = shard(0, 9);
    const files = plan
      .roundsForEpoch(0)
      .flatMap((round) => round.entries.map((entry) => entry.file));
    expect(files).not.toContain('c0-s9.idx.gz');
  });

  it('rejects an invalid class count', () => {
    expect(() => buildSegmentPlan({ shardsByClass: [], shardSize: 10, seed: 1 })).toThrow(
      /classes/i,
    );
    expect(() =>
      buildSegmentPlan({ shardsByClass: regularShards(1, 2), shardSize: 10, seed: 1 }),
    ).toThrow(/classes/i);
    expect(() =>
      buildSegmentPlan({ shardsByClass: regularShards(21, 2), shardSize: 10, seed: 1 }),
    ).toThrow(/classes/i);
  });

  it('rejects an empty class', () => {
    const shardsByClass = regularShards(2, 2);
    shardsByClass[1] = [];
    expect(() => buildSegmentPlan({ shardsByClass, shardSize: 10, seed: 1 })).toThrow(
      /at least one shard/i,
    );
  });

  it('rejects an invalid shard size, count or seed', () => {
    const shardsByClass = regularShards(2, 2);
    expect(() => buildSegmentPlan({ shardsByClass, shardSize: 0, seed: 1 })).toThrow(/shardSize/i);
    expect(() => buildSegmentPlan({ shardsByClass, shardSize: 1.5, seed: 1 })).toThrow(
      /shardSize/i,
    );
    expect(() => buildSegmentPlan({ shardsByClass, shardSize: 10, seed: -1 })).toThrow(/seed/i);
    expect(() => buildSegmentPlan({ shardsByClass, shardSize: 10, seed: 1.5 })).toThrow(/seed/i);

    const zeroCount = regularShards(2, 2);
    zeroCount[0][0] = { ...zeroCount[0][0], count: 0 };
    expect(() => buildSegmentPlan({ shardsByClass: zeroCount, shardSize: 10, seed: 1 })).toThrow(
      /count/i,
    );

    const fractionalCount = regularShards(2, 2);
    fractionalCount[0][1] = { ...fractionalCount[0][1], count: 1.5 };
    expect(() =>
      buildSegmentPlan({ shardsByClass: fractionalCount, shardSize: 10, seed: 1 }),
    ).toThrow(/count/i);
  });

  it('rejects an invalid epoch', () => {
    const plan = buildSegmentPlan({ shardsByClass: regularShards(2, 2), shardSize: 10, seed: 1 });
    expect(() => plan.roundsForEpoch(-1)).toThrow(/epoch/i);
    expect(() => plan.roundsForEpoch(0.5)).toThrow(/epoch/i);
  });
});
