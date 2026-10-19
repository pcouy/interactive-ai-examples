# TODO — Deferred work from the FrozenLake Q-learning branch

This file lists work that the `feat/frozen-lake-q-learning` branch discovered but does not contain.
The work touches other demos or repo-wide infrastructure, so we deferred it on purpose.

## Shared RNG module

The `mulberry32` seeded PRNG is copied into 7 files under `demos/src/core/`.
All 7 copies are byte-for-byte identical.

| File | Origin |
|---|---|
| `demos/src/core/perceptron.ts` | Pre-existing (perceptron demo) |
| `demos/src/core/gradient-descent.ts` | Pre-existing (gradient-descent demo) |
| `demos/src/core/mnist-data.ts` | Pre-existing (MNIST demo, other workstream) |
| `demos/src/core/frozen-lake.ts` | Added by this branch |
| `demos/src/core/q-learning.test.ts` | Added by this branch |
| `demos/src/core/frozen-lake.test.ts` | Added by this branch |
| `demos/src/core/replay-buffer.test.ts` | Added by this branch |

### Proposed fix

1. Create `demos/src/core/rng.ts`.
2. Export `mulberry32` from that module.
3. Migrate all 7 call sites to import it.
4. Remove the local copies.

### Why it is deferred

The fix is repo-wide infrastructure.
It changes 3 files that belong to other demos and are outside this branch.
A migration on this branch would widen the branch scope and risk conflicts with the other workstreams.

### Coordination note

`demos/src/core/mnist-data.ts` belongs to another active workstream.
Coordinate with that workstream before you migrate `mnist-data.ts`.

## Comment language consistency

`demos/src/core/contracts.ts` mixes French doc comments with the English comment convention.
The file uses French for the JSDoc on `Game.pieceCount`, `Evaluator.samplesPerEvaluation`, `SearchStats.effectiveDepth`, `CutInfo`, and other members.

This issue is pre-existing and repo-wide.
The branch did not introduce it, and it is outside the branch scope.
