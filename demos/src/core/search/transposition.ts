export interface TranspositionEntry<M> {
  readonly score: number;
  readonly depth: number;
  readonly exact: boolean;
  readonly bestMove: M | null;
}

export interface TranspositionTable<M> {
  store(key: string, entry: TranspositionEntry<M>): void;
  probe(key: string): TranspositionEntry<M> | null;
  clear(): void;
  size(): number;
}

export function createTranspositionTable<M>(): TranspositionTable<M> {
  const entries = new Map<string, TranspositionEntry<M>>();
  return {
    store(key: string, entry: TranspositionEntry<M>): void {
      const existing = entries.get(key);
      if (existing === undefined || entry.depth >= existing.depth) {
        entries.set(key, entry);
      }
    },
    probe(key: string): TranspositionEntry<M> | null {
      return entries.get(key) ?? null;
    },
    clear(): void {
      entries.clear();
    },
    size(): number {
      return entries.size;
    },
  };
}

export function isEntryUsable<M>(
  entry: TranspositionEntry<M>,
  remainingDepth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
): boolean {
  if (entry.depth < remainingDepth) {
    return false;
  }
  if (entry.exact) {
    return true;
  }
  return maximizing ? entry.score >= beta : entry.score <= alpha;
}
