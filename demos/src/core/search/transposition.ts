export interface TranspositionEntry {
  // Le paramètre M est conservé pour la compatibilité des appelants typés :
  // les coups ne sont plus stockés, seules la valeur et la profondeur le sont.
  readonly score: number;
  readonly depth: number;
  readonly exact: boolean;
  /** Nombre de tirages sous-jacents, présent uniquement pour les entrées de
   * feuilles rollout à profondeur 0. */
  readonly samples?: number;
}

export type ProbedEntry = TranspositionEntry & { readonly turn: number };

export interface TranspositionSnapshotEntry extends ProbedEntry {
  readonly key: string;
}

export interface TranspositionTable {
  store(key: string, entry: TranspositionEntry): void;
  probe(key: string): ProbedEntry | null;
  /** Instantané de toutes les entrées, triées par clé (pour inspection). */
  snapshot(): readonly TranspositionSnapshotEntry[];
  /** Marque le début d'un nouveau tour : les entrées à profondeur égale des
   * tours précédents ne pourront plus être consommées. */
  beginTurn(): void;
  currentTurn(): number;
  clear(): void;
  size(): number;
}

export function createTranspositionTable(): TranspositionTable {
  const entries = new Map<string, ProbedEntry>();
  let turn = 0;
  return {
    store(key: string, entry: TranspositionEntry): void {
      const existing = entries.get(key);
      if (existing === undefined || entry.depth >= existing.depth) {
        entries.set(key, { ...entry, turn });
      }
    },
    probe(key: string): ProbedEntry | null {
      return entries.get(key) ?? null;
    },
    snapshot(): readonly TranspositionSnapshotEntry[] {
      return [...entries.entries()]
        .map(([key, entry]) => ({ ...entry, key }))
        .sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
    },
    beginTurn(): void {
      turn += 1;
    },
    currentTurn(): number {
      return turn;
    },
    clear(): void {
      entries.clear();
    },
    size(): number {
      return entries.size;
    },
  };
}

export function isEntryUsable(
  entry: ProbedEntry,
  remainingDepth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  currentTurn: number,
): boolean {
  // Une entrée plus profonde que la recherche courante sert dans tous les
  // cas. À profondeur égale, elle ne sert que si elle vient du tour courant :
  // les tours précédents ne peuvent fournir que du strictement plus profond.
  const depthOk =
    entry.depth > remainingDepth || (entry.depth === remainingDepth && entry.turn === currentTurn);
  if (!depthOk) {
    return false;
  }
  if (entry.exact) {
    return true;
  }
  return maximizing ? entry.score >= beta : entry.score <= alpha;
}
