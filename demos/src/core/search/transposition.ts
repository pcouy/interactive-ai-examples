export interface TranspositionEntry {
  // Le paramètre M est conservé pour la compatibilité des appelants typés :
  // les coups ne sont plus stockés, seules la valeur et la profondeur le sont.
  /** Valeur intrinsèque à l'état, exprimée du point de vue du joueur qui doit
   * jouer dans CETTE position (`game.currentPlayer(state)`), si bien qu'elle
   * est réutilisable quel que soit le joueur racine de la recherche courante. */
  readonly score: number;
  /** Profondeur effective certifiée (en plies) pour laquelle `score` est
   * exact : la valeur ne dépend plus de l'horizon jusqu'à cette profondeur.
   * Vaut `SOLVED_DEPTH` lorsque la position est résolue indépendamment de
   * l'horizon (toutes les lignes aboutissent à un terminal). */
  readonly depth: number;
  readonly exact: boolean;
  /** Nombre de tirages sous-jacents, présent uniquement pour les entrées de
   * feuilles rollout à profondeur 0. */
  readonly samples?: number;
}

/** Profondeur certifiée d'une position entièrement résolue : sa valeur ne
 * dépend plus de l'horizon (toutes les lignes aboutissent à un terminal). */
export const SOLVED_DEPTH = Number.POSITIVE_INFINITY;

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
  strict = false,
): boolean {
  // Une entrée plus profonde que la recherche courante sert dans tous les
  // cas. À profondeur égale, elle ne sert que si elle vient du tour courant :
  // les tours précédents ne peuvent fournir que du strictement plus profond.
  // C'est la profondeur certifiée STRICTEMENT supérieure qui accroît la
  // profondeur effective ; à profondeur égale il faut redévelopper (un seul
  // cas à part, traité AVANT cet appel : le cache-feuille rollout à
  // profondeur 0, qui fusionne ses tirages).
  const depthOk =
    entry.depth > remainingDepth || (entry.depth === remainingDepth && entry.turn === currentTurn);
  if (!depthOk) {
    return false;
  }
  if (entry.exact) {
    return true;
  }
  // Le mode strict reflète la règle de coupure du moteur : une entrée bornée
  // n'est utilisable que si elle dépasse STRICTEMENT la borne, sans quoi la
  // valeur exactement égale ne serait pas équivalente au fail-soft obtenu.
  return maximizing
    ? strict
      ? entry.score > beta
      : entry.score >= beta
    : strict
      ? entry.score < alpha
      : entry.score <= alpha;
}
