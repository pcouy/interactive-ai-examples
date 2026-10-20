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
  /** Nombre de pièces sur le plateau pour cette position, afin que le
   * navigateur de table de transposition affiche le tour intra-partie ; absent
   * sur les entrées construites à la main. */
  readonly pieces?: number;
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

/**
 * Table de transposition en lecture seule par-dessus une table réelle.
 * WHY: le rejoué dans l'arbre doit montrer exactement la recherche qui a mené
 * au dernier coup, sans jamais polluer la table partagée. Il écrit donc ses
 * propres entrées dans une table temporaire privée ; la table réelle n'est lue
 * qu'en lecture.
 * - `store` n'écrit que dans la table temporaire ;
 * - `probe` et `snapshot` préfèrent l'entrée la plus profonde : la table
 *   temporaire quand elle est au moins aussi profonde que l'entrée réelle,
 *   sinon l'entrée réelle avec `turn: -1` ; à profondeur égale, l'entrée
 *   temporaire l'emporte. Une entrée réelle à profondeur égale n'est donc
 *   jamais traitée comme du tour courant (seule une entrée strictement plus
 *   profonde reste consommable) ;
 * - `currentTurn` / `beginTurn` / `clear` ne pilotent que la table temporaire.
 */
export function createReadOnlyOverlay(real: TranspositionTable): TranspositionTable {
  const temp = createTranspositionTable();
  return {
    store(key: string, entry: TranspositionEntry): void {
      temp.store(key, entry);
    },
    probe(key: string): ProbedEntry | null {
      const own = temp.probe(key);
      const fromReal = real.probe(key);
      if (own === null) {
        return fromReal === null ? null : { ...fromReal, turn: -1 };
      }
      if (fromReal === null) {
        return own;
      }
      // À profondeur égale, l'entrée temporaire (du tour courant) l'emporte.
      return own.depth >= fromReal.depth ? own : { ...fromReal, turn: -1 };
    },
    snapshot(): readonly TranspositionSnapshotEntry[] {
      const merged = new Map<string, TranspositionSnapshotEntry>();
      for (const entry of real.snapshot()) {
        merged.set(entry.key, { ...entry, turn: -1 });
      }
      for (const entry of temp.snapshot()) {
        const existing = merged.get(entry.key);
        if (existing === undefined || entry.depth >= existing.depth) {
          merged.set(entry.key, entry);
        }
      }
      return [...merged.values()].sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
    },
    beginTurn(): void {
      temp.beginTurn();
    },
    currentTurn(): number {
      return temp.currentTurn();
    },
    clear(): void {
      temp.clear();
    },
    size(): number {
      const keys = new Set<string>();
      for (const entry of temp.snapshot()) {
        keys.add(entry.key);
      }
      for (const entry of real.snapshot()) {
        keys.add(entry.key);
      }
      return keys.size;
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
