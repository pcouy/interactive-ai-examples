import type { CutInfo, VisitKind } from '../../core/contracts.js';
import type { TreeModel } from '../../core/search/tree-model.js';

export const MOVE_FIELDS = ['index', 'col', 'column', 'row'] as const;

export function formatNumber(value: number): string {
  return value.toFixed(2).replace('.', ',');
}

export function formatScore(score: number | null): string {
  if (score === null) {
    return '';
  }
  const rounded = Math.abs(score) < 0.005 ? 0 : score;
  return `${score > 0 ? '+' : ''}${formatNumber(rounded)}`;
}

export function formatMove(move: unknown): string {
  if (move === null || move === undefined) {
    return '';
  }
  if (typeof move === 'number' || typeof move === 'string') {
    return String(move);
  }
  if (typeof move === 'object') {
    const record = move as Record<string, unknown>;
    for (const field of MOVE_FIELDS) {
      const value = record[field];
      if (typeof value === 'number') {
        return String(value);
      }
    }
    const json = JSON.stringify(move);
    return json.length <= 12 ? json : `${json.slice(0, 11)}…`;
  }
  return String(move);
}

/**
 * Ne garde que le premier frère élagué (`pruned`).
 * WHY: plusieurs frères élagués portent la même information de coupure,
 * donc un seul suffit pour montrer que le reste de la fratrie a été élagué.
 */
export function keepFirstPruned<T extends { kind: VisitKind }>(children: readonly T[]): T[] {
  let seenPruned = false;
  return children.filter((child) => {
    if (child.kind !== 'pruned') {
      return true;
    }
    if (seenPruned) {
      return false;
    }
    seenPruned = true;
    return true;
  });
}

/** Résumé affichable des bornes d'un nœud (panneau α/β du visualiseur). */
export interface BoundsSummary {
  readonly nodeKey: string;
  readonly move: unknown;
  readonly alpha: number;
  readonly beta: number;
  readonly score: number | null;
  readonly exact: boolean;
  readonly cut: CutInfo | null;
  /** Coup du fils qui a déclenché la coupure (résolu via le modèle). */
  readonly triggerMove: unknown;
}

/** Résumé du nœud demandé, ou null si la clé est inconnue. */
export function boundsSummary<S>(
  model: TreeModel<S>,
  nodeKey: string | null,
): BoundsSummary | null {
  if (nodeKey === null) {
    return null;
  }
  const node = model.node(nodeKey);
  if (!node) {
    return null;
  }
  const cut = node.cut ?? null;
  const triggerKey = cut?.triggerKey ?? null;
  // Un nœud sans fenêtre enregistrée est non borné : ±∞ rend cet état lisible.
  return {
    nodeKey: node.nodeKey,
    move: node.move,
    alpha: node.window?.alpha ?? Number.NEGATIVE_INFINITY,
    beta: node.window?.beta ?? Number.POSITIVE_INFINITY,
    score: node.score,
    exact: node.exact,
    cut,
    triggerMove: triggerKey === null ? null : (model.node(triggerKey)?.move ?? null),
  };
}

/** Formate une borne : '−∞', '+∞', ou un score signé ('+0,50', '0,00'). */
export function formatBound(value: number): string {
  if (value === Number.NEGATIVE_INFINITY) {
    return '−∞';
  }
  if (value === Number.POSITIVE_INFINITY) {
    return '+∞';
  }
  return formatScore(value);
}

/**
 * Libellé neutre du panneau α/β quand le nœud dessiné n'a pas de coupure.
 * Afficher un texte de longueur comparable garde une hauteur de panneau stable.
 */
export const NO_CUT_LABEL = 'Développement du nœud';

/** Étiquette courte d'une coupure : '> β' (strict) / '≥ β' / '< α' / '≤ α'. */
export function cutLabel(cut: CutInfo): string {
  if (cut.bound === 'beta') {
    return cut.strict ? '> β' : '≥ β';
  }
  return cut.strict ? '< α' : '≤ α';
}

/** Phrase explicative : borne franchie, valeur, coup déclencheur. */
export function cutDescription(cut: CutInfo, triggerMove: unknown): string {
  const head = `meilleur ${formatScore(cut.bestScore)} ${cutLabel(cut)} ${formatScore(cut.limit)}`;
  if (triggerMove === null || triggerMove === undefined) {
    return `${head} · sans coup déclencheur`;
  }
  // formatMove peut renvoyer '' pour un coup non représentable.
  const move = formatMove(triggerMove) || 'coup inconnu';
  return `${head} · déclenché par le coup ${move}`;
}
