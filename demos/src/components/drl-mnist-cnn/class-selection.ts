/**
 * Sélection des classes du composant `drl-mnist-cnn`.
 *
 * Le module est purement fonctionnel : il ne dépend que du type du registre des
 * jeux, ni du DOM ni de `fetch`. Il normalise la liste des classes actives,
 * applique une case à cocher et convertit les index entre l'ordre de la
 * sélection et la sortie du réseau.
 *
 * L'ordre de `enabled` est l'ordre de sortie du réseau : les helpers de
 * conversion le lisent tel quel. Les fonctions `defaultEnabledClasses`,
 * `normalizedEnabledClasses` et `toggleEnabledClass` retournent toujours une
 * liste triée par ordre croissant, sans doublon.
 */

import type { DatasetDefinition } from '../../core/dataset-registry.js';

/** Nombre minimal de classes actives. */
export const MIN_ENABLED_CLASSES = 2;

/** Rejette un nombre de classes qui n'est pas un entier strictement positif. */
function requireClassCount(classCount: number): void {
  if (!Number.isInteger(classCount) || classCount < 1) {
    throw new RangeError(
      `class-selection: classCount must be a positive integer, received ${String(classCount)}.`,
    );
  }
}

/** Toutes les classes d'un jeu, dans l'ordre croissant. */
function allClasses(classCount: number): readonly number[] {
  return Array.from({ length: classCount }, (_unused, index) => index);
}

/**
 * Classes actives d'un jeu au démarrage. Un jeu non sélectionnable active
 * toujours toutes ses classes. Un jeu sélectionnable utilise
 * `defaultSelectedClasses`, ou toutes les classes quand ce champ est absent.
 * Le résultat est normalisé : trié, sans doublon et borné au nombre de classes.
 */
export function defaultEnabledClasses(dataset: DatasetDefinition): readonly number[] {
  const classCount = dataset.classNames.length;
  if (!dataset.selectable) {
    return allClasses(classCount);
  }
  return normalizedEnabledClasses(classCount, dataset.defaultSelectedClasses ?? null);
}

/**
 * Normalise une sélection de classes : entiers de `[0, classCount)`, sans
 * doublon, triés par ordre croissant. `null`, une liste vide ou une liste qui
 * donne moins de `MIN_ENABLED_CLASSES` classes valides active toutes les
 * classes, pour ne jamais produire un modèle à une seule sortie.
 */
export function normalizedEnabledClasses(
  classCount: number,
  enabled: readonly number[] | null,
): readonly number[] {
  requireClassCount(classCount);
  if (enabled === null || enabled.length === 0) {
    return allClasses(classCount);
  }
  const selected = new Set<number>();
  for (const value of enabled) {
    if (Number.isInteger(value) && value >= 0 && value < classCount) {
      selected.add(value);
    }
  }
  if (selected.size < MIN_ENABLED_CLASSES) {
    return allClasses(classCount);
  }
  return Array.from(selected).sort((left, right) => left - right);
}

/**
 * Coche ou décoche la classe `index`. Retourne la nouvelle sélection triée, ou
 * `null` quand décocher ferait passer la sélection sous `MIN_ENABLED_CLASSES`.
 * Cocher reste toujours possible, même sous le minimum.
 */
export function toggleEnabledClass(
  enabled: readonly number[],
  index: number,
): readonly number[] | null {
  if (!Number.isInteger(index) || index < 0) {
    throw new RangeError(
      `class-selection: class index must be a non-negative integer, received ${String(index)}.`,
    );
  }
  const current = new Set<number>();
  for (const value of enabled) {
    if (Number.isInteger(value) && value >= 0) {
      current.add(value);
    }
  }
  if (current.has(index)) {
    if (current.size - 1 < MIN_ENABLED_CLASSES) {
      return null;
    }
    current.delete(index);
  } else {
    current.add(index);
  }
  return Array.from(current).sort((left, right) => left - right);
}

/** Copie triée par ordre croissant de `enabled`. */
export function classOrder(enabled: readonly number[]): readonly number[] {
  return [...enabled].sort((left, right) => left - right);
}

/**
 * Indice de sortie du réseau pour la classe `classIndex`, ou `null` quand elle
 * n'est pas active. L'ordre de `enabled` définit la sortie du réseau.
 */
export function networkClassIndex(enabled: readonly number[], classIndex: number): number | null {
  const position = enabled.indexOf(classIndex);
  return position === -1 ? null : position;
}

/**
 * Classe d'origine affichée pour l'indice de sortie `networkIndex`. L'ordre de
 * `enabled` définit la sortie du réseau. Rejette un indice hors de la sélection.
 */
export function displayClass(enabled: readonly number[], networkIndex: number): number {
  if (!Number.isInteger(networkIndex) || networkIndex < 0 || networkIndex >= enabled.length) {
    throw new RangeError(
      `class-selection: networkIndex must be an integer in [0, ${String(enabled.length)}), received ${String(networkIndex)}.`,
    );
  }
  return enabled[networkIndex];
}
