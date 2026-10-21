/**
 * Helpers purs de la galerie d'exemples du composant `drl-mnist-cnn`.
 *
 * Le module ne dépend ni du DOM, ni de `fetch`, ni du registre des jeux : il
 * ne manipule que des libellés, une clé de cache et l'ordre des classes.
 * Chaque jeu peut proposer une galerie par fraction (`train` ou `test`), et
 * les appelants choisissent la fraction à afficher.
 */

/** Fractions pour lesquelles un jeu peut proposer une galerie. */
export type GallerySplit = 'train' | 'test';

/** Fractions de galerie dans l'ordre d'affichage. */
export const GALLERY_SPLITS: readonly GallerySplit[] = ['train', 'test'];

/** Libellé court d'une fraction de galerie. */
export function gallerySplitLabel(split: GallerySplit): string {
  return split === 'train' ? 'Entraînement' : 'Validation';
}

/** Complément de nom utilisé dans les phrases (« exemple d'entraînement »). */
export function gallerySourceLabel(split: GallerySplit): string {
  return split === 'train' ? "d'entraînement" : 'de validation';
}

/**
 * Clé de cache d'une galerie : identifiant du jeu, fraction et nombre
 * d'exemples par classe. Deux fractions du même jeu ne partagent jamais leur
 * entrée de cache.
 */
export function galleryCacheKey(datasetId: string, split: GallerySplit, perClass: number): string {
  return `${datasetId}/${split}/${perClass}`;
}

/**
 * Nombre d'exemples par classe d'une galerie de `sampleCount` images réparties
 * équitablement entre `classCount` classes. Lève une `RangeError` quand les
 * effectifs ne sont pas des entiers valides ou quand la division n'est pas
 * exacte.
 */
export function galleryPerClassCount(sampleCount: number, classCount: number): number {
  if (!Number.isInteger(classCount) || classCount < 1) {
    throw new RangeError(
      `gallery: classCount must be a positive integer, received ${String(classCount)}.`,
    );
  }
  if (!Number.isInteger(sampleCount) || sampleCount < 0) {
    throw new RangeError(
      `gallery: sampleCount must be a non-negative integer, received ${String(sampleCount)}.`,
    );
  }
  if (sampleCount % classCount !== 0) {
    throw new RangeError(
      `gallery: sampleCount ${sampleCount} is not divisible by classCount ${classCount}.`,
    );
  }
  return sampleCount / classCount;
}

/**
 * Ordre croissant des classes proposées par la galerie. Quand `enabledClasses`
 * est `null` ou vide, toutes les classes sont proposées. Sinon, seules les
 * classes entières comprises dans `[0, classCount)` sont gardées, sans doublon
 * et triées.
 */
export function galleryClassOrder(
  classCount: number,
  enabledClasses: readonly number[] | null,
): readonly number[] {
  if (!Number.isInteger(classCount) || classCount < 1) {
    throw new RangeError(
      `gallery: classCount must be a positive integer, received ${String(classCount)}.`,
    );
  }
  if (enabledClasses === null || enabledClasses.length === 0) {
    return Array.from({ length: classCount }, (_unused, index) => index);
  }
  const enabled = new Set<number>();
  for (const value of enabledClasses) {
    if (Number.isInteger(value) && value >= 0 && value < classCount) {
      enabled.add(value);
    }
  }
  return Array.from(enabled).sort((left, right) => left - right);
}

/**
 * Classe à afficher : la classe courante quand elle appartient à `order`, sinon
 * la première classe disponible. Retourne `null` quand aucune classe n'est
 * disponible.
 */
export function firstGalleryClass(order: readonly number[], current: number | null): number | null {
  if (order.length === 0) {
    return null;
  }
  if (current !== null && order.includes(current)) {
    return current;
  }
  return order[0];
}
