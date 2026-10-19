/**
 * Logique pure du formulaire d'édition d'un poids du composant `drl-mnist-cnn`.
 *
 * Le geste est linéaire : un clic sur une liaison sélectionne un couple
 * (canal d'entrée, canal de sortie), puis un clic sur une case de la toile de
 * filtre en aval produit une sélection de poids. Ce module transforme cette
 * sélection en vue de formulaire, formate la valeur courante et applique une
 * saisie au clavier. Il n'importe ni Lit ni le DOM : il ne dépend que des types
 * du cœur `core/mnist-cnn.ts` et des aides exportées par le composant.
 */

import {
  type MnistCnnFilters,
  type MnistCnnWeights,
  withKernelWeight,
} from '../../core/mnist-cnn.js';
import {
  describeWeightSelection,
  parseWeightInput,
  selectedWeightValue,
  type WeightSelection,
} from './drl-mnist-cnn.js';

/** Vue du formulaire pour une sélection : libellé, valeur courante et texte saisi. */
export interface WeightEditorView {
  /** Légende française du poids sélectionné (voir `describeWeightSelection`). */
  readonly label: string;
  /** Valeur numérique courante du poids. */
  readonly value: number;
  /** Valeur formatée pour le champ de saisie (voir `formatWeightValue`). */
  readonly text: string;
}

/** Résultat de l'application d'un brouillon : poids écrits, ou message d'erreur français. */
export type WeightApplyResult =
  | { readonly kind: 'applied'; readonly weights: MnistCnnWeights; readonly value: number }
  | { readonly kind: 'error'; readonly message: string };

/**
 * Déplie une notation exponentielle (`1e-7`) en décimal (`0.0000001`). Les poids
 * extrêmes restent lisibles dans le champ et analysables par `parseWeightInput`,
 * qui refuse l'exposant.
 */
function expandExponential(text: string): string {
  const [mantissa, exponentText] = text.split(/[eE]/);
  const exponent = Number.parseInt(exponentText, 10);
  const sign = mantissa.startsWith('-') ? '-' : '';
  const [integer, fraction = ''] = (sign === '' ? mantissa : mantissa.slice(1)).split('.');
  const digits = integer + fraction;
  const point = integer.length + exponent;
  if (point <= 0) {
    return `${sign}0.${'0'.repeat(-point)}${digits}`;
  }
  if (point >= digits.length) {
    return `${sign}${digits}${'0'.repeat(point - digits.length)}`;
  }
  return `${sign}${digits.slice(0, point)}.${digits.slice(point)}`;
}

/**
 * Représentation compacte et réversible d'un poids pour le champ de saisie :
 * `String` donne la forme décimale la plus courte qui relit la même valeur, et
 * `parseWeightInput` l'accepte. Une notation exponentielle (valeurs extrêmes)
 * est dépliée en décimal. Exige une valeur finie.
 */
export function formatWeightValue(value: number): string {
  if (!Number.isFinite(value)) {
    throw new RangeError(`formatWeightValue: value must be finite, received ${String(value)}.`);
  }
  const text = String(value);
  return text.includes('e') || text.includes('E') ? expandExponential(text) : text;
}

/**
 * Vue du formulaire pour une sélection, ou null si la sélection n'est pas un
 * poids de l'architecture courante. Le libellé vient de
 * `describeWeightSelection` et la valeur de `selectedWeightValue` : une
 * sélection hors architecture ou un indice hors bornes donne null.
 */
export function weightEditorView(
  selection: WeightSelection,
  weights: MnistCnnWeights,
  filters: MnistCnnFilters,
): WeightEditorView | null {
  if (selection === null || selection.kind !== 'weight') {
    return null;
  }
  const label = describeWeightSelection(selection, filters);
  const value = selectedWeightValue(weights, selection);
  if (label === null || value === null || !Number.isFinite(value)) {
    return null;
  }
  return { label, value, text: formatWeightValue(value) };
}

/**
 * Applique un brouillon saisi au clavier au poids sélectionné : analyse le texte
 * (`.` ou `,` comme séparateur décimal), puis écrit le poids avec
 * `withKernelWeight`. `weights` n'est jamais modifié. En cas d'échec, retourne
 * un message français qui nomme le problème : champ vide, nombre illisible,
 * valeur non finie, sélection qui n'est pas un poids, ou poids hors architecture.
 */
export function applyWeightDraft(
  draft: string,
  selection: WeightSelection,
  weights: MnistCnnWeights,
  filters: MnistCnnFilters,
): WeightApplyResult {
  if (selection === null || selection.kind !== 'weight') {
    return { kind: 'error', message: "La sélection courante n'est pas un poids." };
  }
  const text = draft.trim();
  if (text === '') {
    return { kind: 'error', message: 'Saisissez une valeur.' };
  }
  const value = parseWeightInput(text);
  if (value === null) {
    const numeric = Number(text.replace(',', '.'));
    if (!Number.isFinite(numeric) && !Number.isNaN(numeric)) {
      return { kind: 'error', message: 'Valeur non finie : le nombre est trop grand.' };
    }
    return {
      kind: 'error',
      message: 'Nombre illisible : utilisez un décimal, par exemple 0,5.',
    };
  }
  if (describeWeightSelection(selection, filters) === null) {
    return { kind: 'error', message: "Ce poids n'existe pas dans l'architecture courante." };
  }
  const applied = withKernelWeight(
    weights,
    filters,
    selection.boundary,
    selection.inputChannel,
    selection.outputChannel,
    selection.ky,
    selection.kx,
    value,
  );
  return { kind: 'applied', weights: applied, value };
}

/** Une phrase française qui décrit le geste de l'éditeur de poids. */
export function weightEditorHint(): string {
  return 'Cliquez une liaison, puis une case de la toile de filtre en aval, puis saisissez la valeur.';
}
