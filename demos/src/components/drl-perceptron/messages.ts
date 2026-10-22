import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

export type PerceptronMessageKey =
  | 'missingValue'
  | 'classBlue'
  | 'classRed'
  | 'headlineWrong'
  | 'headlineUpdate'
  | 'detail'
  | 'hintShowRule'
  | 'hintNextPoint'
  | 'ruleTitle'
  | 'ruleW1'
  | 'ruleW2'
  | 'ruleBias'
  | 'status'
  | 'converged'
  | 'ariaControls'
  | 'presetSeparable'
  | 'presetNoisy'
  | 'presetNonSeparable'
  | 'classBlueButton'
  | 'classRedButton'
  | 'clear'
  | 'svgAria'
  | 'labelX1'
  | 'labelX2'
  | 'labelSum'
  | 'labelThreshold'
  | 'labelOutput'
  | 'weightW1'
  | 'weightW2'
  | 'biasValue';

export const PERCEPTRON_MESSAGES: Catalog<PerceptronMessageKey> = {
  fr: {
    missingValue: '–',
    classBlue: 'classe bleue',
    classRed: 'classe rouge',
    headlineWrong: 'Prédiction erronée',
    headlineUpdate: 'Règle de mise à jour appliquée',
    detail: 'ŷ = {prediction} ({className}) · attendu t = {expected} ({expectedClass}).',
    hintShowRule: 'Step : afficher la règle de mise à jour.',
    hintNextPoint: 'Step : présenter le point suivant.',
    ruleTitle: 'Règle de mise à jour',
    ruleW1: 'w₁ ← w₁ + η·(t − ŷ)·x₁ = {before} + {rate}·{error}·{x1} = {after}',
    ruleW2: 'w₂ ← w₂ + η·(t − ŷ)·x₂ = {before} + {rate}·{error}·{x2} = {after}',
    ruleBias: 'b ← b + η·(t − ŷ) = {before} + {rate}·{error} = {after}',
    status: 'Époque {epoch} · Précision {accuracy} % · {presentations} présentations',
    converged: 'Convergé en {steps} étapes',
    ariaControls: 'Contrôles du perceptron',
    presetSeparable: 'Séparables',
    presetNoisy: 'Bruités',
    presetNonSeparable: 'Non séparables',
    classBlueButton: 'Classe bleue',
    classRedButton: 'Classe rouge',
    clear: 'Effacer',
    svgAria: 'Plan de points et frontière de décision du perceptron',
    labelX1: 'x₁',
    labelX2: 'x₂',
    labelSum: 'Σ',
    labelThreshold: 'seuil',
    labelOutput: 'ŷ',
    weightW1: 'w₁ = {value}',
    weightW2: 'w₂ = {value}',
    biasValue: 'b = {value}',
  },
  en: {
    missingValue: '–',
    classBlue: 'blue class',
    classRed: 'red class',
    headlineWrong: 'Wrong prediction',
    headlineUpdate: 'Update rule applied',
    detail: 'ŷ = {prediction} ({className}) · expected t = {expected} ({expectedClass}).',
    hintShowRule: 'Step: show the update rule.',
    hintNextPoint: 'Step: present the next point.',
    ruleTitle: 'Update rule',
    ruleW1: 'w₁ ← w₁ + η·(t − ŷ)·x₁ = {before} + {rate}·{error}·{x1} = {after}',
    ruleW2: 'w₂ ← w₂ + η·(t − ŷ)·x₂ = {before} + {rate}·{error}·{x2} = {after}',
    ruleBias: 'b ← b + η·(t − ŷ) = {before} + {rate}·{error} = {after}',
    status: 'Epoch {epoch} · Accuracy {accuracy} % · {presentations} presentations',
    converged: 'Converged in {steps} steps',
    ariaControls: 'Perceptron controls',
    presetSeparable: 'Separable',
    presetNoisy: 'Noisy',
    presetNonSeparable: 'Non-separable',
    classBlueButton: 'Blue class',
    classRedButton: 'Red class',
    clear: 'Clear',
    svgAria: 'Point plane and perceptron decision boundary',
    labelX1: 'x₁',
    labelX2: 'x₂',
    labelSum: 'Σ',
    labelThreshold: 'threshold',
    labelOutput: 'ŷ',
    weightW1: 'w₁ = {value}',
    weightW2: 'w₂ = {value}',
    biasValue: 'b = {value}',
  },
};

export function perceptronMessage(
  locale: Locale,
  key: PerceptronMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, PERCEPTRON_MESSAGES, key, params);
}
