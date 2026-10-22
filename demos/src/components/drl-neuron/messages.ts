import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

export type NeuronMessageKey =
  | 'phaseIdle'
  | 'phasePresent'
  | 'phaseGradient'
  | 'phaseFade'
  | 'phaseMean'
  | 'phaseMove'
  | 'statusIdle'
  | 'statusExample'
  | 'statusGradient'
  | 'statusFade'
  | 'statusMean'
  | 'statusMove'
  | 'labelW1'
  | 'titleW1'
  | 'labelW2'
  | 'titleW2'
  | 'labelBias'
  | 'titleBias'
  | 'hintBias'
  | 'statLoss'
  | 'statGradient'
  | 'statIterations'
  | 'svgAria'
  | 'legendMax'
  | 'legendMin'
  | 'webglError'
  | 'canvasAria'
  | 'targetError'
  | 'axisCaption';

export const NEURON_MESSAGES: Catalog<NeuronMessageKey> = {
  fr: {
    phaseIdle: 'Prêt',
    phasePresent: 'Exemple',
    phaseGradient: 'Gradient',
    phaseFade: 'Archivage',
    phaseMean: 'Moyenne',
    phaseMove: 'Descente',
    statusIdle: 'Prêt — lancez la descente ou avancez pas à pas',
    statusExample: 'Exemple {index}/{total} : (x₁, x₂) = ({x1} ; {x2}) → t = {t}',
    statusGradient: 'Calcul du gradient de cet exemple',
    statusFade: 'Exemple archivé : son gradient entre dans la moyenne de la batch',
    statusMean: 'Moyenne des gradients de la batch',
    statusMove: 'Mise à jour des poids : w ← w − learning rate × gradient moyen',
    labelW1: 'w₁',
    titleW1: 'Poids w₁',
    labelW2: 'w₂',
    titleW2: 'Poids w₂',
    labelBias: 'b',
    titleBias: "Biais b — cochez « Biais » pour l'éditer",
    hintBias: 'Cochez « Biais » pour éditer b.',
    statLoss: 'Loss (moyenne)',
    statGradient: '|gradient|',
    statIterations: 'Itérations',
    svgAria: "Nuage d'exemples et grille de prédictions du neurone",
    legendMax: 't max',
    legendMin: 't min',
    webglError: "WebGL n'est pas disponible dans cet environnement.",
    canvasAria: 'Surface de loss 3D et descente de la bille par batch',
    targetError: 'Expression invalide : « {expression} » — « x1 + x2 » utilisée par défaut.',
    axisCaption: '{axis} (−{min} … {max})',
  },
  en: {
    phaseIdle: 'Ready',
    phasePresent: 'Example',
    phaseGradient: 'Gradient',
    phaseFade: 'Archive',
    phaseMean: 'Average',
    phaseMove: 'Descent',
    statusIdle: 'Ready — start the descent or step forward',
    statusExample: 'Example {index}/{total}: (x₁, x₂) = ({x1}, {x2}) → t = {t}',
    statusGradient: 'Gradient of this example',
    statusFade: 'Example archived: its gradient joins the batch mean',
    statusMean: 'Mean of the batch gradients',
    statusMove: 'Weight update: w ← w − learning rate × mean gradient',
    labelW1: 'w₁',
    titleW1: 'Weight w₁',
    labelW2: 'w₂',
    titleW2: 'Weight w₂',
    labelBias: 'b',
    titleBias: 'Bias b — check “Bias” to edit it',
    hintBias: 'Check “Bias” to edit b.',
    statLoss: 'Loss (mean)',
    statGradient: '|gradient|',
    statIterations: 'Iterations',
    svgAria: 'Example cloud and neuron prediction grid',
    legendMax: 't max',
    legendMin: 't min',
    webglError: 'WebGL is not available in this environment.',
    canvasAria: '3D loss surface and batch gradient descent ball',
    targetError: 'Invalid expression: “{expression}” — “x1 + x2” used by default.',
    axisCaption: '{axis} (−{min} … {max})',
  },
};

export function neuronMessage(
  locale: Locale,
  key: NeuronMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, NEURON_MESSAGES, key, params);
}
