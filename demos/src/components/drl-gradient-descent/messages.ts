import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

export type GradientDescentMessageKey =
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
  | 'ariaControls'
  | 'buttonStep'
  | 'buttonBatch'
  | 'statEpoch'
  | 'statPosition'
  | 'statLoss'
  | 'statGradient'
  | 'webglError'
  | 'canvasAria'
  | 'hint';

export const GRADIENT_DESCENT_MESSAGES: Catalog<GradientDescentMessageKey> = {
  fr: {
    phaseIdle: 'Prêt',
    phasePresent: 'Exemple',
    phaseGradient: 'Gradient',
    phaseFade: 'Archivage',
    phaseMean: 'Moyenne',
    phaseMove: 'Descente',
    statusIdle: 'Prêt — « Un pas » ou « Une batch »',
    statusExample: 'Exemple {index}/{total} : (x₁, x₂) = ({x1} ; {x2}) → t = {t}',
    statusGradient: 'Flèche : sens de descente pour cet exemple (opposé du gradient)',
    statusFade: 'Flèche archivée en rouge translucide et surface effacée',
    statusMean: 'Moyenne des flèches = sens de descente de la batch',
    statusMove: 'La balle descend : position -= learning rate × gradient de batch',
    ariaControls: 'Contrôles de la descente de gradient',
    buttonStep: 'Un pas',
    buttonBatch: 'Une batch',
    statEpoch: 'Époque',
    statPosition: 'Position (a, b)',
    statLoss: 'Loss',
    statGradient: '|gradient|',
    webglError: "WebGL n'est pas disponible dans cet environnement.",
    canvasAria: 'Surface de loss 3D et descente de gradient par batch',
    hint: 'Glissez pour tourner, clic molette + glisser pour déplacer, molette pour zoomer.',
  },
  en: {
    phaseIdle: 'Ready',
    phasePresent: 'Example',
    phaseGradient: 'Gradient',
    phaseFade: 'Archive',
    phaseMean: 'Average',
    phaseMove: 'Descent',
    statusIdle: 'Ready — “One step” or “One batch”',
    statusExample: 'Example {index}/{total}: (x₁, x₂) = ({x1}, {x2}) → t = {t}',
    statusGradient: 'Arrow: descent direction for this example (opposite of the gradient)',
    statusFade: 'Archived arrow in translucent red and cleared surface',
    statusMean: 'Mean of the arrows = batch descent direction',
    statusMove: 'The ball descends: position -= learning rate × batch gradient',
    ariaControls: 'Gradient descent controls',
    buttonStep: 'One step',
    buttonBatch: 'One batch',
    statEpoch: 'Epoch',
    statPosition: 'Position (a, b)',
    statLoss: 'Loss',
    statGradient: '|gradient|',
    webglError: 'WebGL is not available in this environment.',
    canvasAria: '3D loss surface and batch gradient descent',
    hint: 'Drag to rotate, middle-click and drag to pan, scroll to zoom.',
  },
};

export function gradientDescentMessage(
  locale: Locale,
  key: GradientDescentMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, GRADIENT_DESCENT_MESSAGES, key, params);
}
