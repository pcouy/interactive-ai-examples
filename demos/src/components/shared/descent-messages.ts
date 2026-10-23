import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

export type DescentMessageKey =
  | 'phaseIdle'
  | 'phasePresent'
  | 'phaseGradient'
  | 'phaseFade'
  | 'phaseMean'
  | 'phaseMove'
  | 'statusExample'
  | 'statGradient'
  | 'webglError';

export const DESCENT_MESSAGES: Catalog<DescentMessageKey> = {
  fr: {
    phaseIdle: 'Prêt',
    phasePresent: 'Exemple',
    phaseGradient: 'Gradient',
    phaseFade: 'Archivage',
    phaseMean: 'Moyenne',
    phaseMove: 'Descente',
    statusExample: 'Exemple {index}/{total} : (x₁, x₂) = ({x1} ; {x2}) → t = {t}',
    statGradient: '|gradient|',
    webglError: "WebGL n'est pas disponible dans cet environnement.",
  },
  en: {
    phaseIdle: 'Ready',
    phasePresent: 'Example',
    phaseGradient: 'Gradient',
    phaseFade: 'Archive',
    phaseMean: 'Average',
    phaseMove: 'Descent',
    statusExample: 'Example {index}/{total}: (x₁, x₂) = ({x1}, {x2}) → t = {t}',
    statGradient: '|gradient|',
    webglError: 'WebGL is not available in this environment.',
  },
};

export function descentMessage(
  locale: Locale,
  key: DescentMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, DESCENT_MESSAGES, key, params);
}
