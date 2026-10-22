import type { Locale } from '../i18n/locale.js';
import { type Catalog, translate } from '../i18n/translate.js';

export type UiMessageKey = 'play' | 'pause' | 'step' | 'reset' | 'speed' | 'sliderValueFallback';

export const UI_MESSAGES: Catalog<UiMessageKey> = {
  fr: {
    play: 'Lecture',
    pause: 'Pause',
    step: 'Pas',
    reset: 'Réinitialiser',
    speed: 'Vitesse',
    sliderValueFallback: 'valeur',
  },
  en: {
    play: 'Play',
    pause: 'Pause',
    step: 'Step',
    reset: 'Reset',
    speed: 'Speed',
    sliderValueFallback: 'value',
  },
};

export function uiMessage(locale: Locale, key: UiMessageKey): string {
  return translate(locale, UI_MESSAGES, key);
}
