import { COMMON_MESSAGES } from '../i18n/common-messages.js';
import type { Locale } from '../i18n/locale.js';
import { type Catalog, translate } from '../i18n/translate.js';

export type UiMessageKey = 'play' | 'pause' | 'step' | 'reset' | 'speed' | 'sliderValueFallback';

export const UI_MESSAGES: Catalog<UiMessageKey> = {
  fr: {
    play: 'Lecture',
    pause: COMMON_MESSAGES.fr.pause,
    step: 'Pas',
    reset: COMMON_MESSAGES.fr.reset,
    speed: 'Vitesse',
    sliderValueFallback: 'valeur',
  },
  en: {
    play: 'Play',
    pause: COMMON_MESSAGES.en.pause,
    step: 'Step',
    reset: COMMON_MESSAGES.en.reset,
    speed: 'Speed',
    sliderValueFallback: 'value',
  },
};

export function uiMessage(locale: Locale, key: UiMessageKey): string {
  return translate(locale, UI_MESSAGES, key);
}
