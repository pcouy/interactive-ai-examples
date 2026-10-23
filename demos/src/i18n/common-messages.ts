import type { Locale } from './locale.js';
import { type Catalog, type TranslateParams, translate } from './translate.js';

export type CommonMessageKey = 'apply' | 'clear' | 'pause' | 'reset' | 'replay';

export const COMMON_MESSAGES: Catalog<CommonMessageKey> = {
  fr: {
    apply: 'Appliquer',
    clear: 'Effacer',
    pause: 'Pause',
    reset: 'Réinitialiser',
    replay: 'Rejeu',
  },
  en: {
    apply: 'Apply',
    clear: 'Clear',
    pause: 'Pause',
    reset: 'Reset',
    replay: 'Replay',
  },
};

export function commonMessage(
  locale: Locale,
  key: CommonMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, COMMON_MESSAGES, key, params);
}
