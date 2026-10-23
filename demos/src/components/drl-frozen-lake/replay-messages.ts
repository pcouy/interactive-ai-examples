import { COMMON_MESSAGES } from '../../i18n/common-messages.js';
import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

export type ReplayMessageKey = 'ariaLabel' | 'header' | 'row' | 'end' | 'lastStep' | 'empty';

export const REPLAY_MESSAGES: Catalog<ReplayMessageKey> = {
  fr: {
    ariaLabel: COMMON_MESSAGES.fr.replay,
    header: 'Rejeu · échantillon {count} · mémoire {used}/{capacity}',
    row: 'case {from} {glyph} case {to} · r {reward}',
    end: '· fin',
    lastStep: '· dernier pas',
    empty: "Aucun échantillon pour l'instant.",
  },
  en: {
    ariaLabel: COMMON_MESSAGES.en.replay,
    header: 'Replay · sample {count} · memory {used}/{capacity}',
    row: 'cell {from} {glyph} cell {to} · r {reward}',
    end: '· end',
    lastStep: '· last step',
    empty: 'No sample yet.',
  },
};

export function replayMessage(
  locale: Locale,
  key: ReplayMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, REPLAY_MESSAGES, key, params);
}
