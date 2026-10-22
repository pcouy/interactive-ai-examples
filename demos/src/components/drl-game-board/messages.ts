import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

export type GameBoardMessageKey = 'ariaLabel';

export const GAME_BOARD_MESSAGES: Catalog<GameBoardMessageKey> = {
  fr: {
    ariaLabel: 'Plateau de jeu {rows}×{cols}',
  },
  en: {
    ariaLabel: 'Game board {rows}×{cols}',
  },
};

export function gameBoardMessage(
  locale: Locale,
  key: GameBoardMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, GAME_BOARD_MESSAGES, key, params);
}
