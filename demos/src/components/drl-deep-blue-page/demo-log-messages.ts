import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';
import { GAME_MESSAGES } from '../shared/game-messages.js';
import { DEEP_BLUE_MESSAGES } from './messages.js';

export type DemoLogMessageKey =
  | 'logMove'
  | 'logSearchStarted'
  | 'logDisplayRoot'
  | 'logDisplayedNode'
  | 'logUndone'
  | 'logGameReset'
  | 'player'
  | 'square'
  | 'column'
  | 'draw'
  | 'win'
  | 'crosses'
  | 'circles'
  | 'blues'
  | 'reds';

export const DEMO_LOG_MESSAGES: Catalog<DemoLogMessageKey> = {
  fr: {
    logMove: 'Coup {move} ({player})',
    logSearchStarted: 'Recherche lancée pour {player}',
    logDisplayRoot: 'Racine d’affichage : racine de recherche',
    logDisplayedNode: 'Nœud affiché : {key}',
    logUndone: '{count} coup(s) annulé(s)',
    logGameReset: 'Partie réinitialisée',
    player: GAME_MESSAGES.fr.player,
    square: DEEP_BLUE_MESSAGES.fr.moveCase,
    column: DEEP_BLUE_MESSAGES.fr.moveColumn,
    draw: GAME_MESSAGES.fr.draw,
    win: GAME_MESSAGES.fr.win,
    crosses: GAME_MESSAGES.fr.crosses,
    circles: GAME_MESSAGES.fr.circles,
    blues: GAME_MESSAGES.fr.blues,
    reds: GAME_MESSAGES.fr.reds,
  },
  en: {
    logMove: 'Move {move} ({player})',
    logSearchStarted: 'Search started for {player}',
    logDisplayRoot: 'Display root: search root',
    logDisplayedNode: 'Displayed node: {key}',
    logUndone: '{count} move(s) undone',
    logGameReset: 'Game reset',
    player: GAME_MESSAGES.en.player,
    square: DEEP_BLUE_MESSAGES.en.moveCase,
    column: DEEP_BLUE_MESSAGES.en.moveColumn,
    draw: GAME_MESSAGES.en.draw,
    win: GAME_MESSAGES.en.win,
    crosses: GAME_MESSAGES.en.crosses,
    circles: GAME_MESSAGES.en.circles,
    blues: GAME_MESSAGES.en.blues,
    reds: GAME_MESSAGES.en.reds,
  },
};

export function demoLogMessage(
  locale: Locale,
  key: DemoLogMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, DEMO_LOG_MESSAGES, key, params);
}
