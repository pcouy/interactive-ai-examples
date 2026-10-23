import { COMMON_MESSAGES } from '../../i18n/common-messages.js';
import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';
import { GAME_MESSAGES } from '../shared/game-messages.js';

export type DeepBlueMessageKey =
  | 'title'
  | 'game'
  | 'gameAriaLabel'
  | 'ticTacToe'
  | 'connect4'
  | 'treeHint'
  | 'replayTitle'
  | 'replay'
  | 'modeHumanAgent'
  | 'modeAgentHuman'
  | 'modeAgentAgent'
  | 'moveCase'
  | 'moveColumn'
  | 'depth'
  | 'evaluator'
  | 'pruning'
  | 'transposition'
  | 'customEvaluator'
  | 'customHelpReceives'
  | 'customHelpAnd'
  | 'customHelpFinite'
  | 'customHelpExposes'
  | 'customHelpThen'
  | 'customHelpTicTacToe'
  | 'customHelpConnect4'
  | 'customWarning'
  | 'apply'
  | 'reset'
  | 'rolloutSamples'
  | 'discount'
  | 'strictPruning'
  | 'strictPruningTitle';

export const DEEP_BLUE_MESSAGES: Catalog<DeepBlueMessageKey> = {
  fr: {
    title: 'Deep Blue · recherche adversariale',
    game: 'Jeu',
    gameAriaLabel: 'Choix du jeu',
    ticTacToe: GAME_MESSAGES.fr.ticTacToeTitle,
    connect4: GAME_MESSAGES.fr.connect4Title,
    treeHint: "Arbre d'évaluation · la recherche de l'agent se rejoue pas à pas",
    replayTitle: 'Applique dans la partie le premier coup du chemin menant au nœud sélectionné',
    replay: 'Rejouer ce coup ({move})',
    modeHumanAgent: 'Humain vs Agent',
    modeAgentHuman: 'Agent vs Humain',
    modeAgentAgent: 'Agent vs Agent',
    moveCase: 'case {number}',
    moveColumn: 'colonne {number}',
    depth: 'Profondeur',
    evaluator: 'Évaluateur',
    pruning: 'Élagage',
    transposition: 'Transposition',
    customEvaluator: 'Évaluateur personnalisé (JavaScript)',
    customHelpReceives: 'Le corps reçoit ',
    customHelpAnd: ' et ',
    customHelpFinite: ', et doit retourner un nombre fini. ',
    customHelpExposes: ' expose ',
    customHelpThen: ', puis ',
    customHelpTicTacToe: ' (morpion) ou ',
    customHelpConnect4: ' (puissance 4).',
    customWarning:
      'Code exécuté dans la page, sans isolation. À utiliser uniquement en local avec du code de confiance.',
    apply: COMMON_MESSAGES.fr.apply,
    reset: COMMON_MESSAGES.fr.reset,
    rolloutSamples: GAME_MESSAGES.fr.rolloutSamples,
    discount: GAME_MESSAGES.fr.discount,
    strictPruning: GAME_MESSAGES.fr.strictPruning,
    strictPruningTitle: GAME_MESSAGES.fr.strictPruningTitle,
  },
  en: {
    title: 'Deep Blue · adversarial search',
    game: 'Game',
    gameAriaLabel: 'Game choice',
    ticTacToe: GAME_MESSAGES.en.ticTacToeTitle,
    connect4: GAME_MESSAGES.en.connect4Title,
    treeHint: "Evaluation tree · the agent's search replays step by step",
    replayTitle: 'Applies to the game the first move of the path to the selected node',
    replay: 'Replay this move ({move})',
    modeHumanAgent: 'Human vs Agent',
    modeAgentHuman: 'Agent vs Human',
    modeAgentAgent: 'Agent vs Agent',
    moveCase: 'square {number}',
    moveColumn: 'column {number}',
    depth: 'Depth',
    evaluator: 'Evaluator',
    pruning: 'Pruning',
    transposition: 'Transposition',
    customEvaluator: 'Custom evaluator (JavaScript)',
    customHelpReceives: 'The body receives ',
    customHelpAnd: ' and ',
    customHelpFinite: ', and must return a finite number. ',
    customHelpExposes: ' exposes ',
    customHelpThen: ', then ',
    customHelpTicTacToe: ' (tic-tac-toe) or ',
    customHelpConnect4: ' (Connect Four).',
    customWarning:
      'Code runs in the page without isolation. Use it only locally with trusted code.',
    apply: COMMON_MESSAGES.en.apply,
    reset: COMMON_MESSAGES.en.reset,
    rolloutSamples: GAME_MESSAGES.en.rolloutSamples,
    discount: GAME_MESSAGES.en.discount,
    strictPruning: GAME_MESSAGES.en.strictPruning,
    strictPruningTitle: GAME_MESSAGES.en.strictPruningTitle,
  },
};

export function deepBlueMessage(
  locale: Locale,
  key: DeepBlueMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, DEEP_BLUE_MESSAGES, key, params);
}
