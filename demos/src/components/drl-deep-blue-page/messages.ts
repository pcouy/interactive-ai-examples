import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

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
    ticTacToe: 'Morpion',
    connect4: 'Puissance 4',
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
    apply: 'Appliquer',
    reset: 'Réinitialiser',
    rolloutSamples: 'Échantillons de rollout',
    discount: 'Actualisation (γ)',
    strictPruning: 'Élagage strict',
    strictPruningTitle:
      "Ne coupe que sur un dépassement strict de la borne (max : > β, min : < α) et départage au hasard les coups à valeur égale. Décochez pour l'alpha-bêta classique (>= / <=, premier coup retenu à égalité).",
  },
  en: {
    title: 'Deep Blue · adversarial search',
    game: 'Game',
    gameAriaLabel: 'Game choice',
    ticTacToe: 'Tic-tac-toe',
    connect4: 'Connect Four',
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
    apply: 'Apply',
    reset: 'Reset',
    rolloutSamples: 'Rollout samples',
    discount: 'Discount (γ)',
    strictPruning: 'Strict pruning',
    strictPruningTitle:
      'Cuts only on a strict bound overrun (max: > β, min: < α) and breaks ties between equal-valued moves at random. Clear the box for classic alpha-beta (>= / <=, first move kept on a tie).',
  },
};

export function deepBlueMessage(
  locale: Locale,
  key: DeepBlueMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, DEEP_BLUE_MESSAGES, key, params);
}
