import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

export type GameMessageKey =
  | 'ticTacToeTitle'
  | 'connect4Title'
  | 'crosses'
  | 'circles'
  | 'blues'
  | 'reds'
  | 'toMove'
  | 'undo'
  | 'clearTt'
  | 'browseTt'
  | 'visitedStateTitle'
  | 'noMatchingEntries'
  | 'entriesRange'
  | 'filteredSuffix'
  | 'ttTitle'
  | 'close'
  | 'entrySingular'
  | 'entryPlural'
  | 'ttSummary'
  | 'filterByKey'
  | 'filterPlaceholder'
  | 'tableEntriesAria'
  | 'position'
  | 'key'
  | 'score'
  | 'effDepth'
  | 'sortByScore'
  | 'sortByDepth'
  | 'samples'
  | 'sortBySamples'
  | 'exact'
  | 'sortByExact'
  | 'turn'
  | 'sortByTurn'
  | 'pieces'
  | 'sortByPieces'
  | 'yes'
  | 'no'
  | 'previous'
  | 'next'
  | 'page'
  | 'discount'
  | 'strictPruning'
  | 'strictPruningTitle'
  | 'draw'
  | 'win'
  | 'player'
  | 'searchDone'
  | 'searchRunning'
  | 'searchStatus'
  | 'ttCount'
  | 'seriesTitle'
  | 'series'
  | 'autoRestart'
  | 'stepByMoveTitle'
  | 'stepByMove'
  | 'illegalMove'
  | 'rolloutSamples';

export const GAME_MESSAGES: Catalog<GameMessageKey> = {
  fr: {
    ticTacToeTitle: 'Morpion',
    connect4Title: 'Puissance 4',
    crosses: 'Croix',
    circles: 'Cercles',
    blues: 'Bleus',
    reds: 'Rouges',
    toMove: 'Au trait : {player}',
    undo: 'Annuler le coup',
    clearTt: 'Vider la TT',
    browseTt: 'Parcourir la TT',
    visitedStateTitle: 'État non visité par cette partie',
    noMatchingEntries: 'Aucune entrée correspondante.',
    entriesRange: 'Entrées {start}–{end} sur {total}',
    filteredSuffix: ' (filtrées sur {total})',
    ttTitle: 'Table de transposition',
    close: 'Fermer',
    entrySingular: 'entrée',
    entryPlural: 'entrées',
    ttSummary:
      "{count} {entry} · tour courant : {turn} · un nœud déjà développé plus profondément (profondeur certifiée stockée strictement supérieure à la profondeur requise) est réutilisé, y compris depuis un tour antérieur. À profondeur égale, le nœud n'est réutilisable que s'il porte le tour courant, sinon il est redéveloppé ; seule exception, une feuille rollout rencontrée à profondeur 0 voit sa moyenne stockée étendue de nouveaux échantillons (samples). L'option probeTurns: 'previous' existe pour refuser en plus les entrées du tour courant.",
    filterByKey: 'Filtrer par clé',
    filterPlaceholder: 'ex. : 001200…',
    tableEntriesAria: 'Entrées de la table',
    position: 'Position',
    key: 'Clé',
    score: 'Score',
    effDepth: 'Profondeur eff.',
    sortByScore: 'Trier par score',
    sortByDepth: 'Trier par profondeur effective',
    samples: 'Éch.',
    sortBySamples: 'Trier par échantillons',
    exact: 'Exact',
    sortByExact: 'Trier par exactitude',
    turn: 'Tour',
    sortByTurn: 'Trier par tour de recherche',
    pieces: 'Pièces',
    sortByPieces: 'Tour intra-partie : nombre de pièces sur le plateau. Cliquer pour trier.',
    yes: 'oui',
    no: 'non',
    previous: 'Précédent',
    next: 'Suivant',
    page: 'Page {page} / {count}',
    discount: 'Actualisation (γ)',
    strictPruning: 'Élagage strict',
    strictPruningTitle:
      "Ne coupe que sur un dépassement strict de la borne (max : > β, min : < α) et départage au hasard les coups à valeur égale. Décochez pour l'alpha-bêta classique (>= / <=, premier coup retenu à égalité).",
    draw: 'Partie nulle',
    win: 'Victoire : {player}',
    player: 'Joueur {number}',
    searchDone: 'Recherche terminée',
    searchRunning: 'Recherche en cours',
    searchStatus:
      '{label} · {nodes} nœuds · {cutoffs} coupures · {hits} touches TT · {steps} étapes · profondeur effective : {depth}',
    ttCount: 'TT : {count} {entry}',
    seriesTitle:
      'Total des parties terminées de la série : victoires de chaque adversaire et parties nulles.',
    series: 'Série — {first} {firstWins} · {second} {secondWins} · Nulles {draws}',
    autoRestart: 'Relancer automatiquement',
    stepByMoveTitle: "Jouer exactement un coup de l'agent, recherche complète incluse",
    stepByMove: 'Pas à pas par coup',
    illegalMove: 'Coup non jouable (ligne {row}, colonne {col})',
    rolloutSamples: 'Échantillons de rollout',
  },
  en: {
    ticTacToeTitle: 'Tic-tac-toe',
    connect4Title: 'Connect Four',
    crosses: 'Crosses',
    circles: 'Circles',
    blues: 'Blues',
    reds: 'Reds',
    toMove: 'To move: {player}',
    undo: 'Undo move',
    clearTt: 'Clear the TT',
    browseTt: 'Browse the TT',
    visitedStateTitle: 'State not visited by this game',
    noMatchingEntries: 'No matching entries.',
    entriesRange: 'Entries {start}–{end} of {total}',
    filteredSuffix: ' (filtered from {total})',
    ttTitle: 'Transposition table',
    close: 'Close',
    entrySingular: 'entry',
    entryPlural: 'entries',
    ttSummary:
      "{count} {entry} · current turn: {turn} · a node already expanded deeper (certified depth stored strictly greater than the required depth) is reused, including from an earlier turn. At equal depth, the node is reusable only if it carries the current turn, otherwise it is re-expanded; the only exception is a rollout leaf met at depth 0 whose stored average is extended with new samples. The option probeTurns: 'previous' also rejects the current turn's entries.",
    filterByKey: 'Filter by key',
    filterPlaceholder: 'e.g.: 001200…',
    tableEntriesAria: 'Table entries',
    position: 'Position',
    key: 'Key',
    score: 'Score',
    effDepth: 'Eff. depth',
    sortByScore: 'Sort by score',
    sortByDepth: 'Sort by effective depth',
    samples: 'Samples',
    sortBySamples: 'Sort by samples',
    exact: 'Exact',
    sortByExact: 'Sort by exactness',
    turn: 'Turn',
    sortByTurn: 'Sort by search turn',
    pieces: 'Pieces',
    sortByPieces: 'In-game turn: number of pieces on the board. Click to sort.',
    yes: 'yes',
    no: 'no',
    previous: 'Previous',
    next: 'Next',
    page: 'Page {page} / {count}',
    discount: 'Discount (γ)',
    strictPruning: 'Strict pruning',
    strictPruningTitle:
      'Cuts only on a strict bound overrun (max: > β, min: < α) and breaks ties between equal-valued moves at random. Clear the box for classic alpha-beta (>= / <=, first move kept on a tie).',
    draw: 'Draw',
    win: 'Win: {player}',
    player: 'Player {number}',
    searchDone: 'Search complete',
    searchRunning: 'Search in progress',
    searchStatus:
      '{label} · {nodes} nodes · {cutoffs} cutoffs · {hits} TT hits · {steps} steps · effective depth: {depth}',
    ttCount: 'TT: {count} {entry}',
    seriesTitle: "Total finished games in the series: each player's wins and draws.",
    series: 'Series — {first} {firstWins} · {second} {secondWins} · Draws {draws}',
    autoRestart: 'Restart automatically',
    stepByMoveTitle: 'Play exactly one agent move, including the full search',
    stepByMove: 'Step by move',
    illegalMove: 'Illegal move (row {row}, column {col})',
    rolloutSamples: 'Rollout samples',
  },
};

export function gameMessage(locale: Locale, key: GameMessageKey, params?: TranslateParams): string {
  return translate(locale, GAME_MESSAGES, key, params);
}
