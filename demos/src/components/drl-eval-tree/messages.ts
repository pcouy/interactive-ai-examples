import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

export type EvalTreeMessageKey =
  | 'root'
  | 'move'
  | 'rootCrumb'
  | 'hiddenDescendants'
  | 'unknownScore'
  | 'scoreExact'
  | 'scoreProvisional'
  | 'tableDepth'
  | 'nodes'
  | 'hidden'
  | 'cursor'
  | 'searchDone'
  | 'searchRunning'
  | 'noSearch'
  | 'toolbarAria'
  | 'stepInTitle'
  | 'stepOverTitle'
  | 'stepOutTitle'
  | 'enter'
  | 'skip'
  | 'exit'
  | 'rootNowTitle'
  | 'rootNow'
  | 'zoom'
  | 'fitTitle'
  | 'fit'
  | 'autoFitTitle'
  | 'autoFit'
  | 'zoomFullTitle'
  | 'zoomFull'
  | 'exitFullscreenTitle'
  | 'enterFullscreenTitle'
  | 'exitFullscreen'
  | 'fullscreen'
  | 'maxDepth'
  | 'infinity'
  | 'breadcrumbsAria'
  | 'alpha'
  | 'beta'
  | 'score'
  | 'exact'
  | 'provisional'
  | 'noCut'
  | 'best'
  | 'withoutTrigger'
  | 'triggeredBy'
  | 'unknownMove';

export const EVAL_TREE_MESSAGES: Catalog<EvalTreeMessageKey> = {
  fr: {
    root: 'Racine',
    move: 'Coup {move}',
    rootCrumb: 'racine',
    hiddenDescendants: '{count} descendants masqués (profondeur maximale atteinte)',
    unknownScore: 'Score inconnu',
    scoreExact: 'Score {score} (exact)',
    scoreProvisional: 'Score {score} (provisoire)',
    tableDepth: 'Profondeur table : {depth}',
    nodes: 'Nœuds : {count}',
    hidden: 'Masqués : {count}',
    cursor: 'Curseur : {cursor}',
    searchDone: 'Recherche terminée',
    searchRunning: 'Recherche en cours',
    noSearch: 'Aucune recherche liée',
    toolbarAria: "Débogueur d'arbre de recherche",
    stepInTitle: "Pas à pas entrant : descendre d'un nœud visité",
    stepOverTitle: 'Pas à pas principal : terminer le sous-arbre courant',
    stepOutTitle: 'Pas à pas sortant : remonter au parent',
    enter: 'Entrer',
    skip: 'Passer',
    exit: 'Sortir',
    rootNowTitle: 'Revenir à la racine de la recherche',
    rootNow: 'Racine = état courant',
    zoom: 'Zoom',
    fitTitle: "Ajuster le zoom à l'arbre affiché",
    fit: 'Ajuster',
    autoFitTitle:
      "Recentrer et ajuster le zoom à chaque mise à jour de l'arbre ou déplacement du point de vue",
    autoFit: 'Ajuster auto',
    zoomFullTitle: 'Zoom 100 %',
    zoomFull: '100 %',
    exitFullscreenTitle: "Quitter le mode plein écran de l'arbre",
    enterFullscreenTitle: "Afficher l'arbre et ses commandes en plein écran",
    exitFullscreen: 'Quitter le plein écran',
    fullscreen: 'Plein écran',
    maxDepth: 'Prof. max',
    infinity: '∞',
    breadcrumbsAria: "Fil d'Ariane",
    alpha: 'α',
    beta: 'β',
    score: 'Score',
    exact: '(exact)',
    provisional: '(provisoire)',
    noCut: 'Développement du nœud',
    best: 'meilleur {score}',
    withoutTrigger: 'sans coup déclencheur',
    triggeredBy: 'déclenché par le coup {move}',
    unknownMove: 'coup inconnu',
  },
  en: {
    root: 'Root',
    move: 'Move {move}',
    rootCrumb: 'root',
    hiddenDescendants: '{count} hidden descendants (maximum depth reached)',
    unknownScore: 'Unknown score',
    scoreExact: 'Score {score} (exact)',
    scoreProvisional: 'Score {score} (provisional)',
    tableDepth: 'Table depth: {depth}',
    nodes: 'Nodes: {count}',
    hidden: 'Hidden: {count}',
    cursor: 'Cursor: {cursor}',
    searchDone: 'Search complete',
    searchRunning: 'Search in progress',
    noSearch: 'No search attached',
    toolbarAria: 'Search tree debugger',
    stepInTitle: 'Step in: go down to a visited node',
    stepOverTitle: 'Step over: finish the current subtree',
    stepOutTitle: 'Step out: go back to the parent',
    enter: 'Enter',
    skip: 'Skip',
    exit: 'Exit',
    rootNowTitle: 'Return to the search root',
    rootNow: 'Root = current state',
    zoom: 'Zoom',
    fitTitle: 'Fit the zoom to the displayed tree',
    fit: 'Fit',
    autoFitTitle: 'Recenter and fit the zoom on each tree update or view move',
    autoFit: 'Fit auto',
    zoomFullTitle: 'Zoom 100%',
    zoomFull: '100%',
    exitFullscreenTitle: 'Exit the tree fullscreen mode',
    enterFullscreenTitle: 'Show the tree and its controls in fullscreen',
    exitFullscreen: 'Exit fullscreen',
    fullscreen: 'Fullscreen',
    maxDepth: 'Max depth',
    infinity: '∞',
    breadcrumbsAria: 'Breadcrumb',
    alpha: 'α',
    beta: 'β',
    score: 'Score',
    exact: '(exact)',
    provisional: '(provisional)',
    noCut: 'Node expansion',
    best: 'best {score}',
    withoutTrigger: 'without a triggering move',
    triggeredBy: 'triggered by move {move}',
    unknownMove: 'unknown move',
  },
};

export function evalTreeMessage(
  locale: Locale,
  key: EvalTreeMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, EVAL_TREE_MESSAGES, key, params);
}
