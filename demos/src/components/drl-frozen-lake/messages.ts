import type { FineStepPhase, FrozenLakeAction } from '../../core/frozen-lake.js';
import { COMMON_MESSAGES } from '../../i18n/common-messages.js';
import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

export type FrozenLakeMessageKey =
  | 'title'
  | 'lead'
  | 'controlsAria'
  | 'size4'
  | 'size6'
  | 'size8'
  | 'replayToggle'
  | 'gridAria'
  | 'curveAria'
  | 'loopTitle'
  | 'loopDescription'
  | 'environment'
  | 'agent'
  | 'observation'
  | 'nextObservation'
  | 'reward'
  | 'action'
  | 'loopObservationLink'
  | 'loopActionLink'
  | 'loopFeedbackLink'
  | 'phasesAria'
  | 'phaseObserve'
  | 'phaseAct'
  | 'phaseTransition'
  | 'fineStepTitle'
  | 'nextPhase'
  | 'updateTitle'
  | 'updatePlaceholder'
  | 'gamma'
  | 'nextValue'
  | 'bootstrap'
  | 'target'
  | 'previous'
  | 'alpha'
  | 'updated'
  | 'updateNote'
  | 'status'
  | 'caseLabel'
  | 'actionRight'
  | 'actionLeft'
  | 'actionUp'
  | 'actionDown'
  | 'emptyValue'
  | 'bellmanFormula'
  | 'targetFormula'
  | 'updateFormula';

export const FROZEN_LAKE_MESSAGES: Catalog<FrozenLakeMessageKey> = {
  fr: {
    title: 'Frozen Lake — Q-learning tabulaire',
    lead: 'Un agent apprend à traverser un lac gelé sans tomber dans un trou. Il découvre la valeur de chaque paire case et déplacement par essais et erreurs. Epsilon choisit parfois un déplacement au hasard pour explorer. Les flèches montrent le meilleur déplacement pour chaque case.',
    controlsAria: 'Contrôles de Frozen Lake',
    size4: '4×4',
    size6: '6×6',
    size8: '8×8',
    replayToggle: COMMON_MESSAGES.fr.replay,
    gridAria: 'Grille Frozen Lake',
    curveAria: 'Courbe des récompenses',
    loopTitle: 'Boucle agent-environnement',
    loopDescription:
      "L'environnement émet une observation. L'agent choisit une action. L'environnement renvoie une observation et une récompense.",
    environment: 'Environnement',
    agent: 'Agent',
    observation: 'Observation s',
    nextObservation: "Observation s'",
    reward: 'Récompense r',
    action: 'Action a',
    loopObservationLink: 'observation s : Environnement → Agent',
    loopActionLink: 'action a : Agent → Environnement',
    loopFeedbackLink: "observation s' et récompense r : Environnement → Agent",
    phasesAria: 'Phases du pas',
    phaseObserve: 'Observation',
    phaseAct: 'Action',
    phaseTransition: 'Transition',
    fineStepTitle: "Avance d'une phase et met la lecture automatique en pause.",
    nextPhase: 'Phase suivante',
    updateTitle: 'Dernière mise à jour',
    updatePlaceholder: "Aucune mise à jour pour l'instant.",
    gamma: 'Facteur γ',
    nextValue: "Valeur suivante max Q(s', a')",
    bootstrap: "Amorçage γ · max Q(s', a')",
    target: 'Cible',
    previous: 'Valeur précédente Q(s, a)',
    alpha: 'Taux α',
    updated: 'Valeur mise à jour',
    updateNote:
      "La cible utilise la valeur de l'état suivant. La mise à jour écrit la nouvelle valeur dans la table apprise.",
    status: 'Épisodes {episodes} · Réussite {success} % · Pas {steps}',
    caseLabel: 'Case {observation} · {action}',
    actionRight: 'droite',
    actionLeft: 'gauche',
    actionUp: 'haut',
    actionDown: 'bas',
    emptyValue: '—',
    bellmanFormula: "Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') − Q(s, a) ]",
    targetFormula: "r + γ · max Q(s', a') = {reward} + {gamma} × {nextValue} = {target}",
    updateFormula: 'Q(s, a) ← {previous} + {alpha} × ({target} − {previous}) = {updated}',
  },
  en: {
    title: 'Frozen Lake — tabular Q-learning',
    lead: 'An agent learns to cross a frozen lake without falling into a hole. It discovers the value of each cell and move pair by trial and error. Epsilon sometimes chooses a random move to explore. The arrows show the best move for each cell.',
    controlsAria: 'Frozen Lake controls',
    size4: '4×4',
    size6: '6×6',
    size8: '8×8',
    replayToggle: COMMON_MESSAGES.en.replay,
    gridAria: 'Frozen Lake grid',
    curveAria: 'Reward curve',
    loopTitle: 'Agent-environment loop',
    loopDescription:
      'The environment emits an observation. The agent chooses an action. The environment returns an observation and a reward.',
    environment: 'Environment',
    agent: 'Agent',
    observation: 'Observation s',
    nextObservation: "Observation s'",
    reward: 'Reward r',
    action: 'Action a',
    loopObservationLink: 'observation s: Environment → Agent',
    loopActionLink: 'action a: Agent → Environment',
    loopFeedbackLink: "observation s' and reward r: Environment → Agent",
    phasesAria: 'Step phases',
    phaseObserve: 'Observation',
    phaseAct: 'Action',
    phaseTransition: 'Transition',
    fineStepTitle: 'Advances one phase and pauses autoplay.',
    nextPhase: 'Next phase',
    updateTitle: 'Latest update',
    updatePlaceholder: 'No update yet.',
    gamma: 'Factor γ',
    nextValue: "Max next value Q(s', a')",
    bootstrap: "Bootstrapping γ · max Q(s', a')",
    target: 'Target',
    previous: 'Previous value Q(s, a)',
    alpha: 'Rate α',
    updated: 'Updated value',
    updateNote:
      'The target uses the value of the next state. The update writes the new value into the learned table.',
    status: 'Episodes {episodes} · Success {success} % · Steps {steps}',
    caseLabel: 'Cell {observation} · {action}',
    actionRight: 'right',
    actionLeft: 'left',
    actionUp: 'up',
    actionDown: 'down',
    emptyValue: '—',
    bellmanFormula: "Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') − Q(s, a) ]",
    targetFormula: "r + γ · max Q(s', a') = {reward} + {gamma} × {nextValue} = {target}",
    updateFormula: 'Q(s, a) ← {previous} + {alpha} × ({target} − {previous}) = {updated}',
  },
};

const ACTION_KEYS: Record<FrozenLakeAction, FrozenLakeMessageKey> = {
  right: 'actionRight',
  left: 'actionLeft',
  up: 'actionUp',
  down: 'actionDown',
};

const PHASE_KEYS: Record<FineStepPhase, FrozenLakeMessageKey> = {
  observe: 'phaseObserve',
  act: 'phaseAct',
  transition: 'phaseTransition',
};

export function frozenLakeMessage(
  locale: Locale,
  key: FrozenLakeMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, FROZEN_LAKE_MESSAGES, key, params);
}

/** Localized display name of an action. */
export function frozenLakeActionName(locale: Locale, action: FrozenLakeAction): string {
  return frozenLakeMessage(locale, ACTION_KEYS[action]);
}

/** Localized label of a fine-step phase. */
export function frozenLakePhaseLabel(locale: Locale, phase: FineStepPhase): string {
  return frozenLakeMessage(locale, PHASE_KEYS[phase]);
}
