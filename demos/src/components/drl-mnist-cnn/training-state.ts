/**
 * Machine à états pure de l'entraînement du composant `drl-mnist-cnn`.
 *
 * Six états :
 * - `deterministic` : poids déterministes (graine fixe), manifeste absent ou
 *   incompatible ;
 * - `pretrained` : poids pré-entraînés du manifeste, architecture
 *   correspondante ;
 * - `training` : une run est active (préparation ou entraînement) ;
 * - `paused` : une run active a été mise en pause ou annulée ;
 * - `trained` : une run s'est terminée normalement ;
 * - `random` : poids tirés au hasard, sans graine reproductible.
 *
 * Les transitions sont des fonctions pures : elles ne dépendent que de l'état
 * courant et du contexte (manifeste disponible et compatible ou non). La
 * garantie centrale : aucune transition ne laisse l'état à `training` quand
 * aucune run n'est active — toute transition qui met fin à une run (pause,
 * annulation, réinitialisation, édition manuelle des poids, changement de
 * curseur, déconnexion) fait retomber l'état sur un état inactif.
 */

import type { Locale } from '../../i18n/locale.js';
import { trainingStateMessage } from './messages.js';

export type TrainingState =
  | 'deterministic'
  | 'pretrained'
  | 'training'
  | 'paused'
  | 'trained'
  | 'random';

/** Texte du badge affiché pour chaque état. */
export function trainingStateBadge(state: TrainingState, locale: Locale = 'fr'): string {
  switch (state) {
    case 'training':
      return trainingStateMessage(locale, 'training');
    case 'paused':
      return trainingStateMessage(locale, 'paused');
    case 'trained':
      return trainingStateMessage(locale, 'trained');
    case 'pretrained':
      return trainingStateMessage(locale, 'pretrained');
    case 'random':
      return trainingStateMessage(locale, 'random');
    default:
      return trainingStateMessage(locale, 'deterministic');
  }
}

/** Vrai quand une run est active (l'entraînement tourne ou démarre). */
export function isTrainingActive(state: TrainingState): boolean {
  return state === 'training';
}

/** État initial selon que le manifeste est disponible et compatible. */
export function initialTrainingState(manifestMatches: boolean): TrainingState {
  return manifestMatches ? 'pretrained' : 'deterministic';
}

/**
 * Une run démarre : l'état passe à `training`. La ré-entrance est impossible
 * (garde `startingTraining` côté composant) : tout appel démarre une run.
 */
export function onRunStarted(_state: TrainingState): TrainingState {
  return 'training';
}

/** Un évènement de progression ne change pas l'état. */
export function onProgressReceived(state: TrainingState): TrainingState {
  return state;
}

/** Pause demandée : seule une run active devient `paused`. */
export function onPauseRequested(state: TrainingState): TrainingState {
  return state === 'training' ? 'paused' : state;
}

/** Run terminée normalement : `training` devient `trained`. */
export function onRunCompleted(state: TrainingState): TrainingState {
  return state === 'training' ? 'trained' : state;
}

/** Run annulée : `training` devient `paused`. */
export function onRunCancelled(state: TrainingState): TrainingState {
  return state === 'training' ? 'paused' : state;
}

/**
 * Poids réinitialisés : `origin` `initial` retourne à l'état initial selon le
 * manifeste, `random` retient des poids tirés au hasard.
 */
export function onWeightsReset(
  _state: TrainingState,
  manifestMatches: boolean,
  origin: 'initial' | 'random' = 'initial',
): TrainingState {
  return origin === 'random' ? 'random' : initialTrainingState(manifestMatches);
}

/**
 * Poids édités à la main : l'utilisateur fournit les poids affichés. Quel que
 * soit l'état courant, l'état retombe à l'état initial selon le manifeste :
 * une run en cours ne peut pas continuer avec des poids inconnus.
 */
export function onWeightsEdited(_state: TrainingState, manifestMatches: boolean): TrainingState {
  return initialTrainingState(manifestMatches);
}

/**
 * Filtres changés : le modèle est reconstruit, l'entraînement s'arrête et
 * l'état retombe à l'état initial selon le manifeste.
 */
export function onFiltersChanged(_state: TrainingState, manifestMatches: boolean): TrainingState {
  return initialTrainingState(manifestMatches);
}

/**
 * Manifeste chargé : ne s'applique qu'avant la première run. Une run active
 * ou terminée, ou des poids tirés au hasard, ne sont jamais dérangés par une
 * réponse tardive.
 */
export function onManifestLoaded(state: TrainingState, manifestMatches: boolean): TrainingState {
  if (state === 'training' || state === 'paused' || state === 'trained' || state === 'random') {
    return state;
  }
  return initialTrainingState(manifestMatches);
}

/**
 * Manifeste en échec : poids déterministes, sauf si une run a déjà eu lieu ou si
 * des poids ont été tirés au hasard.
 */
export function onManifestFailed(state: TrainingState): TrainingState {
  if (state === 'training' || state === 'paused' || state === 'trained' || state === 'random') {
    return state;
  }
  return 'deterministic';
}

/**
 * Composant déconnecté : la run en cours est arrêtée, l'état retombe sur un
 * état inactif (l'arrêt effectif est géré côté composant).
 */
export function onComponentDisconnected(state: TrainingState): TrainingState {
  return state === 'training' ? 'paused' : state;
}
