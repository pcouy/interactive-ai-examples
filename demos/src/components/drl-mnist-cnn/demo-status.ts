/**
 * Logique pure du badge, de l'origine des poids et de la ligne d'état du
 * composant `drl-mnist-cnn`.
 *
 * Ces trois affichages dépendent de la machine à états `training-state.ts` et
 * de quelques compteurs : ce module les calcule à partir d'un `DemoStatusInput`,
 * sans Lit ni DOM, afin d'être testable seul. Le composant adopte ces
 * fonctions en lieu et place de ses getters `badgeText`, `weightsOrigin` et
 * `statusLine`.
 *
 * Règle du badge : « Modifié » dès que les poids ont été édités à la main et
 * qu'aucune run n'est active ; sinon le badge de l'état. L'emporte une run
 * active, car la machine à états est la source de vérité du badge pendant une
 * run et `onWeightsEdited` fait retomber l'état sur un état inactif, donc une
 * run active avec des poids édités est en pratique inatteignable.
 */

import { isTrainingActive, type TrainingState, trainingStateBadge } from './training-state.js';

/** Données d'entrée du badge et de la ligne d'état, sans dépendance au DOM. */
export interface DemoStatusInput {
  /** État de la machine à états de l'entraînement. */
  readonly state: TrainingState;
  /** Vrai si les poids affichés ont été édités à la main depuis le dernier remplacement. */
  readonly weightsEdited: boolean;
  /** Raison de l'échec du manifeste pré-entraîné, ou null. */
  readonly manifestError: string | null;
  /** Nombre de lots traités par la run courante. */
  readonly batchesProcessed: number;
  /** Dernière perte, ou null avant le premier évènement de progression. */
  readonly loss: number | null;
  /** Précision de validation sur le sous-ensemble, ou null avant le premier lot. */
  readonly validationAccuracy: number | null;
  /** Durée écoulée depuis le début de la run, en millisecondes. */
  readonly elapsedMs: number;
  /** Vrai dès le premier évènement de progression : les métriques sont affichées. */
  readonly metricsStarted: boolean;
  /** Effectif du sous-ensemble d'entraînement sélectionné. */
  readonly trainExampleCount: number;
  /** Effectif du sous-ensemble de validation sélectionné. */
  readonly validationExampleCount: number;
  /** Nombre de paramètres de l'architecture courante. */
  readonly parameters: number;
  /** Chiffre prédit, ou null tant que l'inférence n'a pas produit de résultat. */
  readonly predictedClass: number | null;
  /** Source de l'image d'entrée : « Dessin » ou « Galerie ». */
  readonly inputSource: string;
}

/** Pourcentage à une décimale, format français (virgule). */
export function formatPercent(value: number): string {
  return `${value.toFixed(1).replace('.', ',')} %`;
}

/** Perte à trois décimales, format français (virgule) ; « – » si absente ou non finie. */
function formatLoss(value: number | null): string {
  return value !== null && Number.isFinite(value) ? value.toFixed(3).replace('.', ',') : '–';
}

/** Durée en secondes à une décimale, format français (virgule). */
export function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1).replace('.', ',')} s`;
}

/**
 * Texte du badge : « Modifié » quand les poids ont été édités à la main et
 * qu'aucune run n'est active, le badge de l'état sinon.
 */
export function badgeText(input: DemoStatusInput): string {
  if (input.weightsEdited && !isTrainingActive(input.state)) {
    return 'Modifié';
  }
  return trainingStateBadge(input.state);
}

/**
 * Origine des poids affichée dans les statistiques : « Modifié » quand les
 * poids ont été édités à la main, sinon l'origine déduite de l'état.
 *
 * L'input ne porte pas le manifeste chargé : l'état l'a déjà incorporé
 * (`onManifestLoaded`, `onManifestFailed`, `onWeightsReset`), il sert donc de
 * signal pour « Pré-entraîné » ou « Déterministes ».
 */
export function weightsOriginText(input: DemoStatusInput): string {
  if (input.weightsEdited) {
    return 'Modifié';
  }
  if (input.state === 'random') {
    return 'Aléatoires';
  }
  if (
    input.metricsStarted &&
    (input.state === 'training' || input.state === 'paused' || input.state === 'trained')
  ) {
    return 'Entraîné';
  }
  return input.state === 'pretrained' ? 'Pré-entraîné' : 'Déterministes';
}

/**
 * L'état de la démonstration en une ligne : entrée et classe prédite, paramètres,
 * métriques de la run quand elle a démarré, effectifs des sous-ensembles, note
 * d'édition manuelle des poids, erreur de manifeste. Parties jointes par ` · `.
 */
export function statusLine(input: DemoStatusInput): string {
  const parts = [
    `Entrée : ${input.inputSource}`,
    `classe prédite : ${input.predictedClass === null ? '–' : String(input.predictedClass)}`,
    `${input.parameters} paramètres`,
  ];
  if (input.metricsStarted) {
    parts.push(
      `lots : ${input.batchesProcessed}`,
      `perte : ${formatLoss(input.loss)}`,
      `précision validation (sous-ensemble) : ${
        input.validationAccuracy === null || !Number.isFinite(input.validationAccuracy)
          ? '–'
          : formatPercent(input.validationAccuracy * 100)
      }`,
      `temps : ${formatSeconds(input.elapsedMs)}`,
    );
  }
  parts.push(
    `entraînement : ${input.trainExampleCount} exemples`,
    `validation : ${input.validationExampleCount} exemples`,
  );
  if (input.weightsEdited) {
    parts.push('poids modifiés à la main');
  }
  if (input.manifestError !== null) {
    parts.push(`manifeste inaccessible : ${input.manifestError}`);
  }
  return parts.join(' · ');
}
