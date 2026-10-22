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

import type { Locale } from '../../i18n/locale.js';
import { demoStatusMessage } from './messages.js';
import { formatValue } from './training-chart.js';
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
  /** Nombre d'époques achevées, ou null avant la première époque. */
  readonly completedEpochs?: number | null;
  /** Indice du round en cours dans l'époque, ou null hors run segmentée. */
  readonly roundIndex?: number | null;
  /** Nombre de rounds par époque du plan segmenté, ou null hors run segmentée. */
  readonly roundsPerEpoch?: number | null;
  /** Nombre de fragments couverts par la dernière fenêtre de perte lue. */
  readonly windowShards?: number | null;
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
export function formatPercent(value: number, locale: Locale = 'fr'): string {
  const text = value.toFixed(1);
  return locale === 'fr' ? `${text.replace('.', ',')} %` : `${text}%`;
}

/** Perte à trois décimales, format français (virgule) ; « – » si absente ou non finie. */
function formatLoss(value: number | null, locale: Locale): string {
  return value !== null && Number.isFinite(value) ? formatValue(value, 3, locale) : '–';
}

/** Durée en secondes à une décimale, format français (virgule). */
export function formatSeconds(ms: number, locale: Locale = 'fr'): string {
  const text = (ms / 1000).toFixed(1);
  return locale === 'fr' ? `${text.replace('.', ',')} s` : `${text}s`;
}

/**
 * Texte du badge : « Modifié » quand les poids ont été édités à la main et
 * qu'aucune run n'est active, le badge de l'état sinon.
 */
export function badgeText(input: DemoStatusInput, locale: Locale = 'fr'): string {
  if (input.weightsEdited && !isTrainingActive(input.state)) {
    return demoStatusMessage(locale, 'modified');
  }
  return trainingStateBadge(input.state, locale);
}

/**
 * Origine des poids affichée dans les statistiques : « Modifié » quand les
 * poids ont été édités à la main, sinon l'origine déduite de l'état.
 *
 * L'input ne porte pas le manifeste chargé : l'état l'a déjà incorporé
 * (`onManifestLoaded`, `onManifestFailed`, `onWeightsReset`), il sert donc de
 * signal pour « Pré-entraîné » ou « Déterministes ».
 */
export function weightsOriginText(input: DemoStatusInput, locale: Locale = 'fr'): string {
  if (input.weightsEdited) {
    return demoStatusMessage(locale, 'modified');
  }
  if (input.state === 'random') {
    return demoStatusMessage(locale, 'random');
  }
  if (
    input.metricsStarted &&
    (input.state === 'training' || input.state === 'paused' || input.state === 'trained')
  ) {
    return demoStatusMessage(locale, 'trained');
  }
  return input.state === 'pretrained'
    ? demoStatusMessage(locale, 'pretrained')
    : demoStatusMessage(locale, 'deterministic');
}

/**
 * L'état de la démonstration en une ligne : entrée et classe prédite, paramètres,
 * métriques de la run quand elle a démarré, époque et progression segmentée
 * quand elles sont fournies, effectifs des sous-ensembles, note d'édition
 * manuelle des poids, erreur de manifeste. Parties jointes par ` · `.
 */
export function statusLine(input: DemoStatusInput, locale: Locale = 'fr'): string {
  const parts = [
    demoStatusMessage(locale, 'input', { source: input.inputSource }),
    demoStatusMessage(locale, 'predictedClass', {
      className: input.predictedClass === null ? '–' : String(input.predictedClass),
    }),
    demoStatusMessage(locale, 'parameters', { count: input.parameters }),
  ];
  if (input.metricsStarted) {
    parts.push(
      demoStatusMessage(locale, 'batches', { count: input.batchesProcessed }),
      demoStatusMessage(locale, 'loss', { value: formatLoss(input.loss, locale) }),
      demoStatusMessage(locale, 'validationAccuracy', {
        value:
          input.validationAccuracy === null || !Number.isFinite(input.validationAccuracy)
            ? '–'
            : formatPercent(input.validationAccuracy * 100, locale),
      }),
      demoStatusMessage(locale, 'time', { value: formatSeconds(input.elapsedMs, locale) }),
    );
  }
  if (input.completedEpochs !== null && input.completedEpochs !== undefined) {
    parts.push(demoStatusMessage(locale, 'epoch', { count: input.completedEpochs }));
  }
  if (
    input.roundIndex !== null &&
    input.roundIndex !== undefined &&
    input.roundsPerEpoch !== null &&
    input.roundsPerEpoch !== undefined
  ) {
    parts.push(
      demoStatusMessage(locale, 'round', {
        index: input.roundIndex,
        total: input.roundsPerEpoch,
      }),
    );
  }
  if (input.windowShards !== null && input.windowShards !== undefined) {
    parts.push(demoStatusMessage(locale, 'window', { count: input.windowShards }));
  }
  parts.push(
    demoStatusMessage(locale, 'trainingCount', { count: input.trainExampleCount }),
    demoStatusMessage(locale, 'validationCount', { count: input.validationExampleCount }),
  );
  if (input.weightsEdited) {
    parts.push(demoStatusMessage(locale, 'weightsModified'));
  }
  if (input.manifestError !== null) {
    parts.push(demoStatusMessage(locale, 'manifestUnavailable', { message: input.manifestError }));
  }
  return parts.join(' · ');
}
