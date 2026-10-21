/**
 * Contrôleur pur du cycle de vie de l'entraînement du composant `drl-mnist-cnn`.
 *
 * Le contrôleur orchestre les runs d'entraînement (préparation, entraînement,
 * pause, annulation, fin) sans dépendre de Lit, du DOM ni de TensorFlow.js :
 * toute entrée-sortie passe par les ports injectés `MnistTrainingPorts`. Il
 * réutilise la machine à états pure de `training-state.ts` et les contrats de
 * `core/mnist-cnn-tfjs.ts` (import de type uniquement).
 *
 * Garanties :
 * - une run ne démarre que depuis un état inactif, et l'état passe à `training`
 *   avant toute attente : une assertion « run active » ne peut pas échouer au
 *   démarrage ;
 * - les poids de départ sont toujours `request.initialWeights` : le composant
 *   construit la requête depuis les poids affichés, donc chaque run repart des
 *   poids affichés, qu'ils soient pré-entraînés, aléatoires, édités à la main ou
 *   issus d'une run précédente ;
 * - les poids d'inférence sont rafraîchis depuis l'entraîneur au plus toutes
 *   les 250 ms pendant une run, et toujours à la pause, en fin de course et en
 *   cas d'annulation ;
 * - chaque entraîneur créé est disposé en fin de course, en cas d'annulation,
 *   de changement de curseur, d'édition manuelle des poids et de déconnexion ;
 * - une erreur de run est rapportée une seule fois par `onError`, sauf abandon
 *   volontaire (pause, curseur, déconnexion) qui ne rapporte rien ;
 * - un évènement tardif d'une run invalidée (identifiant de run incrémenté)
 *   est ignoré : l'état de la run plus récente est préservé.
 */

import type { MnistCnnActivation, MnistCnnFilters, MnistCnnWeights } from '../../core/mnist-cnn.js';
import type {
  MnistCnnTfjsTrainer,
  MnistCnnTrainingConfig,
  MnistTrainingProgress,
  MnistTrainSource,
} from '../../core/mnist-cnn-tfjs.js';
import type { MnistIdxDataset } from '../../core/mnist-data.js';
import type { MnistSplits } from './mnist-assets.js';
import {
  initialTrainingState,
  isTrainingActive,
  onComponentDisconnected,
  onFiltersChanged,
  onManifestLoaded,
  onPauseRequested,
  onRunCancelled,
  onRunCompleted,
  onRunStarted,
  onWeightsEdited,
  onWeightsReset,
  type TrainingState,
} from './training-state.js';

/** Intervalle minimal entre deux rafraîchissements des poids d'inférence, en ms. */
const WEIGHT_REFRESH_INTERVAL_MS = 250;

/**
 * Ports d'injection du contrôleur : chargement des jeux, création de
 * l'entraîneur et des sources, sélection stratifiée, et remontée des
 * évènements (état, progression, poids, erreur, effectifs).
 */
export interface MnistTrainingPorts {
  /** Charge les deux jeux IDX bruts (entraînement et test). */
  loadSplits(): Promise<MnistSplits>;
  /** Crée un entraîneur TensorFlow.js à partir de la configuration. */
  createTrainer(config: MnistCnnTrainingConfig): Promise<MnistCnnTfjsTrainer>;
  /** Construit une source d'entraînement sur un jeu et une liste d'indices. */
  createSource(dataset: MnistIdxDataset, indices: Uint32Array): MnistTrainSource;
  /** Sélectionne une fraction stratifiée d'un jeu, de façon déterministe. */
  selectIndices(dataset: MnistIdxDataset, fraction: number, seed: number): Uint32Array;
  /** Signale un changement d'état de l'entraînement. */
  onStateChanged(state: TrainingState): void;
  /** Signale un évènement de progression. */
  onProgress(progress: MnistTrainingProgress): void;
  /** Signale des poids d'inférence rafraîchis depuis l'entraîneur. */
  onWeightsChanged(weights: MnistCnnWeights): void;
  /**
   * Signale des poids à afficher : réinitialisation (`resetWeights`) ou
   * édition manuelle (`notifyWeightsEdited`). Le port unique évite d'ajouter
   * un gestionnaire côté composant pour un même effet.
   */
  onWeightsReset(weights: MnistCnnWeights): void;
  /** Signale une erreur d'entraînement (une seule fois par run). */
  onError(message: string): void;
  /** Signale les effectifs des sous-ensembles d'entraînement et de validation. */
  onCountsChanged(trainCount: number, validationCount: number): void;
}

/** Requête de run d'entraînement soumise à `startRun`. */
export interface MnistRunRequest {
  /** Nombre de filtres des quatre convolutions. */
  readonly filters: MnistCnnFilters;
  /** Pas d'apprentissage Adam. */
  readonly learningRate: number;
  /** Nombre d'exemples par lot. */
  readonly batchSize: number;
  /** Fonction d'activation des convolutions et de la couche dense cachée. */
  readonly activation: MnistCnnActivation;
  /** Nombre de couches de convolution, de 0 à 4. */
  readonly convLayers: number;
  /** Nombre de couches denses, sortie comprise, de 1 à 3. */
  readonly denseLayers: number;
  /**
   * Largeurs des couches denses cachées, dans l'ordre, couche de sortie exclue.
   * Défaut : `[8, 8]`.
   */
  readonly denseSizes?: readonly number[];
  /**
   * Nombre de classes de sortie, de 2 à `MAX_OUTPUT_CLASSES`. Défaut :
   * `LOGIT_COUNT` (10).
   */
  readonly outputClasses?: number;
  /** Graine d'entraînement. */
  readonly seed: number;
  /** Nombre d'époques. */
  readonly epochs: number;
  /** Pourcentage du jeu d'entraînement. */
  readonly trainPercent: number;
  /** Pourcentage du jeu de validation. */
  readonly validationPercent: number;
  /** Poids de départ : les poids affichés, quelle que soit leur origine. */
  readonly initialWeights: MnistCnnWeights;
}

/** Message d'erreur d'une exception, quelle que soit sa nature. */
function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Contrôleur du cycle de vie de l'entraînement. Sans Lit, sans DOM, sans
 * TensorFlow.js : tout passe par les ports injectés au constructeur.
 */
export class MnistTrainingController {
  private currentState: TrainingState = initialTrainingState(false);

  /** Vrai si le manifeste correspond à l'architecture courante. */
  private manifestMatches = false;

  /** Vrai après `notifyDisconnected` : toute run est abandonnée. */
  private disconnected = false;

  /** Identifiant de run : incrémenté pour invalider les rappels d'une run arrêtée. */
  private runId = 0;

  /** Entraîneur de la run courante (null hors entraînement). */
  private trainer: MnistCnnTfjsTrainer | null = null;

  /** Vrai dès le premier évènement de progression de la run. */
  private metricsStarted = false;

  /** Horodatage du dernier rafraîchissement des poids d'inférence. */
  private lastWeightsRefreshAt = 0;

  constructor(private readonly ports: MnistTrainingPorts) {}

  /** État courant de l'entraînement. */
  get state(): TrainingState {
    return this.currentState;
  }

  /** Vrai quand une run est active. */
  isRunActive(): boolean {
    return isTrainingActive(this.currentState);
  }

  /**
   * Lance un entraînement : charge les jeux IDX, sélectionne les deux
   * sous-ensembles stratifiés, crée l'entraîneur, y pousse les poids de départ
   * `request.initialWeights` et démarre la boucle par lots bornés. La préparation
   * est abandonnée après chaque `await` dès que la run n'est plus active (pause,
   * curseur changé, déconnexion).
   */
  async startRun(request: MnistRunRequest): Promise<void> {
    if (this.disconnected) {
      return;
    }
    await this.settleRun(false);
    if (this.disconnected) {
      return;
    }
    this.currentState = onRunStarted(this.currentState);
    this.ports.onStateChanged(this.currentState);
    const runId = ++this.runId;
    this.metricsStarted = false;
    let createdTrainer: MnistCnnTfjsTrainer | null = null;
    try {
      const splits = await this.ports.loadSplits();
      this.assertRunActive(runId);
      const trainIndices = this.ports.selectIndices(
        splits.train,
        request.trainPercent / 100,
        request.seed,
      );
      const validationIndices = this.ports.selectIndices(
        splits.test,
        request.validationPercent / 100,
        request.seed,
      );
      this.ports.onCountsChanged(trainIndices.length, validationIndices.length);
      const trainer = await this.ports.createTrainer({
        filters: request.filters,
        learningRate: request.learningRate,
        batchSize: request.batchSize,
        activation: request.activation,
        convLayers: request.convLayers,
        denseLayers: request.denseLayers,
        denseSizes: request.denseSizes,
        outputClasses: request.outputClasses,
        seed: request.seed,
      });
      createdTrainer = trainer;
      this.trainer = trainer;
      this.assertRunActive(runId);
      // Poids de départ : toujours ceux de la requête, construits par le
      // composant depuis les poids affichés.
      const startingWeights = request.initialWeights;
      await trainer.applyWeights(startingWeights);
      this.assertRunActive(runId);
      const run = trainer.train({
        train: this.ports.createSource(splits.train, trainIndices),
        validation: this.ports.createSource(splits.test, validationIndices),
        epochs: request.epochs,
        onProgress: (progress) => this.handleProgress(runId, progress),
      });
      const final = await run;
      if (runId !== this.runId || this.disconnected) {
        return;
      }
      this.ports.onProgress(final);
      // Rafraîchissement systématique en fin de course, quelle que soit la cadence.
      await this.refreshWeightsFromTrainer(trainer, runId);
      this.currentState = onRunCompleted(this.currentState);
      this.ports.onStateChanged(this.currentState);
      await this.settleRun(false);
    } catch (error) {
      if (runId === this.runId && !this.disconnected && isTrainingActive(this.currentState)) {
        this.ports.onError(toErrorMessage(error));
        this.currentState = this.metricsStarted
          ? onRunCancelled(this.currentState)
          : initialTrainingState(this.manifestMatches);
        this.ports.onStateChanged(this.currentState);
      }
      if (createdTrainer !== null && this.trainer === createdTrainer) {
        // Cette run a créé l'entraîneur et aucune run plus récente ne l'a
        // remplacé : il faut le disposer, même si la run a été invalidée pendant
        // la préparation.
        await this.disposeTrainer(createdTrainer);
        this.trainer = null;
      } else if (runId === this.runId) {
        await this.settleRun(false);
      }
    }
  }

  /** Pause : la boucle s'arrête avant le lot suivant et les poids sont figés. */
  async pause(): Promise<void> {
    if (!isTrainingActive(this.currentState)) {
      return;
    }
    this.currentState = onPauseRequested(this.currentState);
    this.ports.onStateChanged(this.currentState);
    await this.settleRun(true);
  }

  /**
   * Réinitialise les poids : la run courante s'arrête et les poids donnés sont
   * rapportés par `onWeightsReset`. La prochaine run repart des poids affichés,
   * donc de ces poids.
   */
  async resetWeights(
    weights: MnistCnnWeights,
    origin: 'initial' | 'random' = 'initial',
  ): Promise<void> {
    await this.settleRun(false);
    this.ports.onWeightsReset(weights);
    this.currentState = onWeightsReset(this.currentState, this.manifestMatches, origin);
    this.ports.onStateChanged(this.currentState);
  }

  /**
   * Poids édités à la main : la run courante s'arrête et est libérée sans
   * rafraîchir les poids d'inférence (les poids édités sont déjà ceux
   * affichés). Les poids édités sont rapportés par `onWeightsReset` pour être
   * affichés : la prochaine run repart des poids affichés, donc de ces poids.
   */
  async notifyWeightsEdited(weights: MnistCnnWeights): Promise<void> {
    await this.settleRun(false);
    this.ports.onWeightsReset(weights);
    this.currentState = onWeightsEdited(this.currentState, this.manifestMatches);
    this.ports.onStateChanged(this.currentState);
  }

  /**
   * Curseur changé : la run courante s'arrête et l'état retombe à l'état
   * initial selon le manifeste. L'entraînement ne peut pas continuer avec une
   * configuration obsolée. `filtersChanged` vrai arrête la run sans
   * rafraîchir les poids d'inférence : l'architecture a changé, les poids
   * exportés de l'ancien modèle ne seraient pas valides.
   */
  async notifySliderChanged(manifestMatches: boolean, filtersChanged: boolean): Promise<void> {
    this.manifestMatches = manifestMatches;
    this.currentState = onFiltersChanged(this.currentState, manifestMatches);
    this.ports.onStateChanged(this.currentState);
    await this.settleRun(!filtersChanged);
  }

  /** Composant reconnecté : une nouvelle run peut démarrer. */
  notifyConnected(): void {
    this.disconnected = false;
  }

  /** Composant déconnecté : la run en cours est arrêtée et abandonnée. */
  async notifyDisconnected(): Promise<void> {
    this.disconnected = true;
    this.currentState = onComponentDisconnected(this.currentState);
    this.ports.onStateChanged(this.currentState);
    await this.settleRun(true);
  }

  /**
   * Manifeste chargé : ne s'applique qu'avant la première run. Une run active
   * ou terminée n'est jamais dérangée par une réponse tardive.
   */
  notifyManifestChanged(manifestMatches: boolean): void {
    this.manifestMatches = manifestMatches;
    this.currentState = onManifestLoaded(this.currentState, manifestMatches);
    this.ports.onStateChanged(this.currentState);
  }

  /** Aborte la préparation si la run n'est plus active (pause, curseur, déconnexion). */
  private assertRunActive(runId: number): void {
    if (this.disconnected || runId !== this.runId || !isTrainingActive(this.currentState)) {
      throw new Error('MNIST training run aborted.');
    }
  }

  /**
   * Évènement de progression : rappelé à l'état et rafraîchit les poids
   * d'inférence au plus toutes les 250 ms. Un évènement d'une run invalidée
   * est ignoré.
   */
  private handleProgress(runId: number, progress: MnistTrainingProgress): void {
    if (runId !== this.runId) {
      return;
    }
    this.ports.onProgress(progress);
    this.metricsStarted = true;
    if (progress.status !== 'running') {
      return;
    }
    const now = Date.now();
    if (now - this.lastWeightsRefreshAt >= WEIGHT_REFRESH_INTERVAL_MS) {
      this.lastWeightsRefreshAt = now;
      const trainer = this.trainer;
      if (trainer !== null) {
        void this.refreshWeightsFromTrainer(trainer, runId);
      }
    }
  }

  /**
   * Remplace les poids d'inférence par les poids de l'entraîneur, sauf si la
   * run a été invalidée entre-temps.
   */
  private async refreshWeightsFromTrainer(
    trainer: MnistCnnTfjsTrainer,
    runId: number,
  ): Promise<void> {
    if (runId !== this.runId) {
      return;
    }
    try {
      const weights = await trainer.exportWeights();
      if (runId !== this.runId) {
        return;
      }
      this.ports.onWeightsChanged(weights);
    } catch {
      // Rafraîchissement à meilleur effort : les poids courants restent affichés.
    }
  }

  /**
   * Arrête la run courante et libère l'entraîneur. Les rappels de la run sont
   * invalidés par l'incrément de l'identifiant de run. La promesse de la run
   * n'est pas attendue : une fin ou un rejet tardif sont invalidés par
   * l'identifiant de run.
   */
  private async settleRun(refreshWeights: boolean): Promise<void> {
    this.runId += 1;
    const trainer = this.trainer;
    this.trainer = null;
    if (trainer === null) {
      return;
    }
    try {
      await trainer.cancel();
      if (refreshWeights) {
        await this.exportAndReportWeights(trainer);
      }
    } catch {
      // L'annulation et le rafraîchissement sont à meilleur effort.
    }
    await this.disposeTrainer(trainer);
  }

  /** Exporte les poids de l'entraîneur et les rapporte, à meilleur effort. */
  private async exportAndReportWeights(trainer: MnistCnnTfjsTrainer): Promise<void> {
    try {
      const weights = await trainer.exportWeights();
      this.ports.onWeightsChanged(weights);
    } catch {
      // Rafraîchissement à meilleur effort : les poids courants restent affichés.
    }
  }

  /** Libère l'entraîneur, à meilleur effort. */
  private async disposeTrainer(trainer: MnistCnnTfjsTrainer): Promise<void> {
    try {
      await trainer.dispose();
    } catch {
      // La libération est à meilleur effort.
    }
  }
}
