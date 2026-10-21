import { afterEach, describe, expect, it, vi } from 'vitest';
import type { MnistCnnFilters, MnistCnnWeights } from '../../core/mnist-cnn.js';
import type { SegmentLoader, SegmentPlan } from '../../core/mnist-cnn-segmented.js';
import type {
  MnistCnnTfjsTrainer,
  MnistCnnTrainingConfig,
  MnistTrainingProgress,
  MnistTrainingStatus,
  MnistTrainSource,
} from '../../core/mnist-cnn-tfjs.js';
import type { MnistIdxDataset, MnistSample } from '../../core/mnist-data.js';
import type { MnistSplits } from './mnist-assets.js';
import {
  type MnistRunRequest,
  type MnistSegmentedRun,
  MnistTrainingController,
  type MnistTrainingPorts,
} from './training-controller.js';
import type { TrainingState } from './training-state.js';

const FILTERS: MnistCnnFilters = [1, 1, 1, 1];

/** Jeu IDX minimal : `count` exemples, un octet par pixel, libellés cycliques. */
function makeDataset(count: number): MnistIdxDataset {
  const images = new Uint8Array(count * 784);
  const labels = new Uint8Array(count);
  for (let index = 0; index < count; index += 1) {
    labels[index] = index % 10;
  }
  return { count, images, labels };
}

/** Poids distinguables par leur étiquette (remplis avec `tag`). */
function makeWeights(tag: number): MnistCnnWeights {
  const layer = (kernelLength: number, biasLength: number) => ({
    kernel: new Float32Array(kernelLength).fill(tag),
    bias: new Float32Array(biasLength).fill(tag),
  });
  return {
    conv: [layer(9, 1), layer(9, 1), layer(9, 1), layer(9, 1)],
    dense: [layer(200, 8), layer(80, 10)],
  };
}

/** Plan segmenté minimal : aucun round matérialisé n'est demandé par les tests. */
function makeSegmentPlan(totalCount: number): SegmentPlan {
  return {
    classes: 2,
    shardSize: 10,
    totalCount,
    roundsPerEpoch: 1,
    roundsForEpoch: () => [],
  };
}

/** Chargeur segmenté minimal : aucun fragment n'est lu par les tests. */
function makeSegmentLoader(totalCount: number): SegmentLoader {
  return {
    totalCount,
    loadRound: async () => {
      throw new Error('unused segment loader');
    },
    prefetchRound: () => undefined,
    releaseRound: () => undefined,
  };
}

/** Exemple déjà matérialisé, utilisé comme jeu de validation segmenté. */
function makeSample(count: number): MnistSample {
  return { count, images: new Float32Array(count * 784), labels: new Uint8Array(count) };
}

/** Run segmentée minimale, avec validation en mémoire si `validationCount` est fourni. */
function makeSegmentedRun(trainCount: number, validationCount?: number): MnistSegmentedRun {
  return {
    trainPlan: makeSegmentPlan(trainCount),
    trainLoader: makeSegmentLoader(trainCount),
    ...(validationCount === undefined ? {} : { validation: makeSample(validationCount) }),
  };
}

/** Requête de run par défaut. */
function makeRequest(overrides: Partial<MnistRunRequest> = {}): MnistRunRequest {
  return {
    filters: FILTERS,
    learningRate: 0.003,
    batchSize: 64,
    activation: 'sigmoid',
    convLayers: 4,
    denseLayers: 2,
    seed: 42,
    epochs: 1,
    trainPercent: 10,
    validationPercent: 10,
    initialWeights: makeWeights(1),
    ...overrides,
  };
}

/** Évènement de progression minimal. */
function makeProgress(status: MnistTrainingStatus): MnistTrainingProgress {
  return {
    status,
    batchesProcessed: 1,
    loss: 0.5,
    epochLoss: 0.5,
    epochTrainAccuracy: 0.8,
    completedEpochs: 1,
    validationLoss: 0.4,
    validationAccuracy: 0.9,
    elapsedMs: 10,
  };
}

/**
 * Entraîneur factice : n'exécute aucun calcul, mais enregistre les appels et
 * permet de contrôler le déroulement de la run (fin différée, erreur, etc.).
 */
class FakeTrainer {
  appliedWeights: MnistCnnWeights[] = [];
  trainOptions: unknown[] = [];
  lastSegmentedOptions: unknown | null = null;
  cancelCount = 0;
  disposeCount = 0;
  disposed = false;
  /** Vrai pour que `train` retourne une promesse en attente. */
  holdRun = false;
  /** Vrai pour que `cancel` ne résolve pas la promesse de la run. */
  cancelResolvesRun = true;
  /** Erreur levée par `train`. */
  trainError: Error | null = null;

  private runResolve: ((progress: MnistTrainingProgress) => void) | null = null;
  private lastOnProgress: ((progress: MnistTrainingProgress) => void) | null = null;
  exportCount = 0;

  async applyWeights(weights: MnistCnnWeights): Promise<void> {
    this.appliedWeights.push(weights);
  }

  async train(options: unknown): Promise<MnistTrainingProgress> {
    this.trainOptions.push(options);
    const onProgress = (options as { onProgress?: (p: MnistTrainingProgress) => void }).onProgress;
    this.lastOnProgress = onProgress ?? null;
    if (this.trainError !== null) {
      throw this.trainError;
    }
    if (this.holdRun) {
      return new Promise<MnistTrainingProgress>((resolve) => {
        this.runResolve = resolve;
      });
    }
    return makeProgress('completed');
  }

  async trainSegmented(options: unknown): Promise<MnistTrainingProgress> {
    this.lastSegmentedOptions = options;
    if (this.trainError !== null) {
      throw this.trainError;
    }
    return makeProgress('completed');
  }

  async cancel(): Promise<void> {
    this.cancelCount += 1;
    if (this.cancelResolvesRun) {
      this.finishRun('cancelled');
    }
  }

  async exportWeights(): Promise<MnistCnnWeights> {
    this.exportCount += 1;
    return makeWeights(this.exportCount);
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    this.disposeCount += 1;
  }

  /** Émet un évènement de progression et résout la promesse de la run. */
  finishRun(status: MnistTrainingStatus = 'completed'): void {
    const progress = makeProgress(status);
    this.lastOnProgress?.(progress);
    this.runResolve?.(progress);
    this.runResolve = null;
  }

  /** Émet un évènement de progression sans résoudre la run. */
  emitProgress(status: MnistTrainingStatus = 'running'): void {
    this.lastOnProgress?.(makeProgress(status));
  }
}

/** Ports factices : aucun TensorFlow.js, aucun réseau, aucun DOM. */
class FakePorts implements MnistTrainingPorts {
  stateChanges: TrainingState[] = [];
  progresses: MnistTrainingProgress[] = [];
  weightsChanges: MnistCnnWeights[] = [];
  weightsResets: MnistCnnWeights[] = [];
  errors: string[] = [];
  countsChanges: Array<[number, number]> = [];
  selectIndicesCalls: Array<{ fraction: number; seed: number }> = [];
  trainers: FakeTrainer[] = [];
  /** Résultat de `selectIndices` (et therefore des effectifs rapportés). */
  selectIndicesResult: Uint32Array = new Uint32Array(10);
  /** Nombre d'appels à `loadSplits`. */
  loadSplitsCalls = 0;
  /** Erreur levée par `loadSplits`. */
  loadSplitsError: Error | null = null;
  /** Configuration du prochain entraîneur créé. */
  nextTrainerHoldRun = false;
  /** Configuration passée au dernier entraîneur créé. */
  lastTrainerConfig: MnistCnnTrainingConfig | null = null;
  nextTrainerCancelResolvesRun = true;
  nextTrainerTrainError: Error | null = null;
  /** Crochet appelé dans `createTrainer` (abandon pendant la préparation). */
  createTrainerHook: (() => void) | null = null;
  /** Requêtes reçues par `loadSegments`. */
  loadSegmentsCalls: MnistRunRequest[] = [];
  /** Run segmentée retournée par `loadSegments`, ou null pour le chemin actuel. */
  segmentedRun: MnistSegmentedRun | null = null;
  /** Erreur levée par `loadSegments`. */
  loadSegmentsError: Error | null = null;
  /** Porte pour suspendre `loadSegments` (abandon pendant le chargement). */
  loadSegmentsGate: Promise<void> | null = null;

  async loadSegments(request: MnistRunRequest): Promise<MnistSegmentedRun | null> {
    this.loadSegmentsCalls.push(request);
    if (this.loadSegmentsGate !== null) {
      await this.loadSegmentsGate;
    }
    if (this.loadSegmentsError !== null) {
      throw this.loadSegmentsError;
    }
    return this.segmentedRun;
  }

  async loadSplits(): Promise<MnistSplits> {
    this.loadSplitsCalls += 1;
    if (this.loadSplitsError !== null) {
      throw this.loadSplitsError;
    }
    return { train: makeDataset(100), test: makeDataset(50) };
  }

  async createTrainer(config: MnistCnnTrainingConfig): Promise<MnistCnnTfjsTrainer> {
    this.lastTrainerConfig = config;
    this.createTrainerHook?.();
    const trainer = new FakeTrainer();
    trainer.holdRun = this.nextTrainerHoldRun;
    trainer.cancelResolvesRun = this.nextTrainerCancelResolvesRun;
    trainer.trainError = this.nextTrainerTrainError;
    this.trainers.push(trainer);
    return trainer as unknown as MnistCnnTfjsTrainer;
  }

  createSource(_dataset: MnistIdxDataset, indices: Uint32Array): MnistTrainSource {
    return {
      count: indices.length,
      readBatch: (start, end) => ({
        count: end - start,
        images: new Float32Array((end - start) * 784),
        labels: new Uint8Array(end - start),
      }),
    };
  }

  selectIndices(_dataset: MnistIdxDataset, fraction: number, seed: number): Uint32Array {
    this.selectIndicesCalls.push({ fraction, seed });
    return this.selectIndicesResult;
  }

  onStateChanged(state: TrainingState): void {
    this.stateChanges.push(state);
  }

  onProgress(progress: MnistTrainingProgress): void {
    this.progresses.push(progress);
  }

  onWeightsChanged(weights: MnistCnnWeights): void {
    this.weightsChanges.push(weights);
  }

  onWeightsReset(weights: MnistCnnWeights): void {
    this.weightsResets.push(weights);
  }

  onError(message: string): void {
    this.errors.push(message);
  }

  onCountsChanged(trainCount: number, validationCount: number): void {
    this.countsChanges.push([trainCount, validationCount]);
  }
}

/** Vide la file de microtâches pour laisser `startRun` progresser. */
async function flushMicrotasks(): Promise<void> {
  for (let index = 0; index < 20; index += 1) {
    await Promise.resolve();
  }
}

/** Configure le prochain entraîneur créé. */
function configureNextTrainer(
  ports: FakePorts,
  config: Partial<
    Pick<FakePorts, 'nextTrainerHoldRun' | 'nextTrainerCancelResolvesRun' | 'nextTrainerTrainError'>
  >,
): void {
  ports.nextTrainerHoldRun = config.nextTrainerHoldRun ?? false;
  ports.nextTrainerCancelResolvesRun = config.nextTrainerCancelResolvesRun ?? true;
  ports.nextTrainerTrainError = config.nextTrainerTrainError ?? null;
}

/** Amène le contrôleur à un état inactif donné via l'API publique. */
async function driveToState(
  controller: MnistTrainingController,
  ports: FakePorts,
  state: TrainingState,
): Promise<void> {
  if (state === 'deterministic') {
    return;
  }
  if (state === 'pretrained') {
    controller.notifyManifestChanged(true);
    return;
  }
  if (state === 'paused') {
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    await controller.pause();
    await pending;
    return;
  }
  // 'trained' : une run qui se termine normalement.
  configureNextTrainer(ports, { nextTrainerHoldRun: false });
  await controller.startRun(makeRequest());
}

describe('MnistTrainingController.startRun', () => {
  const INACTIVE_STATES: readonly TrainingState[] = [
    'deterministic',
    'pretrained',
    'paused',
    'trained',
  ];

  it('reaches training from every inactive state', async () => {
    for (const initial of INACTIVE_STATES) {
      const ports = new FakePorts();
      const controller = new MnistTrainingController(ports);
      await driveToState(controller, ports, initial);
      configureNextTrainer(ports, { nextTrainerHoldRun: true });
      const pending = controller.startRun(makeRequest());
      await flushMicrotasks();
      expect(controller.state, `from ${initial}`).toBe('training');
      await controller.notifyDisconnected();
      await pending;
    }
  });

  it('applies request.initialWeights on every run', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    const first = makeRequest({ initialWeights: makeWeights(1) });
    await controller.startRun(first);
    const second = makeRequest({ initialWeights: makeWeights(2) });
    await controller.startRun(second);
    // Chaque run repart des poids de sa requête, quelle que soit leur origine.
    expect(ports.trainers[0].appliedWeights[0]).toBe(first.initialWeights);
    expect(ports.trainers[1].appliedWeights[0]).toBe(second.initialWeights);
  });

  it('forwards the requested epochs to the trainer', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest({ epochs: 7 }));
    const trainOptions = ports.trainers[0].trainOptions[0] as { epochs: number };
    expect(trainOptions.epochs).toBe(7);
  });

  it('forwards the requested activation to the trainer', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest({ activation: 'tanh' }));
    expect(ports.lastTrainerConfig?.activation).toBe('tanh');
  });

  it('forwards the sigmoid activation to the trainer', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest());
    expect(ports.lastTrainerConfig?.activation).toBe('sigmoid');
  });

  it('forwards the architecture depth to the trainer', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest({ convLayers: 2, denseLayers: 3 }));
    expect(ports.lastTrainerConfig?.convLayers).toBe(2);
    expect(ports.lastTrainerConfig?.denseLayers).toBe(3);
  });

  it('forwards the requested dense sizes to the trainer', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest({ denseSizes: [3, 7] }));
    expect(ports.lastTrainerConfig?.denseSizes).toEqual([3, 7]);
  });

  it('forwards the requested output classes to the trainer', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest({ outputClasses: 20 }));
    expect(ports.lastTrainerConfig?.outputClasses).toBe(20);
  });

  it('reports counts after selection with the requested fractions', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    const request = makeRequest({ trainPercent: 20, validationPercent: 30, seed: 7 });
    await controller.startRun(request);
    expect(ports.selectIndicesCalls.length).toBe(2);
    expect(ports.selectIndicesCalls[0].fraction).toBeCloseTo(0.2);
    expect(ports.selectIndicesCalls[1].fraction).toBeCloseTo(0.3);
    expect(ports.selectIndicesCalls[0].seed).toBe(7);
    expect(ports.selectIndicesCalls[1].seed).toBe(7);
    expect(ports.countsChanges).toEqual([[10, 10]]);
  });
});

describe('MnistTrainingController segmented runs', () => {
  it('trains with the segmented plan, reports its counts and exports the weights', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    const segmented = makeSegmentedRun(120, 30);
    ports.segmentedRun = segmented;
    await controller.startRun(makeRequest({ epochs: 3 }));
    expect(ports.loadSegmentsCalls).toHaveLength(1);
    expect(ports.loadSplitsCalls).toBe(0);
    expect(ports.selectIndicesCalls).toHaveLength(0);
    expect(ports.trainers[0].trainOptions).toHaveLength(0);
    const options = ports.trainers[0].lastSegmentedOptions as {
      trainPlan: SegmentPlan;
      trainLoader: SegmentLoader;
      validation?: MnistSample;
      epochs: number;
    };
    expect(options.trainPlan).toBe(segmented.trainPlan);
    expect(options.trainLoader).toBe(segmented.trainLoader);
    expect(options.validation).toBe(segmented.validation);
    expect(options.epochs).toBe(3);
    expect(ports.countsChanges).toEqual([[120, 30]]);
    expect(ports.trainers[0].exportCount).toBe(1);
    expect(ports.weightsChanges).toHaveLength(1);
    expect(controller.state).toBe('trained');
  });

  it('falls back to loadSplits and train when loadSegments returns null', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    ports.segmentedRun = null;
    await controller.startRun(makeRequest());
    expect(ports.loadSegmentsCalls).toHaveLength(1);
    expect(ports.selectIndicesCalls).toHaveLength(2);
    expect(ports.trainers[0].trainOptions).toHaveLength(1);
    expect(ports.trainers[0].lastSegmentedOptions).toBeNull();
    expect(ports.loadSplitsCalls).toBe(1);
  });

  it('uses request.segmented without calling the loadSegments port', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    const requested = makeSegmentedRun(84, 12);
    ports.segmentedRun = makeSegmentedRun(999);
    await controller.startRun(makeRequest({ segmented: requested }));
    expect(ports.loadSegmentsCalls).toHaveLength(0);
    expect(ports.trainers[0].lastSegmentedOptions).not.toBeNull();
    expect(ports.countsChanges).toEqual([[84, 12]]);
  });

  it('falls back to the validation plan count without an in-memory sample', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    const validationPlan = makeSegmentPlan(42);
    ports.segmentedRun = {
      ...makeSegmentedRun(120),
      validationPlan,
      validationLoader: makeSegmentLoader(42),
    };
    await controller.startRun(makeRequest());
    expect(ports.countsChanges).toEqual([[120, 42]]);
  });

  it('forwards the validation plan and loader to trainSegmented', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    const validationPlan = makeSegmentPlan(42);
    const validationLoader = makeSegmentLoader(42);
    ports.segmentedRun = {
      ...makeSegmentedRun(120),
      validationPlan,
      validationLoader,
    };
    await controller.startRun(makeRequest());
    const options = ports.trainers[0].lastSegmentedOptions as {
      validationPlan?: SegmentPlan;
      validationLoader?: SegmentLoader;
    };
    expect(options.validationPlan).toBe(validationPlan);
    expect(options.validationLoader).toBe(validationLoader);
  });

  it('does not report an error when the run is cancelled during loadSegments', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    ports.segmentedRun = makeSegmentedRun(120);
    let release = (): void => undefined;
    ports.loadSegmentsGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    await controller.pause();
    release();
    await pending;
    expect(controller.state).toBe('paused');
    expect(ports.errors).toHaveLength(0);
    expect(ports.trainers).toHaveLength(0);
    expect(ports.countsChanges).toHaveLength(0);
  });

  it('reports a loadSegments failure once and leaves an inactive state', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    ports.loadSegmentsError = new Error('segments unavailable');
    await controller.startRun(makeRequest());
    expect(ports.errors).toEqual(['segments unavailable']);
    expect(controller.state).not.toBe('training');
  });
});

describe('MnistTrainingController interruptions', () => {
  it('aborts, disposes and reports no error on slider change during preparation', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    ports.createTrainerHook = () => {
      void controller.notifySliderChanged(true, true);
    };
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(ports.trainers[0].disposed).toBe(true);
    expect(ports.errors).toHaveLength(0);
    expect(controller.state).not.toBe('training');
    await pending;
  });

  it('cancels and disposes on slider change during a run', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    await controller.notifySliderChanged(true, true);
    expect(ports.trainers[0].cancelCount).toBe(1);
    expect(ports.trainers[0].disposeCount).toBe(1);
    expect(ports.trainers[0].disposed).toBe(true);
    expect(ports.errors).toHaveLength(0);
    expect(controller.state).not.toBe('training');
    await pending;
  });

  it('a filter change reports no weights from the old architecture', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    await controller.notifySliderChanged(true, true);
    // L'architecture a changé : les poids de l'ancien modèle ne sont ni exportés ni
    // rapportés, ils ne seraient plus valides.
    expect(ports.weightsChanges).toHaveLength(0);
    expect(ports.trainers[0].exportCount).toBe(0);
    expect(ports.errors).toHaveLength(0);
    await pending;
  });

  it('aborts and disposes on disconnect during preparation', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    ports.createTrainerHook = () => {
      void controller.notifyDisconnected();
    };
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(ports.trainers[0].disposed).toBe(true);
    expect(ports.errors).toHaveLength(0);
    expect(controller.state).not.toBe('training');
    await pending;
  });

  it('cancels and disposes on disconnect during a run', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    await controller.notifyDisconnected();
    expect(ports.trainers[0].cancelCount).toBe(1);
    expect(ports.trainers[0].disposeCount).toBe(1);
    expect(ports.trainers[0].disposed).toBe(true);
    expect(ports.errors).toHaveLength(0);
    await pending;
  });

  it('pause cancels and leaves paused', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    await controller.pause();
    expect(controller.state).toBe('paused');
    expect(ports.trainers[0].cancelCount).toBe(1);
    expect(ports.trainers[0].disposeCount).toBe(1);
    expect(ports.trainers[0].disposed).toBe(true);
    expect(ports.errors).toHaveLength(0);
    await pending;
  });
});

describe('MnistTrainingController run outcomes', () => {
  it('a completed run leaves trained and reports the final progress', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest());
    expect(controller.state).toBe('trained');
    expect(ports.trainers[0].disposeCount).toBe(1);
    expect(ports.progresses.length).toBeGreaterThan(0);
    expect(ports.progresses[ports.progresses.length - 1].status).toBe('completed');
  });

  it('a run that throws reports one error and leaves a non-training state', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, { nextTrainerTrainError: new Error('boom') });
    await controller.startRun(makeRequest());
    expect(ports.errors).toEqual(['boom']);
    expect(controller.state).not.toBe('training');
  });

  it('ignores late progress and completion from a superseded run', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    // Run A : la promesse de la run reste en attente et cancel ne la résout pas.
    configureNextTrainer(ports, {
      nextTrainerHoldRun: true,
      nextTrainerCancelResolvesRun: false,
    });
    const pendingA = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    // Changement de curseur : annule A (sans résoudre sa promesse) et dispose A.
    await controller.notifySliderChanged(true, true);
    expect(controller.state).not.toBe('training');
    // Run B : reste en attente.
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pendingB = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    // Aucune run n'a émis de progression : les deux sont en attente.
    expect(ports.progresses).toHaveLength(0);
    // Fin tardive de A : la progression est ignorée (garde `runId` de
    // `handleProgress`) et l'état de B est préservé (garde `runId` de `startRun`).
    ports.trainers[0].finishRun('completed');
    await flushMicrotasks();
    expect(ports.progresses).toHaveLength(0);
    expect(controller.state).toBe('training');
    await controller.notifyDisconnected();
    await pendingA;
    await pendingB;
  });
});

describe('MnistTrainingController weights refresh', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('refreshes weights at most every 250 ms during a run', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    let now = 1000;
    vi.spyOn(Date, 'now').mockImplementation(() => now);
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    const trainer = ports.trainers[0];
    const emit = (): void => {
      trainer.emitProgress('running');
    };
    emit();
    await flushMicrotasks();
    expect(ports.weightsChanges.length).toBe(1);
    now += 100;
    emit();
    await flushMicrotasks();
    expect(ports.weightsChanges.length).toBe(1);
    now += 200;
    emit();
    await flushMicrotasks();
    expect(ports.weightsChanges.length).toBe(2);
    await controller.notifyDisconnected();
    await pending;
  });

  it('always refreshes weights on pause', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    const before = ports.weightsChanges.length;
    await controller.pause();
    expect(ports.weightsChanges.length).toBeGreaterThan(before);
    await pending;
  });

  it('always refreshes weights on completion', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest());
    expect(ports.weightsChanges.length).toBeGreaterThan(0);
  });
});

describe('MnistTrainingController reset and manifest', () => {
  it('resetWeights settles the run and reports the given weights', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    const request = makeRequest({ initialWeights: makeWeights(5) });
    await controller.startRun(request);
    expect(controller.state).toBe('trained');
    const resetWeights = makeWeights(7);
    await controller.resetWeights(resetWeights);
    expect(ports.weightsResets).toEqual([resetWeights]);
    expect(controller.state).not.toBe('trained');
  });

  it('resetWeights with a random origin reports the weights and keeps the random state', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest());
    expect(controller.state).toBe('trained');
    const randomWeights = makeWeights(8);
    await controller.resetWeights(randomWeights, 'random');
    expect(ports.weightsResets).toEqual([randomWeights]);
    expect(controller.state).toBe('random');
  });

  it('notifyManifestChanged moves to pretrained when the manifest matches', () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    expect(controller.state).toBe('deterministic');
    controller.notifyManifestChanged(true);
    expect(controller.state).toBe('pretrained');
    controller.notifyManifestChanged(false);
    expect(controller.state).toBe('deterministic');
  });
});

describe('MnistTrainingController weights edited by hand', () => {
  it('notifyWeightsEdited reports the edited weights and leaves the next run to its request weights', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    const edited = makeWeights(42);
    await controller.notifyWeightsEdited(edited);
    // La prochaine run repart des poids de sa requête, ici les poids édités.
    const request = makeRequest({ initialWeights: edited });
    await controller.startRun(request);
    expect(ports.trainers[0].appliedWeights[0]).toBe(edited);
    expect(ports.weightsResets).toEqual([edited]);
  });

  it('notifyWeightsEdited cancels, disposes and does not refresh weights during a run', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    const trainer = ports.trainers[0];
    const changesBefore = ports.weightsChanges.length;
    const edited = makeWeights(9);
    await controller.notifyWeightsEdited(edited);
    expect(trainer.cancelCount).toBe(1);
    expect(trainer.disposeCount).toBe(1);
    expect(trainer.disposed).toBe(true);
    // Aucun rafraîchissement des poids d'inférence pendant l'arrêt.
    expect(ports.weightsChanges.length).toBe(changesBefore);
    expect(trainer.exportCount).toBe(0);
    expect(ports.errors).toHaveLength(0);
    // L'édition met fin à la run : l'état n'est plus `training`.
    expect(controller.state).not.toBe('training');
    expect(controller.isRunActive()).toBe(false);
    // La prochaine run repart des poids de sa requête, ici les poids édités.
    const pendingSecond = controller.startRun(makeRequest({ initialWeights: edited }));
    await flushMicrotasks();
    expect(ports.trainers[1].appliedWeights[0]).toBe(edited);
    await controller.notifyDisconnected();
    await pendingSecond;
    await pending;
  });

  it('leaves an inactive state and no active run after an edit', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    await controller.notifyWeightsEdited(makeWeights(3));
    // L'état retombe sur l'état initial selon le manifeste, jamais `training`.
    expect(['deterministic', 'pretrained']).toContain(controller.state);
    expect(controller.isRunActive()).toBe(false);
    await pending;
  });

  it('ignores a late completion from a run cancelled by an edit', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    configureNextTrainer(ports, {
      nextTrainerHoldRun: true,
      nextTrainerCancelResolvesRun: false,
    });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    const trainer = ports.trainers[0];
    const edited = makeWeights(9);
    await controller.notifyWeightsEdited(edited);
    expect(controller.state).not.toBe('training');
    // Fin tardive de la run annulée : elle ne doit rien rapporter.
    trainer.finishRun('completed');
    await flushMicrotasks();
    expect(ports.weightsChanges).toHaveLength(0);
    expect(ports.progresses).toHaveLength(0);
    expect(controller.state).not.toBe('training');
    expect(controller.isRunActive()).toBe(false);
    // La prochaine run repart des poids de sa requête, ici les poids édités.
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pendingSecond = controller.startRun(makeRequest({ initialWeights: edited }));
    await flushMicrotasks();
    expect(ports.trainers[1].appliedWeights[0]).toBe(edited);
    await controller.notifyDisconnected();
    await pendingSecond;
    await pending;
  });
});

describe('MnistTrainingController regression tests', () => {
  it('resetWeights reports the given weights after a filter change', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.startRun(makeRequest());
    // Changement de filtres : l'ancienne run est arrêtée.
    await controller.notifySliderChanged(true, true);
    // Réinitialisation avec les poids du nouveau tuple : le contrôleur ne doit
    // pas renvoyer un champ périmé mais les poids donnés.
    const newWeights = makeWeights(9);
    await controller.resetWeights(newWeights);
    expect(ports.weightsResets).toEqual([newWeights]);
  });

  it('a non-filter slider change leaves the next run to its request weights', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    const first = makeRequest({ initialWeights: makeWeights(1) });
    await controller.startRun(first);
    // Curseur sans changement de filtres : la prochaine run repart des poids de
    // sa requête, pas des derniers poids entraînés.
    await controller.notifySliderChanged(true, false);
    const second = makeRequest({ initialWeights: makeWeights(2) });
    await controller.startRun(second);
    const secondTrainer = ports.trainers[1];
    expect(secondTrainer.appliedWeights[0]).toBe(second.initialWeights);
  });

  it('notifyConnected allows a new run after a disconnect', async () => {
    const ports = new FakePorts();
    const controller = new MnistTrainingController(ports);
    await controller.notifyDisconnected();
    // Sans notifyConnected, startRun ne démarre pas.
    const blocked = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).not.toBe('training');
    expect(ports.trainers).toHaveLength(0);
    await blocked;
    // Après notifyConnected, une run peut démarrer.
    controller.notifyConnected();
    configureNextTrainer(ports, { nextTrainerHoldRun: true });
    const pending = controller.startRun(makeRequest());
    await flushMicrotasks();
    expect(controller.state).toBe('training');
    await controller.notifyDisconnected();
    await pending;
  });
});
