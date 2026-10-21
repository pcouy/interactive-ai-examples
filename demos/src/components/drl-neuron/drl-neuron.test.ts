/**
 * Tests for the drl-neuron demo component.
 *
 * The real WebGL/three render and the actual DOM render are not covered: the
 * suite runs in the node environment, so the Lit render pass is neutralised
 * and the 3D scene is replaced by a fake. Only the component logic is tested.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { NeuronCore, NeuronEvaluation } from '../../core/neuron.js';
import type { LandscapeScene } from '../shared/landscape-scene.js';
import {
  dataX,
  dataY,
  gridCenter,
  isTestPointVisible,
  Neuron,
  planeToData,
  screenToViewBox,
} from './drl-neuron.js';

/**
 * The scene factory is mocked at module level so no WebGL context is ever
 * created. Only the setup guard test drives it, through a deferred promise.
 * The real helpers are kept because the component imports sanitizeHeightScale.
 */
const mocks = vi.hoisted(() => ({ createLandscapeScene: vi.fn() }));
vi.mock('../shared/landscape-scene.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../shared/landscape-scene.js')>();
  return { ...actual, createLandscapeScene: mocks.createLandscapeScene };
});

/**
 * Creates a component instance without the Lit update loop: `performUpdate`
 * is neutralised so no render pass runs, while the reactive property
 * bookkeeping stays intact.
 */
function createNeuron(): Neuron {
  const element = new Neuron();
  (element as unknown as { performUpdate(): void }).performUpdate = () => {};
  return element;
}

/** A scene double that absorbs every 3D call the component makes. */
function fakeScene(): LandscapeScene {
  return {
    scene: {} as never,
    camera: {} as never,
    renderer: {} as never,
    setHeightField: vi.fn(),
    setTestHeightField: vi.fn(),
    setActiveHeightField: vi.fn(),
    setHeightScale: vi.fn(),
    moveBall: vi.fn(),
    tweenBall: vi.fn(),
    clearTween: vi.fn(),
    heightAt: vi.fn(() => 0),
    pushTrail: vi.fn(),
    clearTrail: vi.fn(),
    update: vi.fn(),
    render: vi.fn(),
    dispose: vi.fn(),
  };
}

/** Replaces the 3D scene of a component with the fake. */
function installScene(element: Neuron, scene: LandscapeScene): void {
  (element as unknown as { scene: LandscapeScene }).scene = scene;
}

/** Reads a private `@state()` flag of the component. */
function readFlag(element: Neuron, flag: 'diverged'): boolean {
  return (element as unknown as { diverged: boolean })[flag];
}

/** Marks the element as already rendered so willUpdate passes its guard. */
function markUpdated(element: Neuron): void {
  (element as unknown as { hasUpdated: boolean }).hasUpdated = true;
}

/** Drives the protected willUpdate with a synthetic changed-property map. */
function runWillUpdate(element: Neuron, ...keys: string[]): void {
  const changed = new Map<string, unknown>(keys.map((key) => [key, undefined]));
  (element as unknown as { willUpdate(changed: Map<string, unknown>): void }).willUpdate(changed);
}

/** Reads the component core for assertions. */
function readCore(element: Neuron): NeuronCore {
  return (element as unknown as { core: NeuronCore }).core;
}

/**
 * Flattens a Lit `TemplateResult` into text. The node test bench has no DOM, so
 * the produced template is inspected instead of an actual render. Attribute
 * bindings appear as `name=value` and boolean bindings as `?name=value`.
 */
function renderTemplateTextFlat(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }
  if (Array.isArray(value)) {
    return value.map(renderTemplateTextFlat).join('');
  }
  if (typeof value === 'object' && 'strings' in value && 'values' in value) {
    const template = value as { strings: readonly string[]; values: readonly unknown[] };
    let out = '';
    for (let index = 0; index < template.strings.length; index += 1) {
      out += template.strings[index];
      if (index < template.values.length) {
        out += renderTemplateTextFlat(template.values[index]);
      }
    }
    return out.replace(/\s+/g, ' ').trim();
  }
  return String(value).replace(/\s+/g, ' ').trim();
}

describe('Neuron regenerateDataset', () => {
  beforeEach(() => {
    // sync() schedules a render through requestAnimationFrame.
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('clears the tween and moves the ball when regenerating the dataset', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const core = (element as unknown as { core: { w1: number; w2: number } }).core;

    element.regenerateDataset();

    expect(scene.clearTween).toHaveBeenCalledOnce();
    expect(scene.clearTrail).toHaveBeenCalledOnce();
    expect(scene.moveBall).toHaveBeenLastCalledWith(core.w1, core.w2);
  });

  it('reports the attempted expression when it is invalid', () => {
    const element = createNeuron();
    const details: Array<string | null> = [];
    element.addEventListener('drl-target-error', (event) => {
      details.push((event as CustomEvent<string | null>).detail);
    });
    const scene = fakeScene();
    installScene(element, scene);

    element.expression = 'x1 +';
    element.regenerateDataset();

    expect(details.at(-1)).toBe(
      'Expression invalide : « x1 + » — « x1 + x2 » utilisée par défaut.',
    );
  });
});

describe('Neuron divergence', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('emits drl-diverged once when training diverges', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const details: boolean[] = [];
    element.addEventListener('drl-diverged', (event) => {
      details.push((event as CustomEvent<boolean>).detail);
    });

    element.learningRate = 1000;
    let guard = 400;
    while (!readFlag(element, 'diverged') && guard > 0) {
      element.step();
      guard -= 1;
    }

    expect(readFlag(element, 'diverged')).toBe(true);
    expect(details).toEqual([true]);

    // The flag is latched: later steps must not dispatch a duplicate event.
    for (let extra = 0; extra < 3; extra += 1) {
      element.step();
    }
    expect(details).toEqual([true]);
  });

  it('keeps the divergence latch across play steps', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const details: boolean[] = [];
    element.addEventListener('drl-diverged', (event) => {
      details.push((event as CustomEvent<boolean>).detail);
    });

    // Play mode reaches divergence through advanceBatch, the other step() branch.
    element.running = true;
    element.learningRate = 1000;
    let guard = 400;
    while (!readFlag(element, 'diverged') && guard > 0) {
      element.step();
      guard -= 1;
    }

    expect(readFlag(element, 'diverged')).toBe(true);
    expect(details).toEqual([true]);

    for (let extra = 0; extra < 3; extra += 1) {
      element.step();
    }
    expect(details).toEqual([true]);
  });
});

describe('Neuron stop on diverge', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('routes stopOnDiverge without resetting training or rebuilding the field', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);
    const core = readCore(element);
    const reset = vi.spyOn(core, 'reset');
    const setHeightField = vi.mocked(scene.setHeightField);

    element.stopOnDiverge = false;
    runWillUpdate(element, 'stopOnDiverge');

    expect(core.stopOnDiverge).toBe(false);
    expect(reset).not.toHaveBeenCalled();
    expect(setHeightField).not.toHaveBeenCalled();

    element.stopOnDiverge = true;
    runWillUpdate(element, 'stopOnDiverge');
    expect(core.stopOnDiverge).toBe(true);
  });

  it('routes stopOnDiverge to the core on first connect', async () => {
    const element = createNeuron();
    (element as unknown as { stopOnDiverge: boolean }).stopOnDiverge = false;
    Object.defineProperty(element, 'renderRoot', {
      value: { querySelector: () => null },
      configurable: true,
    });
    vi.spyOn(element as unknown as { setupScene(): Promise<void> }, 'setupScene').mockResolvedValue(
      undefined,
    );

    await (element as unknown as { firstUpdated(): Promise<void> }).firstUpdated();

    expect(readCore(element).stopOnDiverge).toBe(false);
  });

  it('emits drl-divergence per snap-back and no drl-diverged when halting is off', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);
    const divergences: number[] = [];
    const diverged: boolean[] = [];
    element.addEventListener('drl-divergence', (event) => {
      divergences.push((event as CustomEvent<number>).detail);
    });
    element.addEventListener('drl-diverged', (event) => {
      diverged.push((event as CustomEvent<boolean>).detail);
    });

    element.stopOnDiverge = false;
    runWillUpdate(element, 'stopOnDiverge');
    element.running = true;
    element.learningRate = 1000;
    element.step();
    element.step();
    element.step();

    expect(divergences).toEqual([1, 2, 3]);
    expect(diverged).toEqual([]);
  });

  it('emits drl-diverged only when halting is on', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);
    const divergences: number[] = [];
    const diverged: boolean[] = [];
    element.addEventListener('drl-divergence', (event) => {
      divergences.push((event as CustomEvent<number>).detail);
    });
    element.addEventListener('drl-diverged', (event) => {
      diverged.push((event as CustomEvent<boolean>).detail);
    });

    element.stopOnDiverge = true;
    runWillUpdate(element, 'stopOnDiverge');
    element.running = true;
    element.learningRate = 1000;
    let guard = 400;
    while (!readFlag(element, 'diverged') && guard > 0) {
      element.step();
      guard -= 1;
    }

    expect(readFlag(element, 'diverged')).toBe(true);
    expect(diverged).toEqual([true]);
    expect(divergences).toEqual([]);
  });

  it('rebuilds the height field on a bias-enabled snap-back', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);
    const core = readCore(element);
    core.setBiasEnabled(true);
    core.reset();

    element.stopOnDiverge = false;
    runWillUpdate(element, 'stopOnDiverge');
    element.running = true;
    element.learningRate = 1000;
    let guard = 400;
    while (core.divergenceCount === 0 && guard > 0) {
      element.step();
      guard -= 1;
    }

    expect(core.divergenceCount).toBeGreaterThan(0);
    expect(vi.mocked(scene.setHeightField)).toHaveBeenCalled();
  });

  it('does not emit a spurious drl-divergence after a reset', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);
    const divergences: number[] = [];
    element.addEventListener('drl-divergence', (event) => {
      divergences.push((event as CustomEvent<number>).detail);
    });

    element.stopOnDiverge = false;
    runWillUpdate(element, 'stopOnDiverge');
    element.running = true;
    element.learningRate = 1000;
    let guard = 400;
    while (divergences.length === 0 && guard > 0) {
      element.step();
      guard -= 1;
    }
    expect(divergences.length).toBeGreaterThan(0);
    const before = [...divergences];

    // reset() zeroes the divergence count: that must not fire drl-divergence.
    (element as unknown as { onReset(): void }).onReset();

    expect(divergences).toEqual(before);
  });
});

describe('Neuron height field throttle', () => {
  beforeEach(() => {
    // sync() schedules a render through requestAnimationFrame.
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('throttles bias-only rebuilds during play and rebuilds on each release', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const core = readCore(element);
    core.setLoss('l1');
    core.setBiasEnabled(true);
    core.reset();

    element.running = true;
    element.learningRate = 0.05;
    let now = 1000;
    vi.spyOn(performance, 'now').mockImplementation(() => now);

    for (let step = 0; step < 6; step += 1) {
      element.step();
    }

    // The exact counts below pin the current RNG/batching behaviour of the core.
    // The real guard is structural: 6 frozen steps rebuild fewer than 6 times,
    // and each release (time window, big jump, forced reset) adds exactly one.
    expect(scene.setHeightField).toHaveBeenCalledOnce();

    now = 1201;
    element.step();
    expect(scene.setHeightField).toHaveBeenCalledTimes(2);

    element.learningRate = 3;
    element.step();
    // One large step at lr 3 must not diverge: divergence needs lr 1000 here.
    expect(readFlag(element, 'diverged')).toBe(false);
    expect(scene.setHeightField).toHaveBeenCalledTimes(3);

    (element as unknown as { onReset(): void }).onReset();
    expect(scene.setHeightField).toHaveBeenCalledTimes(4);
  });

  it('rebuilds the height field on every L2 play step, including a snap-back', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const core = readCore(element);
    core.setBiasEnabled(true);
    core.reset();

    element.running = true;
    element.learningRate = 0.15;
    for (let step = 0; step < 6; step += 1) {
      element.step();
    }
    // L2 keeps the surface exact, so one rebuild per play step, no throttle.
    // The exact count pins the current RNG/batching behaviour of the core.
    expect(scene.setHeightField).toHaveBeenCalledTimes(6);

    element.learningRate = 1000;
    const setHeightField = vi.mocked(scene.setHeightField);
    const before = setHeightField.mock.calls.length;
    let guard = 400;
    while (!readFlag(element, 'diverged') && guard > 0) {
      element.step();
      guard -= 1;
    }
    expect(readFlag(element, 'diverged')).toBe(true);
    // The snap-back step still rebuilds the surface at the restored weights.
    expect(setHeightField.mock.calls.length).toBeGreaterThan(before);
  });
});

describe('Neuron test-field throttle', () => {
  beforeEach(() => {
    // sync() schedules a render through requestAnimationFrame.
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it.each(['l1', 'l2'] as const)(
    'throttles bias-only test-field rebuilds within the window (loss=%s)',
    (loss) => {
      const element = createNeuron();
      const scene = fakeScene();
      installScene(element, scene);
      const core = readCore(element);
      core.setLoss(loss);
      core.setBiasEnabled(true);
      core.reset();

      let now = 1000;
      vi.spyOn(performance, 'now').mockImplementation(() => now);

      element.setTestPoint(0.5, -0.25);
      const testCalls = vi.mocked(scene.setTestHeightField).mock.calls.length;

      element.running = true;
      element.learningRate = 0.05;
      for (let step = 0; step < 6; step += 1) {
        element.step();
      }
      // The bias moves by less than 0.1 per step and the 200 ms window has not
      // elapsed, so the test field is not rebuilt.
      expect(vi.mocked(scene.setTestHeightField).mock.calls.length).toBe(testCalls);

      // The window elapses: the next step rebuilds the test field.
      now = 1201;
      element.step();
      expect(vi.mocked(scene.setTestHeightField).mock.calls.length).toBe(testCalls + 1);

      // A bias step of at least 0.1 rebuilds the test field outside the window.
      element.learningRate = 3;
      element.step();
      expect(readFlag(element, 'diverged')).toBe(false);
      expect(vi.mocked(scene.setTestHeightField).mock.calls.length).toBe(testCalls + 2);
    },
  );
});

describe('Neuron scene setup guard', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
    // The hoisted mock is shared by every test in this file and
    // vi.restoreAllMocks() does not clear the call history of a vi.fn()
    // created with vi.hoisted, so start from a clean slate.
    mocks.createLandscapeScene.mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('returns the in-flight promise while the scene is still being created', async () => {
    const element = createNeuron();
    vi.stubGlobal('HTMLCanvasElement', class FakeCanvas {});
    const canvas = new (HTMLCanvasElement as unknown as new () => HTMLCanvasElement)();
    let resolveCreate: (scene: LandscapeScene) => void = () => {};
    mocks.createLandscapeScene.mockImplementation(
      () =>
        new Promise<LandscapeScene>((resolve) => {
          resolveCreate = resolve;
        }),
    );
    Object.defineProperty(element, 'renderRoot', {
      value: {
        querySelector: (selector: string) => (selector === 'canvas' ? canvas : null),
      },
      configurable: true,
    });
    Object.defineProperty(element, 'isConnected', { value: true, configurable: true });

    const cast = element as unknown as {
      setupScene(): Promise<void>;
      scene: LandscapeScene | null;
    };
    const first = cast.setupScene();
    const second = cast.setupScene();

    // A second call during scene creation must not start a second WebGL context.
    expect(second).toBe(first);
    expect(mocks.createLandscapeScene).toHaveBeenCalledOnce();

    const scene = fakeScene();
    resolveCreate(scene);
    await first;

    expect(cast.scene).toBe(scene);
    expect(mocks.createLandscapeScene).toHaveBeenCalledOnce();
    expect(scene.setHeightScale).toHaveBeenCalledWith(1.5);
  });

  it('routes the current height scale into the scene on first connect', async () => {
    const element = createNeuron();
    (element as unknown as { heightScale: number }).heightScale = 2;
    vi.stubGlobal('HTMLCanvasElement', class FakeCanvas {});
    const canvas = new (HTMLCanvasElement as unknown as new () => HTMLCanvasElement)();
    let resolveCreate: (scene: LandscapeScene) => void = () => {};
    mocks.createLandscapeScene.mockImplementation(
      () =>
        new Promise<LandscapeScene>((resolve) => {
          resolveCreate = resolve;
        }),
    );
    Object.defineProperty(element, 'renderRoot', {
      value: {
        querySelector: (selector: string) => (selector === 'canvas' ? canvas : null),
      },
      configurable: true,
    });
    Object.defineProperty(element, 'isConnected', { value: true, configurable: true });

    const cast = element as unknown as { setupScene(): Promise<void> };
    const first = cast.setupScene();
    const second = cast.setupScene();
    expect(second).toBe(first);

    const scene = fakeScene();
    resolveCreate(scene);
    await first;

    expect(scene.setHeightScale).toHaveBeenCalledWith(2);
  });
});

describe('Neuron firstUpdated', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('syncs and draws the SVG before creating the 3D scene', async () => {
    const element = createNeuron();
    Object.defineProperty(element, 'renderRoot', {
      value: { querySelector: () => null },
      configurable: true,
    });

    const calls: string[] = [];
    let resolveSetup: () => void = () => {};
    vi.spyOn(element as unknown as { sync(): void }, 'sync').mockImplementation(() => {
      calls.push('sync');
    });
    vi.spyOn(element as unknown as { draw(): void }, 'draw').mockImplementation(() => {
      calls.push('draw');
    });
    vi.spyOn(
      element as unknown as { setupScene(): Promise<void> },
      'setupScene',
    ).mockImplementation(() => {
      calls.push('setupScene');
      return new Promise<void>((resolve) => {
        resolveSetup = resolve;
      });
    });

    const update = (element as unknown as { firstUpdated(): Promise<void> }).firstUpdated();

    // sync() and draw() run synchronously, before the awaited scene creation.
    expect(calls).toEqual(['sync', 'draw', 'setupScene']);

    resolveSetup();
    await update;
  });
});

describe('dataX/dataY mapping', () => {
  it.each([0.1, 1, 5, 20])('maps the input domain d=%s onto the plane', (d) => {
    expect(dataX(-d, d)).toBeCloseTo(40);
    expect(dataX(d, d)).toBeCloseTo(380);
    expect(dataX(0, d)).toBeCloseTo(210);
    expect(dataY(d, d)).toBeCloseTo(10);
    expect(dataY(-d, d)).toBeCloseTo(350);
    // The vertical centre is PLANE_BOTTOM - PLANE_SIZE / 2 = 180, symmetric to
    // dataX(0, d) = PLANE_X + PLANE_SIZE / 2 = 210.
    expect(dataY(0, d)).toBeCloseTo(180);
  });

  it('matches the old fixed [-1, 1] formulas at d=1', () => {
    for (const value of [-1, -0.4, 0, 0.6, 1]) {
      expect(dataX(value, 1)).toBeCloseTo(40 + ((value + 1) / 2) * 340);
      expect(dataY(value, 1)).toBeCloseTo(350 - ((value + 1) / 2) * 340);
    }
  });
});

describe('Neuron height scale', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('applies the height scale without rebuilding the height field', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);

    element.heightScale = 2;
    runWillUpdate(element, 'heightScale');

    expect(scene.setHeightScale).toHaveBeenCalledWith(2);
    expect(scene.setHeightField).not.toHaveBeenCalled();
  });

  it('falls back to the default scale when the value is not finite', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);

    element.heightScale = Number.NaN;
    runWillUpdate(element, 'heightScale');

    expect(scene.setHeightScale).toHaveBeenCalledWith(1.5);
    expect(scene.setHeightField).not.toHaveBeenCalled();
  });
});

describe('Neuron input domain', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('routes inputDomain to the core and rebuilds the field', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);

    element.inputDomain = 5;
    runWillUpdate(element, 'inputDomain');

    const core = readCore(element);
    expect(core.inputDomain).toBe(5);
    for (const example of core.dataset) {
      expect(Math.abs(example.x1)).toBeLessThanOrEqual(5);
      expect(Math.abs(example.x2)).toBeLessThanOrEqual(5);
    }
    expect(scene.setHeightField).toHaveBeenCalled();
  });

  it('keeps the input domain and the sample bounds across regenerateDataset', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    element.inputDomain = 5;

    element.regenerateDataset();

    const core = readCore(element);
    expect(core.inputDomain).toBe(5);
    for (const example of core.dataset) {
      expect(Math.abs(example.x1)).toBeLessThanOrEqual(5);
      expect(Math.abs(example.x2)).toBeLessThanOrEqual(5);
    }
  });
});

describe('Neuron weight domain', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('routes weightDomain to the core and rebuilds the field', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);

    element.weightDomain = 5;
    runWillUpdate(element, 'weightDomain');

    const core = readCore(element);
    expect(core.domain).toBe(5);
    expect(element.effectiveWeightDomain).toBe(5);
    expect(vi.mocked(scene.setHeightField)).toHaveBeenCalled();
  });

  it('restores the fitted domain when resetWeightDomain clears the override', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);
    const core = readCore(element);
    const fitted = core.domain;

    element.weightDomain = 5;
    runWillUpdate(element, 'weightDomain');
    expect(core.domain).toBe(5);

    element.resetWeightDomain();
    runWillUpdate(element, 'weightDomain');

    expect(element.weightDomain).toBe(Number.NaN);
    expect(core.domain).toBe(fitted);
  });

  it('emits drl-weight-domain on a manual write and on the restore', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);
    const core = readCore(element);
    const fitted = core.domain;
    const events: Array<{ domain: number; auto: boolean }> = [];
    element.addEventListener('drl-weight-domain', (event) => {
      events.push((event as CustomEvent<{ domain: number; auto: boolean }>).detail);
    });

    element.weightDomain = 5;
    runWillUpdate(element, 'weightDomain');
    element.resetWeightDomain();
    runWillUpdate(element, 'weightDomain');

    expect(events.at(-2)).toEqual({ domain: 5, auto: false });
    expect(events.at(-1)).toEqual({ domain: fitted, auto: true });
  });

  it('keeps a manual weight domain across a dataset change', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);
    const core = readCore(element);

    element.weightDomain = 5;
    runWillUpdate(element, 'weightDomain');
    expect(core.domain).toBe(5);

    element.regenerateDataset();
    expect(core.domain).toBe(5);
  });

  it('routes a finite weightDomain to the core on first connect', async () => {
    const element = createNeuron();
    (element as unknown as { weightDomain: number }).weightDomain = 5;
    Object.defineProperty(element, 'renderRoot', {
      value: { querySelector: () => null },
      configurable: true,
    });
    vi.spyOn(element as unknown as { setupScene(): Promise<void> }, 'setupScene').mockResolvedValue(
      undefined,
    );

    await (element as unknown as { firstUpdated(): Promise<void> }).firstUpdated();

    expect(readCore(element).domain).toBe(5);
  });
});

describe('Neuron manual weights', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('places the weights, stops play and emits drl-weights once', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const details: Array<{ w1: number; w2: number; b: number }> = [];
    element.addEventListener('drl-weights', (event) => {
      details.push((event as CustomEvent<{ w1: number; w2: number; b: number }>).detail);
    });

    element.running = true;
    (element as unknown as { tweening: boolean }).tweening = true;

    element.setWeights({ w1: 0.5, w2: -0.5 });

    expect(scene.clearTween).toHaveBeenCalledOnce();
    expect(scene.setHeightField).toHaveBeenCalled();
    expect(scene.moveBall).toHaveBeenLastCalledWith(0.5, -0.5);
    expect(element.running).toBe(false);
    expect((element as unknown as { tweening: boolean }).tweening).toBe(false);
    expect(details).toHaveLength(1);
    expect(details[0].w1).toBe(0.5);
    expect(details[0].w2).toBe(-0.5);

    // Same values: the change cache suppresses a duplicate event.
    element.setWeights({ w1: 0.5, w2: -0.5 });
    expect(details).toHaveLength(1);
  });

  it('rebuilds the height field when only the bias changes', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const core = readCore(element);
    core.setBiasEnabled(true);
    core.reset();

    element.setWeights({ b: 0.3 });

    expect(scene.setHeightField).toHaveBeenCalled();
    expect(core.b).toBe(0.3);
  });

  it('clears divergence and emits drl-diverged with false', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const diverged: boolean[] = [];
    element.addEventListener('drl-diverged', (event) => {
      diverged.push((event as CustomEvent<boolean>).detail);
    });

    element.learningRate = 1000;
    let guard = 400;
    while (!readFlag(element, 'diverged') && guard > 0) {
      element.step();
      guard -= 1;
    }
    expect(readFlag(element, 'diverged')).toBe(true);

    element.setWeights({ w1: 0.5, w2: -0.5 });

    expect(readFlag(element, 'diverged')).toBe(false);
    expect(diverged).toContain(true);
    expect(diverged.at(-1)).toBe(false);
  });
});

describe('Neuron header weight inputs', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  /** Reads the protected header template of the component. */
  function headerText(element: Neuron): string {
    return renderTemplateTextFlat(
      (element as unknown as { renderHeader(): unknown }).renderHeader(),
    );
  }

  it('renders editable weight inputs and keeps the read-only tiles', () => {
    const element = createNeuron();

    const text = headerText(element);

    // The inputs carry no value attribute: syncWeightInputs() fills them.
    expect(text).toContain('id="drl-w1"');
    expect(text).toContain('id="drl-w2"');
    expect(text).toContain('id="drl-b"');
    expect(text).not.toContain('value=');
    expect(text).toContain('Loss (moyenne)');
    expect(text).toContain('|gradient|');
    expect(text).toContain('Itérations');
  });

  it('labels the editable weight inputs', () => {
    const element = createNeuron();

    const text = headerText(element);

    expect(text).toContain('for="drl-w1"');
    expect(text).toContain('for="drl-w2"');
    expect(text).toContain('for="drl-b"');
  });

  /** Builds a fake shadow root whose inputs record the written values. */
  function fakeWeightRoot(): {
    root: {
      activeElement: { value: string } | null;
      querySelector: (selector: string) => { value: string } | null;
    };
    inputs: Record<string, { value: string }>;
  } {
    const inputs: Record<string, { value: string }> = {
      'drl-w1': { value: '' },
      'drl-w2': { value: '' },
      'drl-b': { value: '' },
    };
    const root = {
      activeElement: null,
      querySelector: (selector: string) => inputs[selector.replace('#', '')] ?? null,
    };
    return { root, inputs };
  }

  it('writes the formatted weights into the header inputs', () => {
    const element = createNeuron();
    const internals = element as unknown as { w1: number; w2: number; b: number };
    internals.w1 = 0.5;
    internals.w2 = -0.25;
    internals.b = 0.125;
    const { root, inputs } = fakeWeightRoot();
    Object.defineProperty(element, 'renderRoot', { value: root, configurable: true });

    (element as unknown as { syncWeightInputs(): void }).syncWeightInputs();

    expect(inputs['drl-w1'].value).toBe('0.5');
    expect(inputs['drl-w2'].value).toBe('-0.25');
    expect(inputs['drl-b'].value).toBe('0.125');
  });

  it('skips the focused input while syncing the other weights', () => {
    const element = createNeuron();
    const internals = element as unknown as { w1: number; w2: number; b: number };
    internals.w1 = 0.5;
    internals.w2 = -0.25;
    internals.b = 0.125;
    const { root, inputs } = fakeWeightRoot();
    const focused = inputs['drl-w1'];
    focused.value = '0,42';
    root.activeElement = focused;
    Object.defineProperty(element, 'renderRoot', { value: root, configurable: true });

    (element as unknown as { syncWeightInputs(): void }).syncWeightInputs();

    expect(focused.value).toBe('0,42');
    expect(inputs['drl-w2'].value).toBe('-0.25');
    expect(inputs['drl-b'].value).toBe('0.125');
  });

  it('marks the bias input disabled when the bias is off', () => {
    const element = createNeuron();
    (element as unknown as { bias: boolean }).bias = false;

    const text = headerText(element);

    expect(text).toContain('id="drl-b"');
    expect(text).toContain('?disabled=true');
    expect(text).toContain('Cochez « Biais » pour éditer b.');
  });

  it('commits a typed weight through handleWeightChange', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const core = readCore(element);
    const handler = (
      element as unknown as {
        handleWeightChange(field: 'w1' | 'w2' | 'b'): (event: Event) => void;
      }
    ).handleWeightChange('w1');
    const input = { value: '0.5' };

    handler({ target: input } as unknown as Event);

    expect(core.w1).toBe(0.5);
    expect(input.value).toBe('0.5');
  });

  it('restores the current formatted value on an empty entry', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const core = readCore(element);
    element.setWeights({ w1: 0.5 });
    const handler = (
      element as unknown as {
        handleWeightChange(field: 'w1' | 'w2' | 'b'): (event: Event) => void;
      }
    ).handleWeightChange('w1');
    const input = { value: '' };

    handler({ target: input } as unknown as Event);

    expect(input.value).toBe('0.5');
    expect(core.w1).toBe(0.5);
  });
});

describe('gridCenter', () => {
  it('returns the centre of the first and last cells for d=5', () => {
    expect(gridCenter(0, 5)).toBeCloseTo(-5 + 5 / 16);
    expect(gridCenter(15, 5)).toBeCloseTo(5 - 5 / 16);
  });

  it('is symmetric around zero for every cell', () => {
    for (let i = 0; i < 16; i += 1) {
      expect(gridCenter(i, 5)).toBeCloseTo(-gridCenter(15 - i, 5));
    }
  });
});

describe('axisLabel', () => {
  it('reads the current input domain in the axis caption', () => {
    const element = createNeuron();
    const label = element as unknown as { axisLabel(axis: string): string };

    expect(label.axisLabel('x₁')).toContain('1,00');

    readCore(element).setDataset({ inputDomain: 5 });
    expect(label.axisLabel('x₁')).toContain('5,00');
  });
});

describe('Neuron evaluate', () => {
  beforeEach(() => {
    // setWeights calls sync() -> requestRender() -> requestAnimationFrame.
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('delegates to the core', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);

    element.setWeights({ w1: 0.5, w2: -0.3 });

    const result = element.evaluate(0.4, -0.6);
    expect(result.predicted).toBeCloseTo(0.5 * 0.4 + -0.3 * -0.6, 12);
    expect(result.predicted).toBeCloseTo(0.38, 12);
  });

  it('follows the configured expression, loss kind and bias', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);

    element.expression = 'x1 * x2 + 1';
    element.lossKind = 'l1';
    element.bias = true;
    runWillUpdate(element, 'expression', 'lossKind', 'bias');

    element.setWeights({ w1: 0.5, w2: -0.3, b: 0.2 });

    const result = element.evaluate(0.4, -0.6);
    expect(result.target).toBeCloseTo(0.4 * -0.6 + 1, 12);
    expect(result.target).toBeCloseTo(0.76, 12);
    expect(result.predicted).toBeCloseTo(0.5 * 0.4 + -0.3 * -0.6 + 0.2, 12);
    expect(result.predicted).toBeCloseTo(0.58, 12);
    expect(result.loss).toBeCloseTo(Math.abs(result.predicted - result.target), 12);
    expect(result.loss).toBeCloseTo(0.18, 12);
  });
});

/** Reads the private `@state()` test point of the component. */
function readTestPoint(element: Neuron): { x1: number; x2: number } | null {
  return (element as unknown as { testPoint: { x1: number; x2: number } | null }).testPoint;
}

/** Writes the private `@state()` test point of the component. */
function writeTestPoint(element: Neuron, point: { x1: number; x2: number }): void {
  (element as unknown as { testPoint: { x1: number; x2: number } | null }).testPoint = point;
}

describe('Neuron test point', () => {
  beforeEach(() => {
    // setTestPoint and regenerateDataset call sync() -> requestRender().
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('builds the test field, switches the surface and dispatches drl-test-point', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const core = readCore(element);
    const events: NeuronEvaluation[] = [];
    element.addEventListener('drl-test-point', (event) => {
      events.push((event as CustomEvent<NeuronEvaluation>).detail);
    });

    element.setTestPoint(0.5, -0.25);

    expect(readTestPoint(element)).toEqual({ x1: 0.5, x2: -0.25 });
    expect(element.surface).toBe('test');
    expect(scene.setTestHeightField).toHaveBeenCalledOnce();
    expect(scene.setActiveHeightField).toHaveBeenCalledWith('test');
    expect(events).toHaveLength(1);
    expect(events[0].x1).toBe(0.5);
    expect(events[0].x2).toBe(-0.25);
    expect(events[0].predicted).toBeCloseTo(core.w1 * 0.5 + core.w2 * -0.25 + core.b, 12);
    expect(events[0].target).toBeCloseTo(0.5 + -0.25, 12);
  });

  it('treats a repeated test point as a no-op', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    const events: NeuronEvaluation[] = [];
    element.addEventListener('drl-test-point', (event) => {
      events.push((event as CustomEvent<NeuronEvaluation>).detail);
    });

    element.setTestPoint(0.5, -0.25);
    const testCalls = vi.mocked(scene.setTestHeightField).mock.calls.length;
    const activeCalls = vi.mocked(scene.setActiveHeightField).mock.calls.length;
    const globalCalls = vi.mocked(scene.setHeightField).mock.calls.length;

    element.setTestPoint(0.5, -0.25);

    expect(events).toHaveLength(1);
    expect(vi.mocked(scene.setTestHeightField).mock.calls.length).toBe(testCalls);
    expect(vi.mocked(scene.setActiveHeightField).mock.calls.length).toBe(activeCalls);
    expect(vi.mocked(scene.setHeightField).mock.calls.length).toBe(globalCalls);
  });

  it('ignores a non-finite test point', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);

    element.setTestPoint(Number.NaN, 0.5);

    expect(readTestPoint(element)).toBeNull();
    expect(scene.setTestHeightField).not.toHaveBeenCalled();
  });

  it('switches the displayed surface back to global', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    element.setTestPoint(0.5, -0.25);
    vi.mocked(scene.setActiveHeightField).mockClear();

    element.setSurface('global');

    expect(element.surface).toBe('global');
    expect(scene.setActiveHeightField).toHaveBeenCalledWith('global');
  });

  it('keeps the global surface when no test point exists', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);

    element.setSurface('test');

    expect(element.surface).toBe('global');
    expect(scene.setActiveHeightField).not.toHaveBeenCalledWith('test');
  });

  it('rebuilds the test field on a testPoint update', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    markUpdated(element);
    writeTestPoint(element, { x1: 0.5, x2: -0.25 });

    runWillUpdate(element, 'testPoint');

    expect(scene.setTestHeightField).toHaveBeenCalledOnce();
  });

  it('keeps the test point across regenerateDataset', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);
    element.setTestPoint(0.5, -0.25);

    element.regenerateDataset();

    expect(readTestPoint(element)).toEqual({ x1: 0.5, x2: -0.25 });
    expect(scene.setTestHeightField).toHaveBeenCalled();
  });

  it('keeps the test point and the test surface across reset, setWeights and step', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);

    element.setTestPoint(0.5, -0.25);

    (element as unknown as { onReset(): void }).onReset();
    expect(readTestPoint(element)).toEqual({ x1: 0.5, x2: -0.25 });
    expect(element.surface).toBe('test');
    expect(scene.setActiveHeightField).toHaveBeenCalledWith('test');

    element.setWeights({ w1: 0.5 });
    expect(readTestPoint(element)).toEqual({ x1: 0.5, x2: -0.25 });
    expect(element.surface).toBe('test');

    element.step();
    expect(readTestPoint(element)).toEqual({ x1: 0.5, x2: -0.25 });
    expect(element.surface).toBe('test');
  });

  it('sets the test surface and activates the test field when a point exists', () => {
    const element = createNeuron();
    const scene = fakeScene();
    installScene(element, scene);

    element.setTestPoint(0.5, -0.25);
    element.setSurface('global');
    vi.mocked(scene.setActiveHeightField).mockClear();

    element.setSurface('test');

    expect(element.surface).toBe('test');
    expect(scene.setActiveHeightField).toHaveBeenCalledWith('test');
  });

  it('converts a map click into a test point', () => {
    const element = createNeuron();
    installScene(element, fakeScene());
    const setTestPoint = vi.spyOn(element, 'setTestPoint');
    const target = {
      clientLeft: 0,
      clientTop: 0,
      clientWidth: 640,
      clientHeight: 360,
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 640, height: 360 }),
    };
    const event = { clientX: 210, clientY: 180, currentTarget: target } as unknown as MouseEvent;

    (element as unknown as { handleMapClick(event: MouseEvent): void }).handleMapClick(event);

    expect(setTestPoint).toHaveBeenCalledWith(0, 0);
  });

  it('ignores a click outside the plane', () => {
    const element = createNeuron();
    installScene(element, fakeScene());
    const setTestPoint = vi.spyOn(element, 'setTestPoint');
    const target = {
      clientLeft: 0,
      clientTop: 0,
      clientWidth: 640,
      clientHeight: 360,
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 640, height: 360 }),
    };
    const event = { clientX: 450, clientY: 180, currentTarget: target } as unknown as MouseEvent;

    (element as unknown as { handleMapClick(event: MouseEvent): void }).handleMapClick(event);

    expect(setTestPoint).not.toHaveBeenCalled();
  });
});

describe('isTestPointVisible', () => {
  it('shows a point inside the input square', () => {
    expect(isTestPointVisible({ x1: 0.5, x2: -0.25 }, 1)).toBe(true);
    expect(isTestPointVisible({ x1: 1, x2: -1 }, 1)).toBe(true);
  });

  it('hides a point outside the input square', () => {
    expect(isTestPointVisible({ x1: 1.5, x2: 0 }, 1)).toBe(false);
    expect(isTestPointVisible({ x1: 0, x2: -1.25 }, 1)).toBe(false);
  });

  it('hides a null point', () => {
    expect(isTestPointVisible(null, 1)).toBe(false);
  });
});

describe('screenToViewBox', () => {
  const rect = { left: 100, top: 50, width: 640, height: 360 };

  it('maps the corners onto the viewBox', () => {
    expect(screenToViewBox(100, 50, rect)).toEqual({ x: 0, y: 0 });
    expect(screenToViewBox(740, 50, rect)).toEqual({ x: 640, y: 0 });
    expect(screenToViewBox(100, 410, rect)).toEqual({ x: 0, y: 360 });
    expect(screenToViewBox(740, 410, rect)).toEqual({ x: 640, y: 360 });
  });

  it('maps the centre of the element', () => {
    expect(screenToViewBox(420, 230, rect)).toEqual({ x: 320, y: 180 });
  });

  it('maps the content box when the element has a border', () => {
    const border = { left: 1, top: 1, right: 1, bottom: 1 };
    // The rect is the border box; the content starts one pixel inside it.
    expect(screenToViewBox(101, 51, rect, border)).toEqual({ x: 0, y: 0 });
    expect(screenToViewBox(739, 409, rect, border)).toEqual({ x: 640, y: 360 });
    expect(screenToViewBox(420, 230, rect, border)).toEqual({ x: 320, y: 180 });
  });
});

describe('planeToData', () => {
  it('maps the plane corners onto the input square', () => {
    expect(planeToData(40, 10, 2)).toEqual({ x1: -2, x2: 2 });
    expect(planeToData(380, 10, 2)).toEqual({ x1: 2, x2: 2 });
    expect(planeToData(40, 350, 2)).toEqual({ x1: -2, x2: -2 });
    expect(planeToData(380, 350, 2)).toEqual({ x1: 2, x2: -2 });
  });

  it('maps the plane centre to the origin', () => {
    expect(planeToData(210, 180, 2)).toEqual({ x1: 0, x2: 0 });
  });

  it('returns null outside the plane', () => {
    expect(planeToData(39, 180, 2)).toBeNull();
    expect(planeToData(210, 9, 2)).toBeNull();
    expect(planeToData(381, 180, 2)).toBeNull();
    expect(planeToData(210, 351, 2)).toBeNull();
  });
});

describe('Neuron prediction background', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', () => 0);
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function contentText(element: Neuron): string {
    return renderTemplateTextFlat(
      (element as unknown as { renderContent(): unknown }).renderContent(),
    );
  }

  it('shows the prediction heatmap by default', () => {
    const element = createNeuron();
    expect(element.showPrediction).toBe(true);
    expect(contentText(element)).toContain('class="drl-grid" visibility=visible');
  });

  it('hides the prediction heatmap when disabled', () => {
    const element = createNeuron();
    element.showPrediction = false;
    expect(contentText(element)).toContain('class="drl-grid" visibility=hidden');
  });

  it('keeps the setting across config changes and resets', () => {
    const element = createNeuron();
    markUpdated(element);
    installScene(element, fakeScene());
    element.showPrediction = false;

    runWillUpdate(element, 'expression');
    element.regenerateDataset();
    (element as unknown as { onReset(): void }).onReset();

    expect(element.showPrediction).toBe(false);
  });
});
