/**
 * Tests for the drl-neuron demo component.
 *
 * The real WebGL/three render and the actual DOM render are not covered: the
 * suite runs in the node environment, so the Lit render pass is neutralised
 * and the 3D scene is replaced by a fake. Only the component logic is tested.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LandscapeScene } from '../shared/landscape-scene.js';
import { Neuron } from './drl-neuron.js';

/**
 * The scene factory is mocked at module level so no WebGL context is ever
 * created. Only the setup guard test drives it, through a deferred promise.
 */
const mocks = vi.hoisted(() => ({ createLandscapeScene: vi.fn() }));
vi.mock('../shared/landscape-scene.js', () => ({
  createLandscapeScene: mocks.createLandscapeScene,
}));

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
    const core = (
      element as unknown as {
        core: { setBiasEnabled(enabled: boolean): void; reset(): void };
      }
    ).core;
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
