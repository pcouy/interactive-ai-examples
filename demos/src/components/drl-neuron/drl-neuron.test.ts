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
  });
});
