import { drlTokenValues } from '../../styles/token-values.js';

type ThreeModule = typeof import('three');
type ThreeMesh = InstanceType<ThreeModule['Mesh']>;
type ThreeColor = InstanceType<ThreeModule['Color']>;
type ThreeBufferGeometry = InstanceType<ThreeModule['BufferGeometry']>;
type ThreeBufferAttribute = InstanceType<ThreeModule['BufferAttribute']>;
type ThreeMaterial = InstanceType<ThreeModule['Material']>;
type ThreeLineBasicMaterial = InstanceType<ThreeModule['LineBasicMaterial']>;

/**
 * Vertical exaggeration of the landscape: scene Y = normalized height × HEIGHT_SCALE.
 */
const HEIGHT_SCALE = 1.5;
const LANDSCAPE_RES = 70;

/** Camera orbit defaults: slightly top-down view of the valley. */
const CAMERA_THETA = Math.PI * 0.28;
const CAMERA_PHI = Math.PI * 0.32;

const BALL_RADIUS = 0.14;

const TRAIL_MAX = 60;
const TRAIL_LIFT = 0.05;

const TWEEN_MS = 600;

/** Landscape extent used before the first setHeightField call. */
const DEFAULT_DOMAIN = 5;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Reads a CSS custom property off the canvas. The canvas inherits the host
 * custom properties through the shadow boundary, so the value resolves even
 * though this module is framework-agnostic. Falls back to a token value.
 */
function readColor(
  three: ThreeModule,
  canvas: HTMLCanvasElement,
  name: string,
  fallback: string,
): ThreeColor {
  const value = getComputedStyle(canvas).getPropertyValue(name).trim();
  return new three.Color(value.length > 0 ? value : fallback);
}

/** Triangle index buffer for a (res+1)² vertex grid, exported for testing. */
export function gridIndex(res: number): number[] {
  const indices: number[] = [];
  for (let j = 0; j < res; j += 1) {
    for (let i = 0; i < res; i += 1) {
      const a = j * (res + 1) + i;
      const b = a + 1;
      const c = a + (res + 1);
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  return indices;
}

/**
 * A field is flat when its sampled span is at most this threshold. This single
 * rule keeps `heightAt` (surface height) and `buildLandscape` (vertex colors)
 * consistent for near-constant fields.
 */
const FLAT_SPAN_EPSILON = 1e-9;

/**
 * Clamps a raw height sample against [min, min + span]. Finite samples pass
 * through unchanged. Non-finite samples (an L2 loss that overflows, for
 * instance) move to the nearest end of the range, so scene heights stay finite.
 * Exported for testing.
 */
export function clampFiniteHeight(raw: number, min: number, span: number): number {
  if (Number.isFinite(raw)) {
    return raw;
  }
  // NaN fails the > 0 comparison, so it also lands on the low end.
  return raw > 0 ? min + span : min;
}

/**
 * Normalizes a raw height sample to scene units. A flat field (span at most
 * FLAT_SPAN_EPSILON) collapses to 0. A non-flat field maps the clamped sample
 * into [0, HEIGHT_SCALE]. Exported for testing.
 */
export function normalizeHeight(raw: number, min: number, span: number): number {
  if (span <= FLAT_SPAN_EPSILON) {
    return 0;
  }
  const h = clampFiniteHeight(raw, min, span);
  return ((h - min) / span) * HEIGHT_SCALE;
}

export interface HeightRange {
  min: number;
  span: number;
}

/**
 * Samples `heightFn` over the (res+1)² grid covering [-domain, domain] and
 * returns the minimum and the span (max - min). Exported for testing; a
 * constant field returns span 0. Non-finite samples are skipped, and a field
 * with no finite sample reports a flat range so callers stay finite. A finite
 * field whose span overflows also reports a flat range.
 */
export function sampleHeightRange(
  heightFn: (w1: number, w2: number) => number,
  res: number,
  domain: number,
): HeightRange {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let j = 0; j <= res; j += 1) {
    const b = -domain + (2 * domain * j) / res;
    for (let i = 0; i <= res; i += 1) {
      const a = -domain + (2 * domain * i) / res;
      const h = heightFn(a, b);
      if (!Number.isFinite(h)) {
        continue;
      }
      min = Math.min(min, h);
      max = Math.max(max, h);
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, span: 0 };
  }
  const span = max - min;
  if (!Number.isFinite(span)) {
    return { min: 0, span: 0 };
  }
  return { min, span };
}

/** Smoothstep ease used for ball tweens; exported for testing. */
export function easeSmoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

interface BallTween {
  fromW1: number;
  fromW2: number;
  toW1: number;
  toW2: number;
  elapsed: number;
  onDone?: () => void;
}

export interface LandscapeScene {
  scene: import('three').Scene;
  camera: import('three').Camera;
  renderer: import('three').WebGLRenderer;
  setHeightField(heightFn: (w1: number, w2: number) => number, domain: number): void;
  moveBall(w1: number, w2: number): void;
  tweenBall(to: { w1: number; w2: number }, onDone?: () => void): void;
  clearTween(): void;
  heightAt(w1: number, w2: number): number;
  pushTrail(w1: number, w2: number): void;
  clearTrail(): void;
  update(dt: number): void;
  render(): void;
  dispose(): void;
}

/**
 * Builds a framework-agnostic Three.js landscape scene. Three is dynamically
 * imported so Rollup can split it into a separate chunk. The caller drives the
 * frame loop via update(dt) and render(); there is no internal rAF.
 */
export async function createLandscapeScene(canvas: HTMLCanvasElement): Promise<LandscapeScene> {
  const three = await import('three');

  const renderer = new three.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.width, canvas.height, false);

  const scene = new three.Scene();
  const camera = new three.PerspectiveCamera(50, canvas.width / canvas.height, 0.1, 100);

  scene.add(new three.AmbientLight(0xffffff, 0.75));
  const keyLight = new three.DirectionalLight(0xffffff, 2.2);
  keyLight.position.set(6, 10, 7);
  scene.add(keyLight);
  const fillLight = new three.DirectionalLight(0xffffff, 0.9);
  fillLight.position.set(-6, 5, -7);
  scene.add(fillLight);

  const accent = readColor(three, canvas, '--drl-accent', drlTokenValues.accent);
  const target = new three.Vector3(0, HEIGHT_SCALE / 2, 0);

  let heightFn: (w1: number, w2: number) => number = () => 0;
  let gridMin = 0;
  let gridSpan = 1;
  let domain = DEFAULT_DOMAIN;
  let radius = Math.max(10, domain * 1.6);
  let theta = CAMERA_THETA;
  let phi = CAMERA_PHI;

  let landscapeMesh: ThreeMesh | null = null;
  let ball: ThreeMesh | null = null;
  let ballW1 = 0;
  let ballW2 = 0;

  let trailGeometry: ThreeBufferGeometry | null = null;
  let trailMaterial: ThreeLineBasicMaterial | null = null;
  let trailAttribute: ThreeBufferAttribute | null = null;
  let trailPositions: Float32Array | null = null;
  let trailPoints: [number, number][] = [];

  let tween: BallTween | null = null;

  let dragging = false;
  let panning = false;
  let lastX = 0;
  let lastY = 0;

  let disposed = false;

  function heightAt(w1: number, w2: number): number {
    return normalizeHeight(heightFn(w1, w2), gridMin, gridSpan);
  }

  function buildLandscape(): void {
    if (landscapeMesh) {
      scene.remove(landscapeMesh);
      landscapeMesh.geometry.dispose();
      (landscapeMesh.material as ThreeMaterial).dispose();
      landscapeMesh = null;
    }
    const res = LANDSCAPE_RES;
    const count = (res + 1) * (res + 1);
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    const low = accent.clone().lerp(new three.Color(0xffffff), 0.85);
    const high = accent.clone().multiplyScalar(0.35);

    let index = 0;
    for (let j = 0; j <= res; j += 1) {
      const b = -domain + (2 * domain * j) / res;
      for (let i = 0; i <= res; i += 1) {
        const a = -domain + (2 * domain * i) / res;
        const height = heightAt(a, b);
        positions[index * 3] = a;
        positions[index * 3 + 1] = height;
        positions[index * 3 + 2] = b;
        // Colors reuse the scene height, so they can never diverge from the
        // geometry when the field evaluates to Infinity or NaN.
        const t = height / HEIGHT_SCALE;
        colors[index * 3] = low.r + (high.r - low.r) * t;
        colors[index * 3 + 1] = low.g + (high.g - low.g) * t;
        colors[index * 3 + 2] = low.b + (high.b - low.b) * t;
        index += 1;
      }
    }

    const geometry = new three.BufferGeometry();
    geometry.setIndex(gridIndex(res));
    geometry.setAttribute('position', new three.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new three.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new three.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0,
      side: three.DoubleSide,
    });
    const mesh = new three.Mesh(geometry, material);
    mesh.frustumCulled = false;
    scene.add(mesh);
    landscapeMesh = mesh;
  }

  function buildBall(): void {
    const geometry = new three.SphereGeometry(BALL_RADIUS, 24, 24);
    const material = new three.MeshStandardMaterial({
      color: accent,
      roughness: 0.4,
      metalness: 0.05,
    });
    const ballMesh = new three.Mesh(geometry, material);
    scene.add(ballMesh);
    ball = ballMesh;
  }

  function buildTrail(): void {
    const positions = new Float32Array(TRAIL_MAX * 3);
    const geometry = new three.BufferGeometry();
    const attribute = new three.BufferAttribute(positions, 3);
    geometry.setAttribute('position', attribute);
    geometry.setDrawRange(0, 0);

    const material = new three.LineBasicMaterial({
      color: accent,
      transparent: true,
      opacity: 0.55,
    });
    const line = new three.Line(geometry, material);
    line.frustumCulled = false;
    scene.add(line);

    trailGeometry = geometry;
    trailMaterial = material;
    trailAttribute = attribute;
    trailPositions = positions;
  }

  function moveBall(w1: number, w2: number): void {
    ballW1 = w1;
    ballW2 = w2;
    if (ball) {
      ball.position.set(w1, heightAt(w1, w2) + BALL_RADIUS, w2);
    }
  }

  /**
   * Rewrites the trail vertex positions from `trailPoints` using the current
   * `heightAt`. Shared by pushTrail and setHeightField so a height-field change
   * keeps the existing trail on the new surface.
   */
  function writeTrailPositions(): void {
    const positions = trailPositions;
    const attribute = trailAttribute;
    const geometry = trailGeometry;
    if (!positions || !attribute || !geometry) {
      return;
    }
    for (let i = 0; i < trailPoints.length; i += 1) {
      const [px, py] = trailPoints[i];
      positions[i * 3] = px;
      positions[i * 3 + 1] = heightAt(px, py) + TRAIL_LIFT;
      positions[i * 3 + 2] = py;
    }
    attribute.needsUpdate = true;
    geometry.setDrawRange(0, trailPoints.length);
  }

  function pushTrail(w1: number, w2: number): void {
    trailPoints.push([w1, w2]);
    while (trailPoints.length > TRAIL_MAX) {
      trailPoints.shift();
    }
    writeTrailPositions();
  }

  function clearTrail(): void {
    trailPoints = [];
    trailGeometry?.setDrawRange(0, 0);
  }

  function tweenBall(to: { w1: number; w2: number }, onDone?: () => void): void {
    tween = {
      fromW1: ballW1,
      fromW2: ballW2,
      toW1: to.w1,
      toW2: to.w2,
      elapsed: 0,
      onDone,
    };
  }

  function clearTween(): void {
    tween = null;
  }

  function update(dt: number): void {
    if (!tween) {
      return;
    }
    tween.elapsed += dt;
    const t = Math.min(1, tween.elapsed / (TWEEN_MS / 1000));
    const eased = easeSmoothstep(t);
    const w1 = tween.fromW1 + (tween.toW1 - tween.fromW1) * eased;
    const w2 = tween.fromW2 + (tween.toW2 - tween.fromW2) * eased;
    moveBall(w1, w2);
    if (t >= 1) {
      const onDone = tween.onDone;
      tween = null;
      onDone?.();
    }
  }

  function panCamera(dx: number, dy: number): void {
    camera.updateMatrixWorld();
    const right = new three.Vector3().setFromMatrixColumn(camera.matrix, 0);
    const up = new three.Vector3().setFromMatrixColumn(camera.matrix, 1);
    const panScale = radius * 0.0015;
    target.addScaledVector(right, -dx * panScale);
    target.addScaledVector(up, dy * panScale);
    updateCamera();
  }

  function updateCamera(): void {
    camera.position.set(
      target.x + radius * Math.sin(phi) * Math.sin(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.cos(theta),
    );
    camera.lookAt(target);
  }

  function render(): void {
    renderer.render(scene, camera);
  }

  function setHeightField(fn: (w1: number, w2: number) => number, newDomain: number): void {
    const previousDomain = domain;
    heightFn = fn;
    domain = newDomain;
    const { min, span } = sampleHeightRange(fn, LANDSCAPE_RES, domain);
    gridMin = min;
    gridSpan = span;
    buildLandscape();
    // Fit the camera to the domain only when it changes. Rebuilding the same
    // surface must keep the radius the user picked with the wheel.
    if (newDomain !== previousDomain) {
      radius = Math.max(10, domain * 1.6);
    }
    updateCamera();
    // The surface changed: keep the ball and the existing trail on the new one.
    moveBall(ballW1, ballW2);
    writeTrailPositions();
    render();
  }

  function onPointerDown(event: PointerEvent): void {
    if (event.button === 1) {
      panning = true;
      event.preventDefault();
    } else {
      dragging = true;
    }
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  }

  function onPointerUp(): void {
    dragging = false;
    panning = false;
  }

  function onPointerMove(event: PointerEvent): void {
    if (!dragging && !panning) {
      return;
    }
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    if (panning) {
      panCamera(dx, dy);
    } else {
      theta -= dx * 0.006;
      phi = clamp(phi - dy * 0.006, 0.15, Math.PI - 0.15);
      updateCamera();
    }
    render();
  }

  function onAuxClick(event: MouseEvent): void {
    if (event.button === 1) {
      event.preventDefault();
    }
  }

  function onWheel(event: WheelEvent): void {
    event.preventDefault();
    radius = clamp(radius + event.deltaY * 0.01, 1, 50);
    updateCamera();
    render();
  }

  function dispose(): void {
    if (disposed) {
      return;
    }
    disposed = true;
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointerleave', onPointerUp);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('auxclick', onAuxClick);
    canvas.removeEventListener('wheel', onWheel);

    if (landscapeMesh) {
      landscapeMesh.geometry.dispose();
      (landscapeMesh.material as ThreeMaterial).dispose();
      landscapeMesh = null;
    }
    if (ball) {
      ball.geometry.dispose();
      (ball.material as ThreeMaterial).dispose();
      ball = null;
    }
    trailGeometry?.dispose();
    trailMaterial?.dispose();
    trailGeometry = null;
    trailMaterial = null;
    trailAttribute = null;
    trailPositions = null;
    renderer.dispose();
  }

  buildLandscape();
  buildBall();
  buildTrail();
  updateCamera();

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointerleave', onPointerUp);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('auxclick', onAuxClick);
  canvas.addEventListener('wheel', onWheel, { passive: false });

  return {
    scene,
    camera,
    renderer,
    setHeightField,
    moveBall,
    tweenBall,
    clearTween,
    heightAt,
    pushTrail,
    clearTrail,
    update,
    render,
    dispose,
  };
}
