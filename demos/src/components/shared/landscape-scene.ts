import { drlTokenValues } from '../../styles/token-values.js';

type ThreeModule = typeof import('three');
type ThreeMesh = InstanceType<ThreeModule['Mesh']>;
type ThreeColor = InstanceType<ThreeModule['Color']>;
type ThreeBufferGeometry = InstanceType<ThreeModule['BufferGeometry']>;
type ThreeBufferAttribute = InstanceType<ThreeModule['BufferAttribute']>;
type ThreeMaterial = InstanceType<ThreeModule['Material']>;
type ThreeLineBasicMaterial = InstanceType<ThreeModule['LineBasicMaterial']>;

/**
 * Vertical exaggeration of the landscape: scene Y = normalized height × heightScale.
 * This is the default; setHeightScale can change it at runtime.
 */
export const DEFAULT_HEIGHT_SCALE = 1.5;
const LANDSCAPE_RES = 70;

/** Which cached height field the mesh, the ball, and the trail follow. */
export type HeightFieldKind = 'global' | 'test';

/**
 * One cached height field: the live function plus the sampled and normalized
 * data the mesh and the ball read. `domain` is the extent the samples cover,
 * so each field keeps its own extent when the active field switches.
 */
interface HeightField {
  fn: (w1: number, w2: number) => number;
  raw: Float64Array;
  min: number;
  span: number;
  scene: number[];
  domain: number;
}

/** Camera orbit radius floor so the ball stays in frame even for a flat scene. */
const MIN_CAMERA_RADIUS = 10;
/**
 * Extra camera distance per unit of height scale. Taller terrain needs more
 * room, otherwise the surface leaves the frustum.
 */
const CAMERA_RADIUS_PER_SCALE = 2;

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
 * Clamps a height scale to a usable value. Zero, negative, and non-finite
 * scales would break the camera fit and the vertex colors, so they fall back
 * to DEFAULT_HEIGHT_SCALE.
 */
export function sanitizeHeightScale(scale: number): number {
  return Number.isFinite(scale) && scale > 0 ? scale : DEFAULT_HEIGHT_SCALE;
}

/**
 * Normalizes a raw height sample to scene units. A flat field (span at most
 * FLAT_SPAN_EPSILON) collapses to 0. A non-flat field maps the clamped sample
 * into [0, scale]. Exported for testing.
 */
export function normalizeHeight(
  raw: number,
  min: number,
  span: number,
  scale: number = DEFAULT_HEIGHT_SCALE,
): number {
  if (span <= FLAT_SPAN_EPSILON) {
    return 0;
  }
  const h = clampFiniteHeight(raw, min, span);
  return ((h - min) / span) * scale;
}

/**
 * Maps `normalizeHeight` over a batch of raw samples and returns a plain
 * number array. Shared by `buildLandscape` so a scale change reuses the cached
 * raw samples instead of re-evaluating the field per vertex. Exported for
 * testing.
 */
export function applyHeightScale(
  rawHeights: ArrayLike<number>,
  min: number,
  span: number,
  scale: number,
): number[] {
  const heights: number[] = [];
  for (let i = 0; i < rawHeights.length; i += 1) {
    heights.push(normalizeHeight(rawHeights[i], min, span, scale));
  }
  return heights;
}

export interface HeightRange {
  min: number;
  span: number;
}

/**
 * Samples `heightFn` over the (res+1)² grid covering [-domain, domain] in
 * row-major order (j outer, i inner), matching the vertex layout of
 * `buildLandscape`. Returns a Float64Array of length (res+1)². Exported for
 * testing; the caller can reuse the samples instead of re-evaluating the field.
 */
export function sampleGridHeights(
  heightFn: (w1: number, w2: number) => number,
  res: number,
  domain: number,
): Float64Array {
  const samples = new Float64Array((res + 1) * (res + 1));
  let index = 0;
  for (let j = 0; j <= res; j += 1) {
    const b = -domain + (2 * domain * j) / res;
    for (let i = 0; i <= res; i += 1) {
      const a = -domain + (2 * domain * i) / res;
      samples[index] = heightFn(a, b);
      index += 1;
    }
  }
  return samples;
}

/**
 * Reads one sample, treating a non-finite value as 0. The sampler stores
 * non-finite values verbatim, but `applyHeightScale` already clamped them into
 * the normalized array. This guard only covers a hand-crafted sample array that
 * still contains a non-finite value.
 */
function finiteOrZero(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Bilinearly interpolates the row-major samples at the (w1, w2) point. Callers
 * pass either raw samples from `sampleGridHeights` or the normalized scene
 * heights from `applyHeightScale`; the interpolation is the same. The grid
 * spans [-domain, domain] on both axes; i maps from w1 and j maps from w2,
 * matching the vertex layout of `buildLandscape`. Points outside the domain
 * clamp to the nearest edge. Exported for testing.
 */
export function interpolateHeight(
  samples: ArrayLike<number>,
  res: number,
  domain: number,
  w1: number,
  w2: number,
): number {
  const i = clamp(((w1 + domain) / (2 * domain)) * res, 0, res);
  const j = clamp(((w2 + domain) / (2 * domain)) * res, 0, res);
  const i0 = Math.min(Math.floor(i), res - 1);
  const j0 = Math.min(Math.floor(j), res - 1);
  const i1 = Math.min(i0 + 1, res);
  const j1 = Math.min(j0 + 1, res);
  const ti = i - i0;
  const tj = j - j0;
  const stride = res + 1;
  const top = lerp(
    finiteOrZero(samples[j0 * stride + i0]),
    finiteOrZero(samples[j0 * stride + i1]),
    ti,
  );
  const bottom = lerp(
    finiteOrZero(samples[j1 * stride + i0]),
    finiteOrZero(samples[j1 * stride + i1]),
    ti,
  );
  return lerp(top, bottom, tj);
}

/**
 * Reduces raw grid samples to a { min, span } range. Non-finite samples are
 * skipped, a field with no finite sample reports a flat range so callers stay
 * finite, and a finite field whose span overflows also reports a flat range.
 * Exported for testing.
 */
export function heightRangeFromSamples(heights: ArrayLike<number>): HeightRange {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < heights.length; i += 1) {
    const h = heights[i];
    if (!Number.isFinite(h)) {
      continue;
    }
    min = Math.min(min, h);
    max = Math.max(max, h);
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

/**
 * Samples `heightFn` over the (res+1)² grid covering [-domain, domain] and
 * returns the minimum and the span (max - min). Exported for testing; a
 * constant field returns span 0. See heightRangeFromSamples for the edge
 * cases.
 */
export function sampleHeightRange(
  heightFn: (w1: number, w2: number) => number,
  res: number,
  domain: number,
): HeightRange {
  return heightRangeFromSamples(sampleGridHeights(heightFn, res, domain));
}

/**
 * Samples `heightFn` over the (res+1)² grid covering [-domain, domain], reduces
 * the samples to a { min, span } range, and normalizes them to scene units.
 * Shared by setHeightField and setTestHeightField so both cached fields use
 * the same sampling and normalization path. Exported for testing.
 */
export function sampleHeightField(
  heightFn: (w1: number, w2: number) => number,
  res: number,
  domain: number,
  scale: number = DEFAULT_HEIGHT_SCALE,
): { raw: Float64Array; min: number; span: number; scene: number[] } {
  const raw = sampleGridHeights(heightFn, res, domain);
  const { min, span } = heightRangeFromSamples(raw);
  const scene = applyHeightScale(raw, min, span, scale);
  return { raw, min, span, scene };
}

/**
 * Camera orbit radius for a landscape of the given extent and height scale.
 * The domain floor keeps the whole grid in view; the scale floor keeps taller
 * terrain in view. Exported for testing.
 */
export function cameraRadius(domain: number, heightScale: number): number {
  return Math.max(MIN_CAMERA_RADIUS, domain * 1.6, heightScale * CAMERA_RADIUS_PER_SCALE);
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
  setTestHeightField(heightFn: (w1: number, w2: number) => number, domain: number): void;
  setActiveHeightField(field: HeightFieldKind): void;
  setHeightScale(scale: number): void;
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
  const target = new three.Vector3(0, DEFAULT_HEIGHT_SCALE / 2, 0);

  // Two cached height fields. The mesh, the ball, and the trail follow the
  // active one; the inactive one keeps its samples so switching back is cheap.
  let globalField: HeightField | null = null;
  let testField: HeightField | null = null;
  let activeField: HeightFieldKind = 'global';
  // The domain the camera was last fit to. refreshActive refits only when the
  // active field's domain differs from this value.
  let cameraDomain = DEFAULT_DOMAIN;
  let heightScale = DEFAULT_HEIGHT_SCALE;
  let radius = cameraRadius(DEFAULT_DOMAIN, heightScale);
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

  function activeFieldObject(): HeightField | null {
    return activeField === 'test' ? testField : globalField;
  }

  function activeDomain(): number {
    return activeFieldObject()?.domain ?? DEFAULT_DOMAIN;
  }

  function heightAt(w1: number, w2: number): number {
    const field = activeFieldObject();
    if (!field) {
      return 0;
    }
    // Prefer the cached normalized samples so the ball and the trail stay on
    // the mesh that is currently displayed. Between rebuilds the live field may
    // change (the core's bias moves) while the mesh stays cached, which would
    // otherwise leave the ball floating or sunk into the surface. The samples
    // are already normalized, so no second normalizeHeight call is needed.
    if (Math.abs(w1) <= field.domain && Math.abs(w2) <= field.domain) {
      return interpolateHeight(field.scene, LANDSCAPE_RES, field.domain, w1, w2);
    }
    return normalizeHeight(field.fn(w1, w2), field.min, field.span, heightScale);
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

    // Scene heights reuse the active field's cached normalized samples so a
    // scale change never re-evaluates the loss function. With no field the mesh
    // is a flat plane of zeros.
    const field = activeFieldObject();
    const heights = field ? field.scene : null;
    const fieldDomain = field ? field.domain : DEFAULT_DOMAIN;

    const low = accent.clone().lerp(new three.Color(0xffffff), 0.85);
    const high = accent.clone().multiplyScalar(0.35);

    let index = 0;
    for (let j = 0; j <= res; j += 1) {
      const b = -fieldDomain + (2 * fieldDomain * j) / res;
      for (let i = 0; i <= res; i += 1) {
        const a = -fieldDomain + (2 * fieldDomain * i) / res;
        const height = heights === null ? 0 : heights[index];
        positions[index * 3] = a;
        positions[index * 3 + 1] = height;
        positions[index * 3 + 2] = b;
        // Colors reuse the scene height, so they can never diverge from the
        // geometry when the field evaluates to Infinity or NaN. Dividing by
        // the scale keeps the color factor scale-independent.
        const t = height / heightScale;
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
   * `heightAt`. Shared by every path that changes the surface (pushTrail,
   * refreshActive, setHeightScale), so a height-field or scale change keeps the
   * existing trail on the new surface.
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

  /**
   * Rebuilds the mesh from the active field, refits the camera only when the
   * domain changed, and keeps the ball and the trail on the new surface. Shared
   * by every path that changes the displayed field.
   */
  function refreshActive(): void {
    buildLandscape();
    // Fit the camera to the domain only when it changes. Rebuilding the same
    // surface must keep the radius the user picked with the wheel.
    const currentDomain = activeDomain();
    if (currentDomain !== cameraDomain) {
      radius = cameraRadius(currentDomain, heightScale);
      cameraDomain = currentDomain;
    }
    updateCamera();
    // The surface changed: keep the ball and the existing trail on the new one.
    moveBall(ballW1, ballW2);
    writeTrailPositions();
    render();
  }

  function setHeightField(fn: (w1: number, w2: number) => number, newDomain: number): void {
    globalField = {
      fn,
      ...sampleHeightField(fn, LANDSCAPE_RES, newDomain, heightScale),
      domain: newDomain,
    };
    if (activeField === 'global') {
      refreshActive();
    }
  }

  function setTestHeightField(fn: (w1: number, w2: number) => number, newDomain: number): void {
    testField = {
      fn,
      ...sampleHeightField(fn, LANDSCAPE_RES, newDomain, heightScale),
      domain: newDomain,
    };
    if (activeField === 'test') {
      refreshActive();
    }
  }

  function setActiveHeightField(field: HeightFieldKind): void {
    if (field === activeField) {
      return;
    }
    activeField = field;
    const active = activeFieldObject();
    if (active) {
      // The newly active field was sampled at an earlier scale; re-normalize it
      // at the current heightScale so the switch keeps the same exaggeration.
      active.scene = applyHeightScale(active.raw, active.min, active.span, heightScale);
    }
    refreshActive();
  }

  /**
   * Changes the vertical exaggeration without re-evaluating the loss function.
   * The active field's cached raw samples are re-normalized, so the geometry,
   * the ball, and the trail all move to the new surface in one pass.
   */
  function setHeightScale(scale: number): void {
    const previousScale = heightScale;
    const next = sanitizeHeightScale(scale);
    if (next === heightScale) {
      return;
    }
    heightScale = next;
    // Keep the pan offset relative to the surface: the target only moves by
    // the change in surface height, instead of snapping back to the centre.
    target.y += (heightScale - previousScale) / 2;
    // Re-normalize only the active field; the inactive field keeps its raw
    // samples and is re-normalized when it becomes active.
    const field = activeFieldObject();
    if (field) {
      field.scene = applyHeightScale(field.raw, field.min, field.span, heightScale);
    }
    buildLandscape();
    // Never zoom in: a taller surface needs at least as much room.
    radius = Math.max(radius, cameraRadius(activeDomain(), heightScale));
    updateCamera();
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
    setTestHeightField,
    setActiveHeightField,
    setHeightScale,
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
