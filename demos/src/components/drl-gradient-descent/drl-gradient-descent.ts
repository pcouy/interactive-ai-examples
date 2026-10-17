import type { PropertyValues, TemplateResult } from 'lit';
import { css, html } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { DemoElement } from '../../base/demo-element.js';
import type { Example, Gradient2, Phase } from '../../core/gradient-descent.js';
import {
  exampleGradient,
  exampleLoss,
  GradientDescentCore,
  landscapeLoss,
} from '../../core/gradient-descent.js';
import { drlTokens } from '../../styles/tokens.js';

type ThreeModule = typeof import('three');
type ThreeRenderer = InstanceType<ThreeModule['WebGLRenderer']>;
type ThreeScene = InstanceType<ThreeModule['Scene']>;
type ThreeCamera = InstanceType<ThreeModule['PerspectiveCamera']>;
type ThreeMesh = InstanceType<ThreeModule['Mesh']>;
type ThreeGroup = InstanceType<ThreeModule['Group']>;
type ThreeColor = InstanceType<ThreeModule['Color']>;
type ThreeVector3 = InstanceType<ThreeModule['Vector3']>;
type ThreeBufferGeometry = InstanceType<ThreeModule['BufferGeometry']>;
type ThreeBufferAttribute = InstanceType<ThreeModule['BufferAttribute']>;
type ThreeArrowHelper = InstanceType<ThreeModule['ArrowHelper']>;
type ThreeMaterial = InstanceType<ThreeModule['Material']>;
type ThreeMeshStandardMaterial = InstanceType<ThreeModule['MeshStandardMaterial']>;
type ThreeLineBasicMaterial = InstanceType<ThreeModule['LineBasicMaterial']>;

const DOMAIN = GradientDescentCore.DOMAIN;
/**
 * Vertical exaggeration of the landscape: scene Y = raw loss × HEIGHT_SCALE.
 * Raw loss peaks at ≈ 5.35 over [-3,3]², so HEIGHT_SCALE 0.8 gives a max
 * scene height ≈ 4.3 — steep enough for the paraboloid valley to read clearly
 * (the former 0.35 looked flat).
 */
const HEIGHT_SCALE = 1.5;
const LANDSCAPE_RES = 70;

/**
 * Camera framing tuned for the taller valley: slightly more top-down
 * (smaller phi) and a raised look-at target so rim and bottom stay in frame.
 */
const CAMERA_RADIUS = 10;
const CAMERA_THETA = Math.PI * 0.28;
const CAMERA_PHI = Math.PI * 0.32;
const CAMERA_TARGET_Y = 1;

const BALL_RADIUS = 0.14;

const TRAIL_MAX = 60;
const TRAIL_LIFT = 0.05;

const PATCH_RES = 24;
const PATCH_HALF = 0.8;
const PATCH_MAX_HEIGHT = 0.45;
const PATCH_LIFT = 0.03;

const ARROW_SCALE = 0.25;
const ARROW_MAX = 1.2;
const ARROW_COLLECTED_OPACITY = 0.25;
const ARROW_COLOR_FALLBACK = '#d32f2f';
const COLLECTED_Y_STEP = 0.04;

const MAX_BATCH = 12;
const TWEEN_MS = 600;
const RESET_SEED = 42;

const CANVAS_WIDTH = 1920;
const CANVAS_HEIGHT = 1080;

const PHASE_LABELS: Record<Phase, string> = {
  idle: 'Prêt',
  present: 'Exemple',
  gradient: 'Gradient',
  fade: 'Archivage',
  mean: 'Moyenne',
  move: 'Descente',
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formatNumber(value: number): string {
  return value.toFixed(2).replace('.', ',');
}

function gridIndex(res: number): number[] {
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

interface BallTween {
  fromA: number;
  fromB: number;
  toA: number;
  toB: number;
  start: number;
}

@customElement('drl-gradient-descent')
export class GradientDescent extends DemoElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: block;
      }

      .drl-status {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-2);
        min-height: 1.5rem;
      }

      .drl-badge {
        background: var(--drl-accent-2);
        color: var(--drl-surface);
        border-radius: calc(var(--drl-radius) / 2);
        padding: 0 var(--drl-space-2);
        font-size: 0.85em;
      }

      .drl-status-text {
        color: var(--drl-muted);
      }

      .drl-toolbar {
        display: flex;
        flex-wrap: wrap;
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-2);
      }

      .drl-button {
        font: inherit;
        display: inline-flex;
        align-items: center;
        gap: var(--drl-space-1);
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-bg);
        color: var(--drl-fg);
        cursor: pointer;
      }

      .drl-button:hover:not(:disabled) {
        border-color: var(--drl-accent);
      }

      .drl-button:disabled {
        color: var(--drl-muted);
        cursor: not-allowed;
      }

      .drl-stats {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: var(--drl-space-2);
        margin-bottom: var(--drl-space-2);
      }

      .drl-stat {
        background: var(--drl-bg);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        padding: var(--drl-space-2);
      }

      .drl-stat-label {
        margin: 0 0 2px;
        color: var(--drl-muted);
        font-size: 0.8em;
      }

      .drl-stat-value {
        margin: 0;
        font-variant-numeric: tabular-nums;
      }

      canvas {
        display: block;
        width: 100%;
        max-width: ${CANVAS_WIDTH}px;
        height: auto;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
        cursor: grab;
        touch-action: none;
      }

      canvas:active {
        cursor: grabbing;
      }

      .drl-hint {
        margin: var(--drl-space-1) 0 0;
        color: var(--drl-muted);
        font-size: 0.85em;
      }

      .drl-webgl-error {
        color: var(--drl-muted);
      }
    `,
  ];

  @property({ type: Number, attribute: 'learning-rate' })
  learningRate = 0.15;

  @property({ type: Number, attribute: 'start-a' })
  startA = -2.5;

  @property({ type: Number, attribute: 'start-b' })
  startB = 2.2;

  @property({ type: Number, attribute: 'batch-size' })
  batchSize = 4;

  @state()
  private webglError = false;

  @state()
  private phase: Phase = 'idle';

  @state()
  private epoch = 0;

  @state()
  private ballA = 0;

  @state()
  private ballB = 0;

  @state()
  private lossValue = 0;

  @state()
  private gradMag = 0;

  @state()
  private currentExample: Example | null = null;

  @state()
  private exampleNumber = 0;

  @state()
  private animating = false;

  private readonly core = new GradientDescentCore();

  private THREE: ThreeModule | null = null;

  private renderer: ThreeRenderer | null = null;

  private scene: ThreeScene | null = null;

  private camera: ThreeCamera | null = null;

  private canvasEl: HTMLCanvasElement | null = null;

  private landscapeGeometry: ThreeBufferGeometry | null = null;

  private landscapeMaterial: ThreeMeshStandardMaterial | null = null;

  private patchGeometry: ThreeBufferGeometry | null = null;

  private patchMaterial: ThreeMeshStandardMaterial | null = null;

  private patchMesh: ThreeMesh | null = null;

  private patchExample: Example | null = null;

  private patchLowColor: ThreeColor | null = null;

  private patchHighColor: ThreeColor | null = null;

  private patchOpacity = 0;

  private patchTarget = 0;

  private ball: ThreeMesh | null = null;

  private trailGeometry: ThreeBufferGeometry | null = null;

  private trailMaterial: ThreeLineBasicMaterial | null = null;

  private trailAttribute: ThreeBufferAttribute | null = null;

  private trailPositions: Float32Array | null = null;

  private trailPoints: [number, number][] = [];

  private gradientGroup: ThreeGroup | null = null;

  private solidArrow: ThreeArrowHelper | null = null;

  private meanArrow: ThreeArrowHelper | null = null;

  private collectedArrows: ThreeArrowHelper[] = [];

  private tmpDir: ThreeVector3 | null = null;

  private target: ThreeVector3 | null = null;

  private radius = CAMERA_RADIUS;

  private theta = CAMERA_THETA;

  private phi = CAMERA_PHI;

  private dragging = false;

  private panning = false;

  private lastX = 0;

  private lastY = 0;

  private tween: BallTween | null = null;

  private tweenBallA = 0;

  private tweenBallB = 0;

  private animFrame: number | null = null;

  private prevFrame = 0;

  private lastFrameDelta = 0.016;

  protected override async firstUpdated(): Promise<void> {
    const canvas = this.renderRoot.querySelector('canvas');
    if (!(canvas instanceof HTMLCanvasElement)) {
      return;
    }
    try {
      const three = await import('three');
      if (!this.isConnected || this.renderer !== null) {
        return;
      }
      this.buildScene(three, canvas);
      this.attachControls(canvas);
      this.core.reset({
        startA: this.startA,
        startB: this.startB,
        batchSize: clamp(this.batchSize, 1, MAX_BATCH),
        seed: RESET_SEED,
      });
      this.sync();
    } catch {
      this.webglError = true;
    }
  }

  private buildScene(three: ThreeModule, canvas: HTMLCanvasElement): void {
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

    const accent = this.readColor(three, '--drl-accent', '#ffffff');
    const accentTwo = this.readColor(three, '--drl-accent-2', '#ffffff');
    const arrowColor = this.readColor(three, '--drl-arrow', ARROW_COLOR_FALLBACK);

    this.THREE = three;
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.canvasEl = canvas;
    this.target = new three.Vector3(0, CAMERA_TARGET_Y, 0);
    this.tmpDir = new three.Vector3();

    this.buildLandscape(three, accent);
    this.buildPatchMesh(three, accentTwo);
    this.buildBall(three, accent);
    this.buildTrail(three, accent);
    this.buildArrows(three, arrowColor);
    this.updateCamera();
  }

  private readColor(three: ThreeModule, name: string, fallback: string): ThreeColor {
    const value = getComputedStyle(this).getPropertyValue(name).trim();
    return new three.Color(value.length > 0 ? value : fallback);
  }

  private buildLandscape(three: ThreeModule, accent: ThreeColor): void {
    const scene = this.scene;
    if (!scene) {
      return;
    }
    const res = LANDSCAPE_RES;
    const count = (res + 1) * (res + 1);
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const heights: number[] = [];
    let minHeight = Number.POSITIVE_INFINITY;
    let maxHeight = Number.NEGATIVE_INFINITY;

    for (let j = 0; j <= res; j += 1) {
      const b = -DOMAIN + (2 * DOMAIN * j) / res;
      for (let i = 0; i <= res; i += 1) {
        const a = -DOMAIN + (2 * DOMAIN * i) / res;
        const h = landscapeLoss(a, b);
        heights.push(h);
        minHeight = Math.min(minHeight, h);
        maxHeight = Math.max(maxHeight, h);
      }
    }

    const low = accent.clone().lerp(new three.Color(0xffffff), 0.85);
    const high = accent.clone().multiplyScalar(0.35);
    const span = maxHeight - minHeight || 1;

    let index = 0;
    for (let j = 0; j <= res; j += 1) {
      const b = -DOMAIN + (2 * DOMAIN * j) / res;
      for (let i = 0; i <= res; i += 1) {
        const a = -DOMAIN + (2 * DOMAIN * i) / res;
        const h = heights[index];
        positions[index * 3] = a;
        positions[index * 3 + 1] = h * HEIGHT_SCALE;
        positions[index * 3 + 2] = b;
        const t = (h - minHeight) / span;
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

    this.landscapeGeometry = geometry;
    this.landscapeMaterial = material;
  }

  private buildPatchMesh(three: ThreeModule, accentTwo: ThreeColor): void {
    const scene = this.scene;
    if (!scene) {
      return;
    }
    const count = (PATCH_RES + 1) * (PATCH_RES + 1);
    const geometry = new three.BufferGeometry();
    geometry.setIndex(gridIndex(PATCH_RES));
    geometry.setAttribute('position', new three.BufferAttribute(new Float32Array(count * 3), 3));
    geometry.setAttribute('color', new three.BufferAttribute(new Float32Array(count * 3), 3));

    const material = new three.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.65,
      metalness: 0,
      side: three.DoubleSide,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    const mesh = new three.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.visible = false;
    scene.add(mesh);

    this.patchGeometry = geometry;
    this.patchMaterial = material;
    this.patchMesh = mesh;
    this.patchLowColor = accentTwo.clone().lerp(new three.Color(0xffffff), 0.8);
    this.patchHighColor = accentTwo.clone().multiplyScalar(0.4);
  }

  private buildBall(three: ThreeModule, accent: ThreeColor): void {
    const scene = this.scene;
    if (!scene) {
      return;
    }
    const geometry = new three.SphereGeometry(BALL_RADIUS, 24, 24);
    const material = new three.MeshStandardMaterial({
      color: accent,
      roughness: 0.4,
      metalness: 0.05,
    });
    const ball = new three.Mesh(geometry, material);
    scene.add(ball);
    this.ball = ball;
  }

  private buildTrail(three: ThreeModule, accent: ThreeColor): void {
    const scene = this.scene;
    if (!scene) {
      return;
    }
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

    this.trailGeometry = geometry;
    this.trailMaterial = material;
    this.trailAttribute = attribute;
    this.trailPositions = positions;
  }

  private buildArrows(three: ThreeModule, arrowColor: ThreeColor): void {
    const scene = this.scene;
    if (!scene) {
      return;
    }
    const group = new three.Group();
    scene.add(group);
    this.gradientGroup = group;

    this.solidArrow = this.createArrow(three, arrowColor, 0.95);
    this.meanArrow = this.createArrow(three, arrowColor, 0.95);
    this.collectedArrows = [];
    for (let i = 0; i < MAX_BATCH; i += 1) {
      this.collectedArrows.push(this.createArrow(three, arrowColor, ARROW_COLLECTED_OPACITY));
    }
  }

  private createArrow(three: ThreeModule, color: ThreeColor, opacity: number): ThreeArrowHelper {
    const arrow = new three.ArrowHelper(
      new three.Vector3(0, 1, 0),
      new three.Vector3(),
      1,
      color,
      0.22,
      0.12,
    );
    const lineMaterial = arrow.line.material as ThreeMaterial;
    lineMaterial.transparent = opacity < 1;
    lineMaterial.opacity = opacity;
    lineMaterial.depthWrite = opacity >= 1;
    const coneMaterial = arrow.cone.material as ThreeMaterial;
    coneMaterial.transparent = opacity < 1;
    coneMaterial.opacity = opacity;
    coneMaterial.depthWrite = opacity >= 1;
    arrow.visible = false;
    this.gradientGroup?.add(arrow);
    return arrow;
  }

  private buildPatch(example: Example): void {
    const three = this.THREE;
    const geometry = this.patchGeometry;
    if (!three || !geometry) {
      return;
    }
    const res = PATCH_RES;
    const half = PATCH_HALF;
    const count = (res + 1) * (res + 1);
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const heights = new Float32Array(count);
    const baseA = this.core.ballA;
    const baseB = this.core.ballB;
    const baseY = landscapeLoss(baseA, baseB) * HEIGHT_SCALE;

    let maxHeight = 0;
    let index = 0;
    for (let j = 0; j <= res; j += 1) {
      const db = -half + (2 * half * j) / res;
      for (let i = 0; i <= res; i += 1) {
        const da = -half + (2 * half * i) / res;
        const h = exampleLoss(baseA + da, baseB + db, example);
        heights[index] = h;
        maxHeight = Math.max(maxHeight, h);
        index += 1;
      }
    }

    const scale = maxHeight > 1e-9 ? PATCH_MAX_HEIGHT / maxHeight : 0;
    const low = this.patchLowColor;
    const high = this.patchHighColor;

    index = 0;
    for (let j = 0; j <= res; j += 1) {
      const db = -half + (2 * half * j) / res;
      for (let i = 0; i <= res; i += 1) {
        const da = -half + (2 * half * i) / res;
        const h = heights[index];
        const t = maxHeight > 1e-9 ? h / maxHeight : 0;
        positions[index * 3] = baseA + da;
        positions[index * 3 + 1] = baseY + h * scale + PATCH_LIFT;
        positions[index * 3 + 2] = baseB + db;
        if (low && high) {
          colors[index * 3] = low.r + (high.r - low.r) * t;
          colors[index * 3 + 1] = low.g + (high.g - low.g) * t;
          colors[index * 3 + 2] = low.b + (high.b - low.b) * t;
        }
        index += 1;
      }
    }

    geometry.setAttribute('position', new three.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new three.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
  }

  private get displayA(): number {
    return this.tween ? this.tweenBallA : this.core.ballA;
  }

  private get displayB(): number {
    return this.tween ? this.tweenBallB : this.core.ballB;
  }

  private updateVisuals(): void {
    const three = this.THREE;
    const ball = this.ball;
    if (!three || !ball) {
      return;
    }
    const a = this.displayA;
    const b = this.displayB;
    ball.position.set(a, landscapeLoss(a, b) * HEIGHT_SCALE + BALL_RADIUS, b);

    const example = this.core.currentExample;
    if (example !== this.patchExample) {
      this.patchExample = example;
      if (example) {
        this.buildPatch(example);
      }
    }

    this.patchTarget = this.tween ? 0 : this.patchTargetFor(this.core.phase);
    const blend = clamp(this.lastFrameDelta * 8, 0, 1);
    this.patchOpacity += (this.patchTarget - this.patchOpacity) * blend;
    if (this.patchMesh && this.patchMaterial) {
      this.patchMaterial.opacity = this.patchOpacity;
      this.patchMesh.visible = this.patchOpacity > 0.01;
    }

    this.updateArrows();
  }

  private patchTargetFor(phase: Phase): number {
    return phase === 'present' || phase === 'gradient' ? 1 : 0;
  }
  private updateArrows(): void {
    if (this.tween) {
      this.hideAllArrows();
      return;
    }
    const core = this.core;
    const phase = core.phase;
    const baseA = core.ballA;
    const baseB = core.ballB;
    const originY = landscapeLoss(baseA, baseB) * HEIGHT_SCALE + BALL_RADIUS;
    const example = core.currentExample;

    if ((phase === 'present' || phase === 'gradient') && example !== null) {
      this.placeArrow(
        this.solidArrow,
        baseA,
        originY,
        baseB,
        exampleGradient(baseA, baseB, example),
      );
    } else if (this.solidArrow) {
      this.solidArrow.visible = false;
    }

    const collected = core.collected;
    const showCollected = phase === 'present' || phase === 'gradient' || phase === 'fade';
    for (let i = 0; i < this.collectedArrows.length; i += 1) {
      const arrow = this.collectedArrows[i];
      if (showCollected && i < collected.length) {
        this.placeArrow(arrow, baseA, originY + (i + 1) * COLLECTED_Y_STEP, baseB, collected[i]);
      } else {
        arrow.visible = false;
      }
    }

    if (phase === 'mean' && core.batchGradient !== null) {
      this.placeArrow(this.meanArrow, baseA, originY, baseB, core.batchGradient);
    } else if (this.meanArrow) {
      this.meanArrow.visible = false;
    }
  }

  private hideAllArrows(): void {
    if (this.solidArrow) {
      this.solidArrow.visible = false;
    }
    if (this.meanArrow) {
      this.meanArrow.visible = false;
    }
    for (const arrow of this.collectedArrows) {
      arrow.visible = false;
    }
  }

  private placeArrow(
    arrow: ThreeArrowHelper | null,
    x: number,
    y: number,
    z: number,
    gradient: Gradient2,
  ): void {
    const dir = this.tmpDir;
    if (!arrow || !dir) {
      return;
    }
    const magnitude = Math.hypot(gradient.da, gradient.db);
    if (magnitude < 1e-4) {
      arrow.visible = false;
      return;
    }
    arrow.visible = true;
    arrow.position.set(x, y, z);
    // Arrows show the descent direction (-gradient), i.e. where the ball will
    // move; the core keeps exposing true gradients.
    dir.set(-gradient.da / magnitude, 0, -gradient.db / magnitude);
    arrow.setDirection(dir);
    const length = clamp(magnitude * ARROW_SCALE, 0.25, ARROW_MAX);
    arrow.setLength(length, Math.min(0.28, length * 0.32), Math.min(0.18, length * 0.2));
  }

  private pushTrail(a: number, b: number): void {
    this.trailPoints.push([a, b]);
    while (this.trailPoints.length > TRAIL_MAX) {
      this.trailPoints.shift();
    }
    const positions = this.trailPositions;
    const attribute = this.trailAttribute;
    const geometry = this.trailGeometry;
    if (!positions || !attribute || !geometry) {
      return;
    }
    for (let i = 0; i < this.trailPoints.length; i += 1) {
      const [px, py] = this.trailPoints[i];
      positions[i * 3] = px;
      positions[i * 3 + 1] = landscapeLoss(px, py) * HEIGHT_SCALE + TRAIL_LIFT;
      positions[i * 3 + 2] = py;
    }
    attribute.needsUpdate = true;
    geometry.setDrawRange(0, this.trailPoints.length);
  }

  private clearTrail(): void {
    this.trailPoints = [];
    this.trailGeometry?.setDrawRange(0, 0);
  }

  override step(): void {
    if (this.tween) {
      return;
    }
    this.core.advance(this.learningRate, { detailed: !this.running });
    if (this.core.phase === 'move') {
      this.pushTrail(this.core.ballA, this.core.ballB);
    }
    this.sync();
  }

  private readonly handleSingleStep = (): void => {
    if (!this.running) {
      this.step();
    }
  };

  private readonly handleBatch = (): void => {
    if (this.running || this.tween) {
      return;
    }
    const move = this.core.advanceBatch(this.learningRate);
    if (move) {
      this.tween = {
        fromA: move.fromA,
        fromB: move.fromB,
        toA: move.toA,
        toB: move.toB,
        start: performance.now(),
      };
      this.tweenBallA = move.fromA;
      this.tweenBallB = move.fromB;
    }
    this.animating = move !== null;
    this.sync();
  };

  private sync(): void {
    const core = this.core;
    this.phase = core.phase;
    this.epoch = core.epoch;
    this.ballA = core.ballA;
    this.ballB = core.ballB;
    this.lossValue = core.loss;
    this.gradMag = core.gradientMagnitude;
    this.currentExample = core.currentExample;
    this.exampleNumber = core.exampleIndex + 1;
    this.updateVisuals();
    this.requestRender();
  }

  private applyReset(): void {
    if (this.animFrame !== null) {
      cancelAnimationFrame(this.animFrame);
      this.animFrame = null;
    }
    this.tween = null;
    this.animating = false;
    this.core.reset({
      startA: this.startA,
      startB: this.startB,
      batchSize: clamp(this.batchSize, 1, MAX_BATCH),
      seed: RESET_SEED,
    });
    this.patchExample = null;
    this.patchOpacity = 0;
    this.patchTarget = 0;
    if (this.patchMesh && this.patchMaterial) {
      this.patchMesh.visible = false;
      this.patchMaterial.opacity = 0;
    }
    this.hideAllArrows();
    this.clearTrail();
    this.sync();
  }

  protected override onReset(): void {
    this.applyReset();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    super.willUpdate(changed);
    if (this.hasUpdated && (changed.has('startA') || changed.has('startB'))) {
      this.applyReset();
    }
  }

  private readonly requestRender = (): void => {
    if (this.animFrame !== null) {
      return;
    }
    this.animFrame = requestAnimationFrame(this.onAnimFrame);
  };

  private readonly onAnimFrame = (time: number): void => {
    this.animFrame = null;
    const delta = this.prevFrame === 0 ? 0.016 : Math.min(0.1, (time - this.prevFrame) / 1000);
    this.prevFrame = time;
    this.lastFrameDelta = delta;

    let busy = false;
    if (this.tween) {
      const t = Math.min(1, (time - this.tween.start) / TWEEN_MS);
      const eased = t * t * (3 - 2 * t);
      this.tweenBallA = this.tween.fromA + (this.tween.toA - this.tween.fromA) * eased;
      this.tweenBallB = this.tween.fromB + (this.tween.toB - this.tween.fromB) * eased;
      if (t >= 1) {
        this.pushTrail(this.tween.toA, this.tween.toB);
        this.tween = null;
        this.animating = false;
      } else {
        busy = true;
      }
    }
    this.updateVisuals();
    if (Math.abs(this.patchOpacity - this.patchTarget) > 0.01) {
      busy = true;
    }
    this.drawScene();
    if (busy) {
      this.requestRender();
    }
  };

  private attachControls(canvas: HTMLCanvasElement): void {
    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointerleave', this.onPointerUp);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('auxclick', this.onAuxClick);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
  }

  private readonly onPointerDown = (event: PointerEvent): void => {
    const canvas = this.canvasEl;
    if (!canvas) {
      return;
    }
    if (event.button === 1) {
      this.panning = true;
      event.preventDefault();
    } else {
      this.dragging = true;
    }
    this.lastX = event.clientX;
    this.lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  };

  private readonly onPointerUp = (): void => {
    this.dragging = false;
    this.panning = false;
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (!this.dragging && !this.panning) {
      return;
    }
    const dx = event.clientX - this.lastX;
    const dy = event.clientY - this.lastY;
    this.lastX = event.clientX;
    this.lastY = event.clientY;
    if (this.panning) {
      this.panCamera(dx, dy);
    } else {
      this.theta -= dx * 0.006;
      this.phi = clamp(this.phi - dy * 0.006, 0.15, Math.PI - 0.15);
      this.updateCamera();
    }
    this.requestRender();
  };

  private readonly onAuxClick = (event: MouseEvent): void => {
    if (event.button === 1) {
      event.preventDefault();
    }
  };

  private readonly onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    this.radius = clamp(this.radius + event.deltaY * 0.01, 1, 50);
    this.updateCamera();
    this.requestRender();
  };

  private panCamera(dx: number, dy: number): void {
    const three = this.THREE;
    const camera = this.camera;
    const target = this.target;
    if (!three || !camera || !target) {
      return;
    }
    camera.updateMatrixWorld();
    const right = new three.Vector3().setFromMatrixColumn(camera.matrix, 0);
    const up = new three.Vector3().setFromMatrixColumn(camera.matrix, 1);
    const panScale = this.radius * 0.0015;
    target.addScaledVector(right, -dx * panScale);
    target.addScaledVector(up, dy * panScale);
    this.updateCamera();
  }

  private updateCamera(): void {
    const camera = this.camera;
    const target = this.target;
    if (!camera || !target) {
      return;
    }
    camera.position.set(
      target.x + this.radius * Math.sin(this.phi) * Math.sin(this.theta),
      target.y + this.radius * Math.cos(this.phi),
      target.z + this.radius * Math.sin(this.phi) * Math.cos(this.theta),
    );
    camera.lookAt(target);
  }

  private drawScene(): void {
    const renderer = this.renderer;
    const scene = this.scene;
    const camera = this.camera;
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  }

  private statusLine(): string {
    switch (this.phase) {
      case 'idle':
        return 'Prêt — « Un pas » ou « Une batch »';
      case 'present': {
        const example = this.currentExample;
        if (!example) {
          return '';
        }
        const total = this.core.batch.length;
        return (
          `Exemple ${this.exampleNumber}/${total} : ` +
          `(x₁, x₂) = (${formatNumber(example.x1)} ; ${formatNumber(example.x2)}) ` +
          `→ t = ${formatNumber(example.t)}`
        );
      }
      case 'gradient':
        return 'Flèche : sens de descente pour cet exemple (opposé du gradient)';
      case 'fade':
        return 'Flèche archivée en rouge translucide et surface effacée';
      case 'mean':
        return 'Moyenne des flèches = sens de descente de la batch';
      case 'move':
        return 'La balle descend : position -= learning rate × gradient de batch';
    }
  }

  protected override renderHeader(): TemplateResult {
    return html`
      <div class="drl-status">
        <span class="drl-badge">${PHASE_LABELS[this.phase]}</span>
        <span class="drl-status-text">${this.statusLine()}</span>
      </div>
      <div class="drl-toolbar" role="group" aria-label="Contrôles de la descente de gradient">
        <button type="button" class="drl-button" @click=${this.handleSingleStep}>
          Un pas
        </button>
        <button
          type="button"
          class="drl-button"
          ?disabled=${this.running || this.animating}
          @click=${this.handleBatch}
        >
          Une batch
        </button>
      </div>
      <div class="drl-stats">
        <div class="drl-stat">
          <p class="drl-stat-label">Époque</p>
          <p class="drl-stat-value">${this.epoch}</p>
        </div>
        <div class="drl-stat">
          <p class="drl-stat-label">Position (a, b)</p>
          <p class="drl-stat-value">
            ${formatNumber(this.ballA)}, ${formatNumber(this.ballB)}
          </p>
        </div>
        <div class="drl-stat">
          <p class="drl-stat-label">Loss</p>
          <p class="drl-stat-value">${formatNumber(this.lossValue)}</p>
        </div>
        <div class="drl-stat">
          <p class="drl-stat-label">|gradient|</p>
          <p class="drl-stat-value">${formatNumber(this.gradMag)}</p>
        </div>
      </div>
    `;
  }

  protected override renderContent(): TemplateResult {
    if (this.webglError) {
      return html`<p class="drl-webgl-error">WebGL n'est pas disponible dans cet environnement.</p>`;
    }
    return html`
      <canvas
        width=${CANVAS_WIDTH}
        height=${CANVAS_HEIGHT}
        aria-label="Surface de loss 3D et descente de gradient par batch"
      ></canvas>
      <p class="drl-hint">
        Glissez pour tourner, clic molette + glisser pour déplacer, molette pour zoomer.
      </p>
    `;
  }

  override disconnectedCallback(): void {
    if (this.animFrame !== null) {
      cancelAnimationFrame(this.animFrame);
      this.animFrame = null;
    }
    const canvas = this.canvasEl;
    if (canvas) {
      canvas.removeEventListener('pointerdown', this.onPointerDown);
      canvas.removeEventListener('pointerup', this.onPointerUp);
      canvas.removeEventListener('pointerleave', this.onPointerUp);
      canvas.removeEventListener('pointermove', this.onPointerMove);
      canvas.removeEventListener('auxclick', this.onAuxClick);
      canvas.removeEventListener('wheel', this.onWheel);
    }

    const disposeMaterial = (material: ThreeMaterial | null): void => {
      material?.dispose();
    };
    this.landscapeGeometry?.dispose();
    disposeMaterial(this.landscapeMaterial);
    this.patchGeometry?.dispose();
    disposeMaterial(this.patchMaterial);
    if (this.ball) {
      this.ball.geometry.dispose();
      disposeMaterial(this.ball.material as ThreeMaterial);
    }
    this.trailGeometry?.dispose();
    disposeMaterial(this.trailMaterial);
    for (const arrow of [this.solidArrow, this.meanArrow, ...this.collectedArrows]) {
      if (!arrow) {
        continue;
      }
      disposeMaterial(arrow.line.material as ThreeMaterial);
      disposeMaterial(arrow.cone.material as ThreeMaterial);
    }
    this.renderer?.dispose();

    this.THREE = null;
    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.canvasEl = null;
    this.landscapeGeometry = null;
    this.landscapeMaterial = null;
    this.patchGeometry = null;
    this.patchMaterial = null;
    this.patchMesh = null;
    this.patchLowColor = null;
    this.patchHighColor = null;
    this.ball = null;
    this.trailGeometry = null;
    this.trailMaterial = null;
    this.trailAttribute = null;
    this.trailPositions = null;
    this.gradientGroup = null;
    this.solidArrow = null;
    this.meanArrow = null;
    this.collectedArrows = [];
    this.tmpDir = null;
    this.target = null;
    super.disconnectedCallback();
  }
}
