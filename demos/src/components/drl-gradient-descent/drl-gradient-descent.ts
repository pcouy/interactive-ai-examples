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
import { createLandscapeScene, type LandscapeScene } from '../shared/landscape-scene.js';

type ThreeModule = typeof import('three');
type ThreeMesh = InstanceType<ThreeModule['Mesh']>;
type ThreeGroup = InstanceType<ThreeModule['Group']>;
type ThreeColor = InstanceType<ThreeModule['Color']>;
type ThreeVector3 = InstanceType<ThreeModule['Vector3']>;
type ThreeBufferGeometry = InstanceType<ThreeModule['BufferGeometry']>;
type ThreeArrowHelper = InstanceType<ThreeModule['ArrowHelper']>;
type ThreeMaterial = InstanceType<ThreeModule['Material']>;
type ThreeMeshStandardMaterial = InstanceType<ThreeModule['MeshStandardMaterial']>;

const DOMAIN = GradientDescentCore.DOMAIN;
/**
 * The shared landscape module normalizes the loss field so the visible surface
 * spans [0, 1.5] in scene Y. The component works in raw (a, b) coordinates and
 * asks the module for the matching scene height.
 */
const BALL_RADIUS = 0.14;

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

  private landscape: LandscapeScene | null = null;

  private patchGeometry: ThreeBufferGeometry | null = null;

  private patchMaterial: ThreeMeshStandardMaterial | null = null;

  private patchMesh: ThreeMesh | null = null;

  private patchExample: Example | null = null;

  private patchLowColor: ThreeColor | null = null;

  private patchHighColor: ThreeColor | null = null;

  private patchOpacity = 0;

  private patchTarget = 0;

  private gradientGroup: ThreeGroup | null = null;

  private solidArrow: ThreeArrowHelper | null = null;

  private meanArrow: ThreeArrowHelper | null = null;

  private collectedArrows: ThreeArrowHelper[] = [];

  private tmpDir: ThreeVector3 | null = null;

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
      const landscape = await createLandscapeScene(canvas);
      if (!this.isConnected) {
        landscape.dispose();
        return;
      }
      this.THREE = three;
      this.landscape = landscape;
      this.tmpDir = new three.Vector3();
      landscape.setHeightField(landscapeLoss, DOMAIN);

      const accentTwo = this.readColor(three, '--drl-accent-2', '#ffffff');
      const arrowColor = this.readColor(three, '--drl-arrow', ARROW_COLOR_FALLBACK);
      this.buildPatchMesh(three, accentTwo);
      this.buildArrows(three, arrowColor);

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

  private readColor(three: ThreeModule, name: string, fallback: string): ThreeColor {
    const value = getComputedStyle(this).getPropertyValue(name).trim();
    return new three.Color(value.length > 0 ? value : fallback);
  }

  private buildPatchMesh(three: ThreeModule, accentTwo: ThreeColor): void {
    const scene = this.landscape?.scene;
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

  private buildArrows(three: ThreeModule, arrowColor: ThreeColor): void {
    const scene = this.landscape?.scene;
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
    const landscape = this.landscape;
    if (!three || !geometry || !landscape) {
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
    const baseY = landscape.heightAt(baseA, baseB);

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

  private updateVisuals(): void {
    const three = this.THREE;
    if (!three) {
      return;
    }
    if (!this.animating) {
      this.landscape?.moveBall(this.core.ballA, this.core.ballB);
    }

    const example = this.core.currentExample;
    if (example !== this.patchExample) {
      this.patchExample = example;
      if (example) {
        this.buildPatch(example);
      }
    }

    this.patchTarget = this.animating ? 0 : this.patchTargetFor(this.core.phase);
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
    if (this.animating) {
      this.hideAllArrows();
      return;
    }
    const landscape = this.landscape;
    if (!landscape) {
      this.hideAllArrows();
      return;
    }
    const core = this.core;
    const phase = core.phase;
    const baseA = core.ballA;
    const baseB = core.ballB;
    const originY = landscape.heightAt(baseA, baseB) + BALL_RADIUS;
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

  override step(): void {
    if (this.animating) {
      return;
    }
    this.core.advance(this.learningRate, { detailed: !this.running });
    if (this.core.phase === 'move') {
      this.landscape?.pushTrail(this.core.ballA, this.core.ballB);
    }
    this.sync();
  }

  private readonly handleSingleStep = (): void => {
    if (!this.running) {
      this.step();
    }
  };

  private readonly handleBatch = (): void => {
    if (this.running || this.animating) {
      return;
    }
    const move = this.core.advanceBatch(this.learningRate);
    if (move && this.landscape) {
      this.animating = true;
      this.landscape.tweenBall({ w1: move.toA, w2: move.toB }, () => {
        this.landscape?.pushTrail(move.toA, move.toB);
        this.animating = false;
      });
    }
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
    this.landscape?.clearTween();
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
    this.landscape?.clearTrail();
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
    this.landscape?.update(delta);
    if (this.animating) {
      busy = true;
    }
    this.updateVisuals();
    if (Math.abs(this.patchOpacity - this.patchTarget) > 0.01) {
      busy = true;
    }
    this.landscape?.render();
    if (busy) {
      this.requestRender();
    }
  };

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
    const disposeMaterial = (material: ThreeMaterial | null): void => {
      material?.dispose();
    };
    this.patchGeometry?.dispose();
    disposeMaterial(this.patchMaterial);
    for (const arrow of [this.solidArrow, this.meanArrow, ...this.collectedArrows]) {
      if (!arrow) {
        continue;
      }
      disposeMaterial(arrow.line.material as ThreeMaterial);
      disposeMaterial(arrow.cone.material as ThreeMaterial);
    }
    this.landscape?.dispose();
    this.landscape = null;

    this.THREE = null;
    this.patchGeometry = null;
    this.patchMaterial = null;
    this.patchMesh = null;
    this.patchLowColor = null;
    this.patchHighColor = null;
    this.gradientGroup = null;
    this.solidArrow = null;
    this.meanArrow = null;
    this.collectedArrows = [];
    this.tmpDir = null;
    super.disconnectedCallback();
  }
}
