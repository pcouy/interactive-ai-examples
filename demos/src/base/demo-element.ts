import type { ComplexAttributeConverter, PropertyValues, TemplateResult } from 'lit';
import { css, html, LitElement, nothing } from 'lit';
import { property } from 'lit/decorators.js';
import { drlTokens } from '../styles/tokens.js';

export const BASE_TICKS_PER_SECOND = 30;

const positiveSpeed: ComplexAttributeConverter<number> = {
  fromAttribute: (value) => {
    const parsed = value === null ? 1 : Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
  },
  toAttribute: (value) => String(value),
};

export abstract class DemoElement extends LitElement {
  static styles = [
    drlTokens,
    css`
      :host {
        display: block;
        color: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: var(--drl-font-size);
      }

      .drl-header {
        margin-bottom: var(--drl-space-3);
      }

      .drl-content {
        margin-top: var(--drl-space-3);
      }
    `,
  ];

  @property({ type: Boolean, reflect: true })
  running = false;

  @property({ type: Number, reflect: true, converter: positiveSpeed })
  speed = 1;

  private rafId: number | null = null;

  private lastTime = 0;

  private accumulator = 0;

  abstract step(): void;

  reset(): void {
    this.running = false;
    this.stopLoop();
    this.onReset();
  }

  protected onReset(): void {}

  /** Appelée après le démarrage de l'élément (bouton Play). */
  protected onPlay(): void {}

  private readonly onFrame = (time: number): void => {
    if (!this.running) {
      this.rafId = null;
      return;
    }
    if (this.lastTime === 0) {
      this.lastTime = time;
    }
    const deltaSeconds = (time - this.lastTime) / 1000;
    this.lastTime = time;
    this.accumulator += deltaSeconds * BASE_TICKS_PER_SECOND * this.speed;
    while (this.accumulator >= 1) {
      this.accumulator -= 1;
      this.step();
    }
    this.rafId = requestAnimationFrame(this.onFrame);
  };

  private startLoop(): void {
    if (!this.isConnected || this.rafId !== null) {
      return;
    }
    this.lastTime = 0;
    this.accumulator = 0;
    this.rafId = requestAnimationFrame(this.onFrame);
  }

  private stopLoop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
    }
    this.rafId = null;
    this.accumulator = 0;
  }

  private readonly handlePlay = (): void => {
    this.running = true;
    this.onPlay();
  };

  private readonly handlePause = (): void => {
    this.running = false;
  };

  private readonly handleStep = (): void => {
    if (!this.running) {
      this.step();
    }
  };

  private readonly handleReset = (): void => {
    this.reset();
  };

  private readonly handleSpeed = (event: CustomEvent<number>): void => {
    this.speed = event.detail;
  };

  override connectedCallback(): void {
    super.connectedCallback();
    if (this.running) {
      this.startLoop();
    }
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.stopLoop();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('running')) {
      if (this.running) {
        this.startLoop();
      } else {
        this.stopLoop();
      }
    }
    if (changed.has('speed') && (!Number.isFinite(this.speed) || this.speed <= 0)) {
      this.speed = 1;
    }
  }

  protected override render(): TemplateResult {
    return html`
      ${this.renderHeader()}
      <drl-controls
        .running=${this.running}
        .speed=${this.speed}
        @drl-play=${this.handlePlay}
        @drl-pause=${this.handlePause}
        @drl-step=${this.handleStep}
        @drl-reset=${this.handleReset}
        @drl-speed=${this.handleSpeed}
      ></drl-controls>
      <div class="drl-content">${this.renderContent()}</div>
    `;
  }

  protected renderHeader(): TemplateResult | typeof nothing {
    return nothing;
  }

  protected abstract renderContent(): TemplateResult;
}
