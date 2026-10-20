import type { TemplateResult } from 'lit';
import { css, html, LitElement } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { drlTokens } from '../styles/tokens.js';
import './drl-slider.js';

@customElement('drl-controls')
export class DrlControls extends LitElement {
  static styles = [
    drlTokens,
    css`
      :host {
        display: block;
        color: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: var(--drl-font-size);
      }

      .drl-controls {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--drl-space-2);
        padding: var(--drl-space-2);
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
      }

      button {
        font: inherit;
        padding: var(--drl-space-1) var(--drl-space-2);
        border: 1px solid var(--drl-border);
        border-radius: calc(var(--drl-radius) / 2);
        background: var(--drl-bg);
        color: var(--drl-fg);
        cursor: pointer;
      }

      button:hover:not(:disabled) {
        border-color: var(--drl-accent);
      }

      button:disabled {
        color: var(--drl-muted);
        cursor: not-allowed;
      }

      button.drl-primary {
        background: var(--drl-accent);
        border-color: var(--drl-accent);
        color: var(--drl-surface);
      }

      drl-slider {
        flex: 1;
        min-width: 12rem;
      }
    `,
  ];

  @property({ type: Boolean, reflect: true })
  running = false;

  @property({ type: Number, reflect: true })
  speed = 1;

  private emit(name: string, detail?: number): void {
    this.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: true }));
  }

  private readonly onToggle = (): void => {
    this.running = !this.running;
    this.emit(this.running ? 'drl-play' : 'drl-pause');
  };

  private readonly onStep = (): void => {
    this.emit('drl-step');
  };

  private readonly onReset = (): void => {
    this.emit('drl-reset');
  };

  private readonly onSpeedInput = (event: CustomEvent<number>): void => {
    this.speed = event.detail;
    this.emit('drl-speed', event.detail);
  };

  protected override render(): TemplateResult {
    return html`
      <div class="drl-controls" part="controls">
        <button
          type="button"
          class="drl-primary"
          aria-label=${this.running ? 'Pause' : 'Play'}
          @click=${this.onToggle}
        >
          ${this.running ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          aria-label="Step"
          ?disabled=${this.running}
          @click=${this.onStep}
        >
          Step
        </button>
        <button type="button" aria-label="Reset" @click=${this.onReset}>Reset</button>
        <drl-slider
          label="Speed"
          min="0.01"
          max="4"
          step="0.01"
          .value=${this.speed}
          @drl-input=${this.onSpeedInput}
        ></drl-slider>
      </div>
    `;
  }
}
