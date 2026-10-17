import type { TemplateResult } from 'lit';
import { css, html, LitElement, nothing } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { drlTokens } from '../styles/tokens.js';

@customElement('drl-slider')
export class DrlSlider extends LitElement {
  static styles = [
    drlTokens,
    css`
      :host {
        display: block;
        color: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: var(--drl-font-size);
      }

      .drl-slider {
        display: flex;
        align-items: center;
        gap: var(--drl-space-2);
      }

      .drl-slider-label {
        color: var(--drl-muted);
        white-space: nowrap;
      }

      input[type='range'] {
        accent-color: var(--drl-accent);
        flex: 1;
      }

      .drl-slider-value {
        min-width: 3ch;
        text-align: right;
        font-variant-numeric: tabular-nums;
      }
    `,
  ];

  @property({ type: String })
  label = '';

  @property({ type: Number })
  min = 0;

  @property({ type: Number })
  max = 1;

  @property({ type: Number })
  step = 0.01;

  @property({ type: Number })
  value = 0;

  private readonly onInput = (event: Event): void => {
    const target = event.target as HTMLInputElement;
    this.value = Number(target.value);
    this.dispatchEvent(
      new CustomEvent('drl-input', { detail: this.value, bubbles: true, composed: true }),
    );
  };

  protected override render(): TemplateResult {
    return html`
      <label class="drl-slider">
        ${this.label ? html`<span class="drl-slider-label">${this.label}</span>` : nothing}
        <input
          type="range"
          min=${this.min}
          max=${this.max}
          step=${this.step}
          .value=${String(this.value)}
          aria-label=${this.label || 'value'}
          @input=${this.onInput}
        />
        <span class="drl-slider-value">${this.value}</span>
      </label>
    `;
  }
}
