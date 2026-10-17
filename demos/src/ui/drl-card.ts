import type { TemplateResult } from 'lit';
import { css, html, LitElement, nothing } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { drlTokens } from '../styles/tokens.js';

@customElement('drl-card')
export class DrlCard extends LitElement {
  static styles = [
    drlTokens,
    css`
      :host {
        display: block;
        color: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: var(--drl-font-size);
      }

      .drl-card {
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
        padding: var(--drl-space-3);
        box-shadow: 0 1px 2px rgb(15 23 42 / 6%);
      }

      .drl-card-heading {
        margin: 0 0 var(--drl-space-2);
        font-size: 1.05em;
      }
    `,
  ];

  @property({ type: String })
  heading = '';

  protected override render(): TemplateResult {
    return html`
      <article class="drl-card" part="card">
        ${
          this.heading
            ? html`<h3 class="drl-card-heading" part="heading">${this.heading}</h3>`
            : nothing
        }
        <div class="drl-card-body"><slot></slot></div>
      </article>
    `;
  }
}
