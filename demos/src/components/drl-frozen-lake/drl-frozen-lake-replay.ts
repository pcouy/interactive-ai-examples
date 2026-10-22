import type { TemplateResult } from 'lit';
import { css, html, nothing } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { LocalizedElement } from '../../base/localized-element.js';
import { drlTokens } from '../../styles/tokens.js';
import {
  type FrozenLakeReplayState,
  type ReplayRow,
  replayHeader,
  replayRows,
} from './frozen-lake-ui.js';
import { replayMessage } from './replay-messages.js';

@customElement('drl-frozen-lake-replay')
export class FrozenLakeReplay extends LocalizedElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: block;
        color: var(--drl-fg);
        font-family: var(--drl-font);
        font-size: var(--drl-font-size);
      }

      .drl-replay {
        margin-bottom: var(--drl-space-2);
      }

      .drl-replay-header {
        margin: 0 0 var(--drl-space-1);
        color: var(--drl-muted);
        font-size: 0.85em;
        font-variant-numeric: tabular-nums;
      }

      .drl-replay-list {
        display: grid;
        gap: var(--drl-space-1);
        margin: 0;
        padding: 0;
        list-style: none;
      }

      .drl-replay-row {
        font-size: 0.9em;
        font-variant-numeric: tabular-nums;
      }

      .drl-replay-row--last {
        color: var(--drl-accent);
        font-weight: 600;
      }

      .drl-replay-row--muted {
        color: var(--drl-muted);
      }
    `,
  ];

  @property({ attribute: false })
  replayState: FrozenLakeReplayState | null = null;

  private renderRow(row: ReplayRow): TemplateResult {
    const className = row.newest ? 'drl-replay-row drl-replay-row--last' : 'drl-replay-row';
    return html`<li class=${className}>${row.text}</li>`;
  }

  protected override render(): TemplateResult | typeof nothing {
    const state = this.replayState;
    if (state === null || !state.enabled) {
      return nothing;
    }
    const rows = replayRows(state, this.locale);
    return html`
      <section class="drl-replay" aria-label=${replayMessage(this.locale, 'ariaLabel')}>
        <p class="drl-replay-header">${replayHeader(state, this.locale)}</p>
        <ul class="drl-replay-list">
          ${
            rows.length === 0
              ? html`
                <li class="drl-replay-row drl-replay-row--muted">
                  ${replayMessage(this.locale, 'empty')}
                </li>
              `
              : rows.map((row) => this.renderRow(row))
          }
        </ul>
      </section>
    `;
  }
}
