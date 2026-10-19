import type { Selection } from 'd3';
import { select } from 'd3';
import type { TemplateResult } from 'lit';
import { css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { DemoElement } from '../../base/demo-element.js';
import { FrozenLakeCore } from '../../core/frozen-lake.js';
import { drlTokens } from '../../styles/tokens.js';

const BAR_COUNT = 6;

@customElement('drl-frozen-lake')
export class FrozenLake extends DemoElement {
  static override styles = [
    drlTokens,
    css`
      :host {
        display: block;
      }

      svg {
        display: block;
        width: 100%;
        max-width: 480px;
        height: auto;
        background: var(--drl-surface);
        border: 1px solid var(--drl-border);
        border-radius: var(--drl-radius);
      }
    `,
  ];

  @property({ type: Number, attribute: 'learning-rate' })
  learningRate = 0.1;

  private readonly core = new FrozenLakeCore();

  private bars: Selection<SVGRectElement, number, SVGSVGElement, unknown> | null = null;

  // TODO: replace this placeholder D3 binding with the real demo visualization.
  protected override firstUpdated(): void {
    const svg = this.renderRoot.querySelector('svg');
    if (!svg) {
      return;
    }
    this.bars = select(svg)
      .selectAll<SVGRectElement, number>('rect')
      .data(Array.from({ length: BAR_COUNT }, (_, index) => index))
      .join('rect')
      .attr('x', (_, index) => 40 + index * 68)
      .attr('width', 44)
      .attr('rx', 6)
      .attr('fill', 'var(--drl-accent)');
    this.drawBars();
  }

  override step(): void {
    this.core.tick(this.learningRate);
    this.drawBars();
  }

  protected override onReset(): void {
    this.core.reset();
    this.drawBars();
  }

  private drawBars(): void {
    this.bars
      ?.attr('y', (index) => 320 - this.core.valueAt(index) * 260)
      .attr('height', (index) => this.core.valueAt(index) * 260);
  }

  protected override renderContent(): TemplateResult {
    return html`
      <svg viewBox="0 0 480 360" role="img" aria-label="frozen-lake demo"></svg>
    `;
  }
}
