import { css, unsafeCSS } from 'lit';
import { drlTokenValues } from './token-values.js';

// The `css` tag only accepts CSSResultGroup | number, so each token value is
// wrapped with unsafeCSS. The emitted CSS text is identical to the literals.
export const drlTokens = css`
  :host {
    --drl-bg: ${unsafeCSS(drlTokenValues.bg)};
    --drl-surface: ${unsafeCSS(drlTokenValues.surface)};
    --drl-fg: ${unsafeCSS(drlTokenValues.fg)};
    --drl-muted: ${unsafeCSS(drlTokenValues.muted)};
    --drl-accent: ${unsafeCSS(drlTokenValues.accent)};
    --drl-accent-2: ${unsafeCSS(drlTokenValues.accent2)};
    --drl-arrow: ${unsafeCSS(drlTokenValues.arrow)};
    --drl-border: ${unsafeCSS(drlTokenValues.border)};
    --drl-radius: ${unsafeCSS(drlTokenValues.radius)};
    --drl-font: ${unsafeCSS(drlTokenValues.font)};
    --drl-font-size: ${unsafeCSS(drlTokenValues.fontSize)};
    --drl-space-1: ${unsafeCSS(drlTokenValues.space1)};
    --drl-space-2: ${unsafeCSS(drlTokenValues.space2)};
    --drl-space-3: ${unsafeCSS(drlTokenValues.space3)};
  }
`;
