import { css, unsafeCSS } from 'lit';
import { drlPalette, drlTokenValues } from './token-values.js';

// Every fallback below is a module constant, never user input, so unsafeCSS is
// safe here. The `css` tag only accepts CSSResultGroup | number.
export const drlTokens = css`
  :host {
    --drl-bg: var(--drl-theme-bg, ${unsafeCSS(drlPalette.light.bg)});
    --drl-surface: var(--drl-theme-surface, ${unsafeCSS(drlPalette.light.surface)});
    --drl-fg: var(--drl-theme-fg, ${unsafeCSS(drlPalette.light.fg)});
    --drl-muted: var(--drl-theme-muted, ${unsafeCSS(drlPalette.light.muted)});
    --drl-accent: var(--drl-theme-accent, ${unsafeCSS(drlPalette.light.accent)});
    --drl-accent-2: var(--drl-theme-accent-2, ${unsafeCSS(drlPalette.light['accent-2'])});
    --drl-arrow: var(--drl-theme-arrow, ${unsafeCSS(drlPalette.light.arrow)});
    --drl-border: var(--drl-theme-border, ${unsafeCSS(drlPalette.light.border)});
    --drl-fg-2: var(--drl-theme-fg-2, ${unsafeCSS(drlPalette.light['fg-2'])});
    --drl-muted-2: var(--drl-theme-muted-2, ${unsafeCSS(drlPalette.light['muted-2'])});
    --drl-warn: var(--drl-theme-warn, ${unsafeCSS(drlPalette.light.warn)});
    --drl-danger: var(--drl-theme-danger, ${unsafeCSS(drlPalette.light.danger)});
    --drl-danger-2: var(--drl-theme-danger-2, ${unsafeCSS(drlPalette.light['danger-2'])});
    --drl-radius: ${unsafeCSS(drlTokenValues.radius)};
    --drl-font: ${unsafeCSS(drlTokenValues.font)};
    --drl-font-size: ${unsafeCSS(drlTokenValues.fontSize)};
    --drl-space-1: ${unsafeCSS(drlTokenValues.space1)};
    --drl-space-2: ${unsafeCSS(drlTokenValues.space2)};
    --drl-space-3: ${unsafeCSS(drlTokenValues.space3)};
  }
`;
