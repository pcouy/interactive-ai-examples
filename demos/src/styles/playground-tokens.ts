import { drlPalette } from './token-values.js';

const TOKENS_ELEMENT_ID = 'drl-playground-tokens';

function themeVariables(palette: Readonly<Record<string, string>>): string {
  return Object.entries(palette)
    .map(([name, value]) => `  --drl-theme-${name}: ${value};`)
    .join('\n');
}

/** Light and dark theme variables for the playground pages. */
export function playgroundTokensCss(): string {
  return [
    ':root {',
    '  color-scheme: light;',
    themeVariables(drlPalette.light),
    '}',
    '',
    ":root[data-drl-theme='dark'] {",
    '  color-scheme: dark;',
    themeVariables(drlPalette.dark),
    '}',
    '',
  ].join('\n');
}

function installPlaygroundTokens(): void {
  if (typeof document === 'undefined') {
    return;
  }
  if (document.getElementById(TOKENS_ELEMENT_ID) !== null) {
    return;
  }
  const style = document.createElement('style');
  style.id = TOKENS_ELEMENT_ID;
  style.textContent = playgroundTokensCss();
  document.head.prepend(style);
}

installPlaygroundTokens();
