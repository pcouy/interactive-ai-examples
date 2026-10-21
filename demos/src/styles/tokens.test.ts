import { describe, expect, it } from 'vitest';
import { playgroundTokensCss } from './playground-tokens.js';
import { drlPalette, drlTokenValues } from './token-values.js';
import { drlTokens } from './tokens.js';

const COLOUR_KEYS = [
  'bg',
  'surface',
  'fg',
  'muted',
  'accent',
  'accent-2',
  'arrow',
  'border',
  'fg-2',
  'muted-2',
  'warn',
  'danger',
  'danger-2',
] as const;

const HEX_COLOUR = /^#[0-9a-f]{6}$/;

describe('drlTokens', () => {
  it('emits every colour token through a --drl-theme-* fallback', () => {
    for (const key of COLOUR_KEYS) {
      expect(drlTokens.cssText).toContain(
        `--drl-${key}: var(--drl-theme-${key}, ${drlPalette.light[key]});`,
      );
    }
  });

  it('keeps the non-colour tokens as direct values', () => {
    expect(drlTokens.cssText).toContain('--drl-radius: 10px;');
    expect(drlTokens.cssText).toContain(`--drl-font: ${drlTokenValues.font};`);
    expect(drlTokens.cssText).toContain('--drl-font-size: 14px;');
    expect(drlTokens.cssText).toContain('--drl-space-1: 0.25rem;');
    expect(drlTokens.cssText).toContain('--drl-space-2: 0.5rem;');
    expect(drlTokens.cssText).toContain('--drl-space-3: 1rem;');
    expect(drlTokens.cssText).not.toContain('--drl-theme-radius');
  });
});

describe('drlPalette', () => {
  it('reuses the shared token values in the light palette', () => {
    expect(drlPalette.light.bg).toBe(drlTokenValues.bg);
    expect(drlPalette.light.surface).toBe(drlTokenValues.surface);
    expect(drlPalette.light.fg).toBe(drlTokenValues.fg);
    expect(drlPalette.light.muted).toBe(drlTokenValues.muted);
    expect(drlPalette.light.accent).toBe(drlTokenValues.accent);
    expect(drlPalette.light['accent-2']).toBe(drlTokenValues.accent2);
    expect(drlPalette.light.arrow).toBe(drlTokenValues.arrow);
    expect(drlPalette.light.border).toBe(drlTokenValues.border);
  });

  it('declares the same 13 colour keys in both themes with 6-digit hex values', () => {
    expect(Object.keys(drlPalette.light)).toHaveLength(COLOUR_KEYS.length);
    expect(drlPalette.light).toHaveProperty('fg-2');
    expect(Object.keys(drlPalette.light).sort()).toEqual(Object.keys(drlPalette.dark).sort());
    for (const key of COLOUR_KEYS) {
      expect(drlPalette.light[key]).toMatch(HEX_COLOUR);
      expect(drlPalette.dark[key]).toMatch(HEX_COLOUR);
    }
  });
});

describe('playgroundTokensCss', () => {
  it('emits both theme blocks with every palette value', () => {
    const cssText = playgroundTokensCss();
    expect(cssText).toContain(':root {');
    expect(cssText).toContain(":root[data-drl-theme='dark'] {");
    expect(cssText).toContain('color-scheme: light;');
    expect(cssText).toContain('color-scheme: dark;');
    for (const key of COLOUR_KEYS) {
      expect(cssText).toContain(`--drl-theme-${key}: ${drlPalette.light[key]};`);
      expect(cssText).toContain(`--drl-theme-${key}: ${drlPalette.dark[key]};`);
    }
  });

  it('is deterministic', () => {
    expect(playgroundTokensCss()).toBe(playgroundTokensCss());
  });
});
