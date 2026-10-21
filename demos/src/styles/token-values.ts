/**
 * This file is the only place in `demos/` that may contain colour literals.
 * `drlTokenValues` holds the light values shared by the Lit stylesheet
 * (`./tokens.ts`) and framework-agnostic modules (e.g. the Three.js landscape
 * scene) that cannot import Lit. `drlPalette` adds the dark theme used by the
 * playground pages.
 */
export const drlTokenValues = {
  bg: '#f8fafc',
  surface: '#ffffff',
  fg: '#0f172a',
  muted: '#64748b',
  accent: '#2563eb',
  accent2: '#c81e3a',
  arrow: '#d32f2f',
  border: '#e2e8f0',
  radius: '10px',
  font: "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  fontSize: '14px',
  space1: '0.25rem',
  space2: '0.5rem',
  space3: '1rem',
} as const;

/**
 * Colour palette for both site themes. The light side reuses `drlTokenValues`
 * so the component stylesheet and the playground can never drift apart.
 */
export const drlPalette = {
  light: {
    bg: drlTokenValues.bg,
    surface: drlTokenValues.surface,
    fg: drlTokenValues.fg,
    muted: drlTokenValues.muted,
    accent: drlTokenValues.accent,
    'accent-2': drlTokenValues.accent2,
    arrow: drlTokenValues.arrow,
    border: drlTokenValues.border,
    'fg-2': '#334155',
    'muted-2': '#94a3b8',
    warn: '#b45309',
    danger: '#b91c1c',
    'danger-2': '#dc2626',
  },
  dark: {
    bg: '#0b1220',
    surface: '#111a2b',
    fg: '#e2e8f0',
    muted: '#94a3b8',
    accent: '#60a5fa',
    'accent-2': '#fb7185',
    arrow: '#f87171',
    border: '#1e293b',
    'fg-2': '#cbd5e1',
    'muted-2': '#8b98ad',
    warn: '#fbbf24',
    danger: '#fca5a5',
    'danger-2': '#f87171',
  },
} as const;
