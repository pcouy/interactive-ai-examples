/**
 * Raw CSS token values shared by the Lit stylesheet (`./tokens.ts`) and
 * framework-agnostic modules (e.g. the Three.js landscape scene) that cannot
 * import Lit but still need the values at runtime. Keep in sync with the
 * literals interpolated in `./tokens.ts`.
 */
export const drlTokenValues = {
  bg: '#f8fafc',
  surface: '#ffffff',
  fg: '#0f172a',
  muted: '#64748b',
  accent: '#2563eb',
  accent2: '#7c3aed',
  arrow: '#d32f2f',
  border: '#e2e8f0',
  radius: '10px',
  font: "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  fontSize: '14px',
  space1: '0.25rem',
  space2: '0.5rem',
  space3: '1rem',
} as const;
