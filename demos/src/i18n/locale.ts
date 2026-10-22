export type Locale = 'fr' | 'en';

export const DEFAULT_LOCALE: Locale = 'fr';

/** Accepts `fr`, `en`, and their BCP 47 variants, for example `en-US` or `fr_CA`. */
export function normalizeLocale(value: unknown): Locale | null {
  if (typeof value !== 'string' || !/^(?:fr|en)(?:[-_][A-Za-z0-9]+)*$/i.test(value)) {
    return null;
  }
  return value.slice(0, 2).toLowerCase() as Locale;
}
