import { DEFAULT_LOCALE, type Locale } from './locale.js';

export type Catalog<Key extends string> = Record<Locale, Record<Key, string>>;

export type TranslateParams = Record<string, string | number>;

const PLACEHOLDER = /\{([^{}]+)\}/g;

function replacePlaceholder(
  match: string,
  name: string,
  params: TranslateParams | undefined,
): string {
  if (params === undefined || !Object.hasOwn(params, name)) {
    return match;
  }
  return String(params[name]);
}

/** Returns the selected locale entry, the French fallback, or the raw key. */
export function translate<Key extends string>(
  locale: Locale,
  catalog: Catalog<Key>,
  key: Key,
  params?: TranslateParams,
): string {
  const message = catalog[locale]?.[key] ?? catalog[DEFAULT_LOCALE][key] ?? key;
  if (params === undefined) {
    return message;
  }
  return message.replace(PLACEHOLDER, (match, name: string) =>
    replacePlaceholder(match, name, params),
  );
}
