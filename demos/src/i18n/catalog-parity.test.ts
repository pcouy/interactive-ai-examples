import { describe, expect, it } from 'vitest';

declare global {
  interface ImportMeta {
    glob(pattern: string, options: { eager: true }): Record<string, unknown>;
  }
}

type StringRecord = Record<string, string>;

interface DiscoveredCatalog {
  modulePath: string;
  exportName: string;
  fr: StringRecord;
  en: StringRecord;
}

/** Narrows a value to a plain object whose own values are all strings. */
function isStringRecord(value: unknown): value is StringRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }
  return Object.values(value).every((entry) => typeof entry === 'string');
}

/** Narrows a value to a translation catalog with `fr` and `en` string records. */
function isCatalog(value: unknown): value is { fr: StringRecord; en: StringRecord } {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const { fr, en } = value as { fr?: unknown; en?: unknown };
  return isStringRecord(fr) && isStringRecord(en);
}

const PLACEHOLDER = /\{[^{}]+\}/g;

/** Returns the sorted, unique placeholders of one message. */
function placeholders(message: string): string[] {
  return [...new Set(message.match(PLACEHOLDER) ?? [])].sort();
}

const modules = import.meta.glob('../**/*messages.ts', { eager: true });

const catalogs: DiscoveredCatalog[] = Object.entries(modules).flatMap(([modulePath, module]) =>
  Object.entries(module as Record<string, unknown>).flatMap(([exportName, value]) =>
    isCatalog(value) ? [{ modulePath, exportName, fr: value.fr, en: value.en }] : [],
  ),
);

describe('catalog parity', () => {
  it('discovers every messages.ts module', () => {
    const modulePaths = new Set(catalogs.map((catalog) => catalog.modulePath));
    expect(catalogs.length, 'no catalog export was discovered').toBeGreaterThan(0);
    expect(modulePaths.size, 'a messages.ts module exports no catalog').toBe(
      Object.keys(modules).length,
    );
  });

  it.each(catalogs)(
    'keeps the French and English keys equal in $modulePath ($exportName)',
    ({ modulePath, exportName, fr, en }) => {
      expect(
        Object.keys(en).sort(),
        `${modulePath} (${exportName}) has French and English keys that differ`,
      ).toEqual(Object.keys(fr).sort());
    },
  );

  it.each(catalogs)(
    'keeps the French and English placeholders equal in $modulePath ($exportName)',
    ({ modulePath, exportName, fr, en }) => {
      for (const key of Object.keys(fr)) {
        const frPlaceholders = placeholders(fr[key]);
        const enPlaceholders = placeholders(en[key] ?? '');
        expect(
          enPlaceholders,
          `${modulePath} (${exportName}) key "${key}" has French placeholders ${JSON.stringify(frPlaceholders)} and English placeholders ${JSON.stringify(enPlaceholders)}`,
        ).toEqual(frPlaceholders);
      }
    },
  );
});
