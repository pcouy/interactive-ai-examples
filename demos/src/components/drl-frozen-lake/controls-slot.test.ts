import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const COMPONENT_DIR = dirname(fileURLToPath(import.meta.url));
const DEMO_ROOT = resolve(COMPONENT_DIR, '../../..');

const COMPONENT_SOURCE = readFileSync(resolve(COMPONENT_DIR, 'drl-frozen-lake.ts'), 'utf8');
const PAGE_SOURCE = readFileSync(
  resolve(DEMO_ROOT, 'public_site/src/pages/frozen-lake.astro'),
  'utf8',
);

function indexOfOrFail(value: string, needle: string): number {
  const index = value.indexOf(needle);
  expect(index, `missing ${needle}`).toBeGreaterThan(-1);
  return index;
}

describe('frozen lake controls slot', () => {
  it('renders the default slot under the reward curve in the main column', () => {
    const mainStart = indexOfOrFail(COMPONENT_SOURCE, '<div class="drl-main">');
    const asideStart = indexOfOrFail(COMPONENT_SOURCE, '<aside class="drl-side">');
    const curveIndex = indexOfOrFail(COMPONENT_SOURCE, 'class="drl-curve-svg"');
    const slotIndex = indexOfOrFail(COMPONENT_SOURCE, '<slot></slot>');

    expect(slotIndex).toBeGreaterThan(mainStart);
    expect(slotIndex).toBeGreaterThan(curveIndex);
    expect(slotIndex).toBeLessThan(asideStart);
  });

  it('keeps the seven sliders inside the demo element', () => {
    const openIndex = indexOfOrFail(PAGE_SOURCE, '<drl-frozen-lake ');
    const closeIndex = indexOfOrFail(PAGE_SOURCE, '</drl-frozen-lake>');
    expect(closeIndex).toBeGreaterThan(openIndex);

    const inner = PAGE_SOURCE.slice(openIndex, closeIndex);
    expect(inner).toContain('demo-controls');
    expect((inner.match(/data-param=/g) ?? []).length).toBe(7);
  });

  it('keeps the replay panel after the demo element', () => {
    const closeIndex = indexOfOrFail(PAGE_SOURCE, '</drl-frozen-lake>');
    const replayIndex = indexOfOrFail(PAGE_SOURCE, '<drl-frozen-lake-replay>');

    expect(replayIndex).toBeGreaterThan(closeIndex);
  });
});
