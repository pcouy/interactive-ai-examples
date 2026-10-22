import type { TemplateResult } from 'lit';
import { html } from 'lit';
import { describe, expect, it } from 'vitest';
import { DrlGameBoard } from '../components/drl-game-board/drl-game-board.js';
import type { BoardView } from '../core/contracts.js';
import { normalizeLocale } from '../i18n/locale.js';
import { DrlControls } from '../ui/drl-controls.js';
import { DrlSlider } from '../ui/drl-slider.js';
import { DemoElement } from './demo-element.js';
import { LocalizedElement, localeConverter } from './localized-element.js';

class Probe extends LocalizedElement {
  calls = 0;

  protected override onLocaleChange(): void {
    this.calls += 1;
  }
}

class TestDemo extends DemoElement {
  step(): void {}

  protected override renderContent(): TemplateResult {
    return html`<p>content</p>`;
  }
}

/** No DOM in the node test environment: neutralise Lit's async render pass. */
function preventRender<T extends LocalizedElement>(element: T): T {
  (element as unknown as { performUpdate(): void }).performUpdate = () => {};
  return element;
}

/** Runs the Lit update hook directly, as the reactive cycle would. */
function runUpdated(element: LocalizedElement, property: string): void {
  (element as unknown as { updated(changed: Map<PropertyKey, unknown>): void }).updated(
    new Map([[property, undefined]]),
  );
}

/** Flattens a Lit template into text, including nested templates and values. */
function templateText(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'number') {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value.map(templateText).join('');
  }
  if (typeof value === 'object' && value !== null && 'strings' in value && 'values' in value) {
    const { strings, values } = value as {
      strings: readonly string[];
      values: readonly unknown[];
    };
    let text = '';
    for (let index = 0; index < strings.length; index += 1) {
      text += strings[index];
      if (index < values.length) {
        text += templateText(values[index]);
      }
    }
    return text;
  }
  return '';
}

function renderText(element: LocalizedElement): string {
  const internals = element as unknown as { render(): unknown };
  return templateText(internals.render());
}

describe('LocalizedElement', () => {
  it('defaults to the French locale', () => {
    const probe = preventRender(new Probe());
    expect(probe.locale).toBe('fr');
  });

  it('converts lang attributes and falls back to French for invalid values', () => {
    const convert = localeConverter.fromAttribute;
    expect(convert?.('en')).toBe('en');
    expect(convert?.('fr')).toBe('fr');
    expect(convert?.(null)).toBe('fr');
    expect(convert?.('de')).toBe('fr');
    expect(convert?.('')).toBe('fr');
  });

  it('normalizes BCP 47 tags to the supported locales', () => {
    expect(normalizeLocale('en-US')).toBe('en');
    expect(normalizeLocale('en_US')).toBe('en');
    expect(normalizeLocale('fr-CA')).toBe('fr');
    expect(normalizeLocale('EN')).toBe('en');
    expect(normalizeLocale('de-DE')).toBeNull();
    expect(normalizeLocale('fr-')).toBeNull();
    expect(normalizeLocale('en_')).toBeNull();
    expect(normalizeLocale('fr-_x')).toBeNull();
    expect(normalizeLocale(null)).toBeNull();
  });

  it('calls onLocaleChange when the locale is in the change set', () => {
    const probe = preventRender(new Probe());
    runUpdated(probe, 'other');
    expect(probe.calls).toBe(0);

    probe.locale = 'en';
    runUpdated(probe, 'locale');
    expect(probe.calls).toBe(1);

    runUpdated(probe, 'locale');
    expect(probe.calls).toBe(2);
  });
});

describe('DemoElement localization', () => {
  it('passes its locale to the shared controls', () => {
    const demo = preventRender(new TestDemo());
    expect(renderText(demo)).toContain('lang=fr');

    demo.locale = 'en';
    expect(renderText(demo)).toContain('lang=en');
  });
});

describe('shared UI localization', () => {
  it('renders the controls in French by default and in English after a locale change', () => {
    const controls = preventRender(new DrlControls());
    let text = renderText(controls);
    expect(text).toContain('Lecture');
    expect(text).toContain('Pas');
    expect(text).toContain('Réinitialiser');
    expect(text).toContain('Vitesse');
    expect(text).toContain('lang=fr');

    controls.locale = 'en';
    text = renderText(controls);
    expect(text).toContain('Play');
    expect(text).toContain('Step');
    expect(text).toContain('Reset');
    expect(text).toContain('Speed');
    expect(text).toContain('lang=en');
    expect(text).not.toContain('Lecture');
  });

  it('localizes the slider fallback aria-label', () => {
    const slider = preventRender(new DrlSlider());
    expect(renderText(slider)).toContain('aria-label=valeur');

    slider.locale = 'en';
    expect(renderText(slider)).toContain('aria-label=value');

    slider.label = 'Speed';
    expect(renderText(slider)).toContain('aria-label=Speed');
  });

  it('localizes the game board aria label', () => {
    const board = preventRender(new DrlGameBoard());
    const view: BoardView = {
      rows: 2,
      cols: 3,
      cells: [
        [null, null, null],
        [null, null, null],
      ],
      clickable: [],
      highlight: [],
    };
    board.view = view;
    expect(renderText(board)).toContain('Plateau de jeu 2×3');

    board.locale = 'en';
    expect(renderText(board)).toContain('Game board 2×3');
  });
});
