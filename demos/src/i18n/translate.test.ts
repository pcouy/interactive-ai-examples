import { describe, expect, it } from 'vitest';
import { gameBoardMessage } from '../components/drl-game-board/messages.js';
import { uiMessage } from '../ui/messages.js';
import { type Catalog, translate } from './translate.js';

type TestKey = 'hello' | 'onlyFr' | 'greet' | 'count' | 'mixed';

const CATALOG: Catalog<TestKey> = {
  fr: {
    hello: 'Bonjour',
    onlyFr: 'Seulement',
    greet: 'Bonjour {name}',
    count: '{count} éléments',
    mixed: 'Bonjour {name}, {unknown} reste inconnu',
  },
  en: {
    hello: 'Hello',
    onlyFr: 'Only',
    greet: 'Hello {name}',
    count: '{count} items',
    mixed: 'Hello {name}, {unknown} stays unknown',
  },
};

describe('translate', () => {
  it('returns the French entry for the French locale', () => {
    expect(translate('fr', CATALOG, 'hello')).toBe('Bonjour');
  });

  it('returns the English entry for the English locale', () => {
    expect(translate('en', CATALOG, 'hello')).toBe('Hello');
  });

  it('falls back to the French entry when the selected locale is missing the key', () => {
    const partial = {
      fr: { onlyFr: 'Seulement' },
      en: {},
    } as unknown as Catalog<TestKey>;
    expect(translate('en', partial, 'onlyFr')).toBe('Seulement');
  });

  it('returns the key when both locales are missing it', () => {
    const partial = {
      fr: {},
      en: {},
    } as unknown as Catalog<TestKey>;
    expect(translate('en', partial, 'hello')).toBe('hello');
  });

  it('replaces string placeholders from the params', () => {
    expect(translate('fr', CATALOG, 'greet', { name: 'Ada' })).toBe('Bonjour Ada');
    expect(translate('en', CATALOG, 'greet', { name: 'Ada' })).toBe('Hello Ada');
  });

  it('replaces number placeholders from the params', () => {
    expect(translate('fr', CATALOG, 'count', { count: 3 })).toBe('3 éléments');
    expect(translate('en', CATALOG, 'count', { count: 0 })).toBe('0 items');
  });

  it('leaves unknown placeholders unchanged', () => {
    expect(translate('en', CATALOG, 'mixed', { name: 'Ada' })).toBe(
      'Hello Ada, {unknown} stays unknown',
    );
  });

  it('leaves placeholders unchanged when no params are given', () => {
    expect(translate('fr', CATALOG, 'greet')).toBe('Bonjour {name}');
  });
});

describe('shared catalogs', () => {
  it('localizes the shared control labels', () => {
    expect(uiMessage('fr', 'play')).toBe('Lecture');
    expect(uiMessage('fr', 'pause')).toBe('Pause');
    expect(uiMessage('fr', 'step')).toBe('Pas');
    expect(uiMessage('fr', 'reset')).toBe('Réinitialiser');
    expect(uiMessage('fr', 'speed')).toBe('Vitesse');
    expect(uiMessage('fr', 'sliderValueFallback')).toBe('valeur');
    expect(uiMessage('en', 'play')).toBe('Play');
    expect(uiMessage('en', 'step')).toBe('Step');
    expect(uiMessage('en', 'reset')).toBe('Reset');
    expect(uiMessage('en', 'speed')).toBe('Speed');
    expect(uiMessage('en', 'sliderValueFallback')).toBe('value');
  });

  it('localizes the game board aria label', () => {
    expect(gameBoardMessage('fr', 'ariaLabel', { rows: 3, cols: 4 })).toBe('Plateau de jeu 3×4');
    expect(gameBoardMessage('en', 'ariaLabel', { rows: 3, cols: 4 })).toBe('Game board 3×4');
  });
});
