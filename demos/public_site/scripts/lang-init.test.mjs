import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const script = readFileSync(new URL('../src/scripts/lang-init.js', import.meta.url), 'utf8');

/** Minimal element: `closest` returns a language link only for a known value. */
class FakeElement {
  constructor(lang) {
    this.lang = lang;
  }

  closest(selector) {
    if (selector !== '[data-set-lang]' || typeof this.lang !== 'string') {
      return null;
    }
    return {
      getAttribute: (name) => (name === 'data-set-lang' ? this.lang : null),
    };
  }
}

/**
 * Runs the language init script with fake browser globals.
 * `href` drives the default `pathname`, `search`, and `hash` values.
 */
function createPage(options = {}) {
  const replacements = [];
  const writes = [];
  const listeners = new Map();
  const href = options.href ?? 'https://example.test/deep-blue';
  const url = new URL(href);

  const context = vm.createContext({
    document: {
      documentElement: {
        lang: options.lang ?? 'fr',
        dataset: {
          langFr: options.langFr ?? '/deep-blue',
          langEn: options.langEn ?? '/en/deep-blue',
        },
      },
      addEventListener(type, listener) {
        listeners.set(type, listener);
      },
    },
    localStorage: {
      getItem() {
        if (options.storageError) {
          throw new Error('storage unavailable');
        }
        return options.stored ?? null;
      },
      setItem(key, value) {
        writes.push({ key, value });
      },
    },
    navigator: {
      languages: options.languages,
      language: options.language ?? 'fr-FR',
    },
    location: {
      href,
      pathname: options.pathname ?? url.pathname,
      search: options.search ?? url.search,
      hash: options.hash ?? url.hash,
      replace(target) {
        replacements.push(target);
      },
    },
    URL,
    Element: FakeElement,
  });

  vm.runInContext(script, context);

  const dispatch = (type, target, button = 0) => {
    const listener = listeners.get(type);
    assert.ok(listener, `the script must listen for ${type}`);
    listener({ target, button });
  };

  return {
    replacements,
    writes,
    dispatch,
    click: (target) => dispatch('click', target),
    auxclick: (target, button) => dispatch('auxclick', target, button),
  };
}

test('a stored English choice redirects a French page', () => {
  const page = createPage({ stored: 'en', lang: 'fr' });
  assert.deepEqual(page.replacements, ['/en/deep-blue']);
});

test('a stored French choice redirects an English page', () => {
  const page = createPage({
    stored: 'fr',
    lang: 'en',
    href: 'https://example.test/en/deep-blue',
  });
  assert.deepEqual(page.replacements, ['/deep-blue']);
});

test('the browser language redirects a French page to English', () => {
  const page = createPage({ lang: 'fr', languages: ['en-US'] });
  assert.deepEqual(page.replacements, ['/en/deep-blue']);
});

test('the browser language redirects an English page to French', () => {
  const page = createPage({
    lang: 'en',
    href: 'https://example.test/en/deep-blue',
    languages: ['fr-FR'],
  });
  assert.deepEqual(page.replacements, ['/deep-blue']);
});

test('a matching language and stored value do not redirect', () => {
  const page = createPage({
    stored: 'en',
    lang: 'en',
    href: 'https://example.test/en/deep-blue',
  });
  assert.deepEqual(page.replacements, []);
});

test('the redirect keeps the query and the hash', () => {
  const page = createPage({
    stored: 'en',
    lang: 'fr',
    href: 'https://example.test/deep-blue?mode=demo#board',
  });
  assert.deepEqual(page.replacements, ['/en/deep-blue?mode=demo#board']);
});

test('a relative alternate target redirects from the current page', () => {
  const page = createPage({
    stored: 'fr',
    lang: 'en',
    href: 'https://example.test/nested/site/en/deep-blue.html',
    langFr: '../deep-blue.html',
    langEn: './deep-blue.html',
  });
  assert.deepEqual(page.replacements, ['../deep-blue.html']);
});

test('a relative alternate target of the current language does not redirect', () => {
  const page = createPage({
    stored: 'en',
    lang: 'en',
    href: 'https://example.test/nested/site/en/deep-blue.html',
    langFr: '../deep-blue.html',
    langEn: './deep-blue.html',
  });
  assert.deepEqual(page.replacements, []);
});

test('a page without an alternate target does not redirect', () => {
  const page = createPage({ stored: 'en', lang: 'fr', langEn: '' });
  assert.deepEqual(page.replacements, []);
});

test('an unavailable storage does not crash the script', () => {
  const page = createPage({ storageError: true, lang: 'fr', languages: ['fr-FR'] });
  assert.deepEqual(page.replacements, []);
});

test('a click on a language link stores the chosen language', () => {
  const page = createPage({ stored: 'fr', lang: 'fr' });
  page.click(new FakeElement('en'));
  assert.deepEqual(page.writes, [{ key: 'drl-lang', value: 'en' }]);
});

test('a click outside a language link does not store anything', () => {
  const page = createPage({ stored: 'fr', lang: 'fr' });
  page.click(new FakeElement(undefined));
  assert.deepEqual(page.writes, []);
});

test('a middle click on a language link stores the chosen language', () => {
  const page = createPage({ stored: 'fr', lang: 'fr' });
  page.auxclick(new FakeElement('fr'), 1);
  assert.deepEqual(page.writes, [{ key: 'drl-lang', value: 'fr' }]);
});

test('a right click on a language link does not store anything', () => {
  const page = createPage({ stored: 'fr', lang: 'fr' });
  page.auxclick(new FakeElement('fr'), 2);
  assert.deepEqual(page.writes, []);
});
