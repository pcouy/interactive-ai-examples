import assert from 'node:assert/strict';
import test from 'node:test';

import {
  absoluteUrl,
  outputFileForRoute,
  pageFromPath,
  relativeHref,
  routeFor,
  ui,
} from '../src/i18n/site.ts';

test('routeFor maps every language and page to its route', () => {
  assert.equal(routeFor('fr', 'home'), '/');
  assert.equal(routeFor('en', 'home'), '/en/');
  assert.equal(routeFor('fr', 'deep-blue'), '/deep-blue');
  assert.equal(routeFor('en', 'deep-blue'), '/en/deep-blue');
  assert.equal(routeFor('fr', 'frozen-lake'), '/frozen-lake');
  assert.equal(routeFor('en', 'frozen-lake'), '/en/frozen-lake');
});

test('pageFromPath maps every supported URL shape', () => {
  const cases = [
    ['/', 'home'],
    ['/index.html', 'home'],
    ['/en/', 'home'],
    ['/en/index.html', 'home'],
    ['/deep-blue', 'deep-blue'],
    ['/deep-blue.html', 'deep-blue'],
    ['/en/deep-blue', 'deep-blue'],
    ['/en/deep-blue.html', 'deep-blue'],
    ['/en/deep-blue/', 'deep-blue'],
    ['/unknown-page', 'home'],
  ];

  for (const [pathname, page] of cases) {
    assert.equal(pageFromPath(pathname), page, pathname);
  }
});

test('outputFileForRoute maps a route to its build file', () => {
  assert.equal(outputFileForRoute('/'), 'index.html');
  assert.equal(outputFileForRoute('/en/'), 'en/index.html');
  assert.equal(outputFileForRoute('/deep-blue'), 'deep-blue.html');
  assert.equal(outputFileForRoute('/en/deep-blue'), 'en/deep-blue.html');
});

test('relativeHref links pages from the current page directory', () => {
  assert.equal(relativeHref('/', '/en/'), './en/index.html');
  assert.equal(relativeHref('/', '/en/deep-blue'), './en/deep-blue.html');
  assert.equal(relativeHref('/en/', '/'), '../index.html');
  assert.equal(relativeHref('/en/deep-blue', '/deep-blue'), '../deep-blue.html');
  assert.equal(relativeHref('/en/deep-blue', '/en/'), './index.html');
  assert.equal(relativeHref('/en/deep-blue', '/en/deep-blue'), './deep-blue.html');
  assert.equal(relativeHref('/', '/'), './index.html');
});

test('absoluteUrl prefixes the site base path', () => {
  const site = new URL('https://example.test/interactive-ai-examples');

  assert.equal(absoluteUrl(site, '/'), 'https://example.test/interactive-ai-examples/');
  assert.equal(absoluteUrl(site, '/en/'), 'https://example.test/interactive-ai-examples/en/');
  assert.equal(
    absoluteUrl(site, '/deep-blue'),
    'https://example.test/interactive-ai-examples/deep-blue',
  );
  assert.equal(
    absoluteUrl(site, '/en/deep-blue'),
    'https://example.test/interactive-ai-examples/en/deep-blue',
  );
});

/** Fails with the full key path when the two catalogs differ. */
function assertMatchingKeys(frValue, enValue, path) {
  assert.deepEqual(Object.keys(enValue).sort(), Object.keys(frValue).sort(), `${path} keys`);

  for (const key of Object.keys(frValue)) {
    const frEntry = frValue[key];
    const enEntry = enValue[key];
    if (typeof frEntry === 'object' && frEntry !== null) {
      assert.equal(typeof enEntry, 'object', `${path}.${key} must be an object in English`);
      assert.notEqual(enEntry, null, `${path}.${key} must be an object in English`);
      assertMatchingKeys(frEntry, enEntry, `${path}.${key}`);
    } else {
      assert.equal(typeof enEntry, 'string', `${path}.${key} must be a string in English`);
    }
  }
}

test('the French and English site catalogs have the same keys', () => {
  assertMatchingKeys(ui.fr, ui.en, 'ui');
});
