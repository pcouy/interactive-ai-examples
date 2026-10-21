import assert from 'node:assert/strict';
import test from 'node:test';

import { rewriteHtml, rewriteUrl } from './lib/url-rewrite.mjs';

test('rewriteUrl maps the root href to index.html', () => {
  assert.equal(rewriteUrl('/'), './index.html');
});

test('rewriteUrl appends .html to an extensionless path', () => {
  assert.equal(rewriteUrl('/deep-blue'), './deep-blue.html');
});

test('rewriteUrl keeps an existing file extension', () => {
  assert.equal(rewriteUrl('/deep-blue.html'), './deep-blue.html');
  assert.equal(rewriteUrl('/demo/drl-three.js'), './demo/drl-three.js');
  assert.equal(rewriteUrl('/katex/katex.min.css'), './katex/katex.min.css');
});

test('rewriteUrl keeps query strings and fragments', () => {
  assert.equal(rewriteUrl('/deep-blue?mode=demo#board'), './deep-blue.html?mode=demo#board');
  assert.equal(rewriteUrl('/demo/drl-demos.js?v=2'), './demo/drl-demos.js?v=2');
  assert.equal(rewriteUrl('/#main'), './index.html#main');
});

test('rewriteUrl leaves external, fragment, and data URLs alone', () => {
  assert.equal(rewriteUrl('https://example.com/page'), 'https://example.com/page');
  assert.equal(rewriteUrl('http://example.com/page'), 'http://example.com/page');
  assert.equal(rewriteUrl('//cdn.example.com/app.js'), '//cdn.example.com/app.js');
  assert.equal(rewriteUrl('mailto:hello@example.com'), 'mailto:hello@example.com');
  assert.equal(rewriteUrl('tel:+33123456789'), 'tel:+33123456789');
  assert.equal(rewriteUrl('#main'), '#main');
  assert.equal(rewriteUrl('data:image/png;base64,AAAA'), 'data:image/png;base64,AAAA');
  assert.equal(rewriteUrl(''), '');
});

test('rewriteUrl never appends .html to a src value', () => {
  assert.equal(rewriteUrl('/demo/drl-demos', 'src'), './demo/drl-demos');
  assert.equal(rewriteUrl('/', 'src'), './');
});

test('rewriteHtml rewrites double-quoted href and src values', () => {
  const html = [
    '<a href="/">Accueil</a>',
    '<a href="/deep-blue">Deep Blue</a>',
    '<a href="/frozen-lake.html">Frozen Lake</a>',
    '<script src="/demo/drl-demos.js"></script>',
    '<link rel="stylesheet" href="/katex/katex.min.css">',
  ].join('\n');

  assert.equal(
    rewriteHtml(html),
    [
      '<a href="./index.html">Accueil</a>',
      '<a href="./deep-blue.html">Deep Blue</a>',
      '<a href="./frozen-lake.html">Frozen Lake</a>',
      '<script src="./demo/drl-demos.js"></script>',
      '<link rel="stylesheet" href="./katex/katex.min.css">',
    ].join('\n'),
  );
});

test('rewriteHtml rewrites single-quoted attributes', () => {
  const html = "<a href='/deep-blue'>Démo</a><img src='/demo/board.svg'>";
  assert.equal(
    rewriteHtml(html),
    "<a href='./deep-blue.html'>Démo</a><img src='./demo/board.svg'>",
  );
});

test('rewriteHtml rewrites CSS url() values but keeps quotes', () => {
  const html = [
    '<style>.a{background:url(/demo/board.svg)}</style>',
    "<style>.b{background:url('/demo/board.svg')}</style>",
    '<style>.c{background:url("/katex/fonts/KaTeX_Main-Regular.woff2")}</style>',
  ].join('');

  assert.equal(
    rewriteHtml(html),
    [
      '<style>.a{background:url(./demo/board.svg)}</style>',
      "<style>.b{background:url('./demo/board.svg')}</style>",
      '<style>.c{background:url("./katex/fonts/KaTeX_Main-Regular.woff2")}</style>',
    ].join(''),
  );
});

test('rewriteHtml leaves unsafe and external values untouched', () => {
  const html = [
    '<a href="https://example.com">x</a>',
    '<a href="//cdn.example.com/app.js">x</a>',
    '<a href="mailto:hello@example.com">x</a>',
    '<a href="#main">x</a>',
    '<img src="data:image/png;base64,AAAA">',
    '<style>.a{background:url(//cdn.example.com/a.png)}</style>',
  ].join('');

  assert.equal(rewriteHtml(html), html);
});

test('rewriteHtml does not touch inline script bodies', () => {
  const html = [
    '<script>',
    'element.href = "/deep-blue";',
    'const background = "url(/x)";',
    '</script>',
  ].join('\n');

  assert.equal(rewriteHtml(html), html);
});

test('rewriteHtml ignores prefixed href attributes', () => {
  const html = '<div data-href="/x"></div><svg><use xlink:href="/x"></use></svg>';
  assert.equal(rewriteHtml(html), html);
});

test('rewriteHtml rewrites normal href and src attributes', () => {
  const html = '<a href="/x">x</a><img src="/x.js">';
  assert.equal(rewriteHtml(html), '<a href="./x.html">x</a><img src="./x.js">');
});

test('rewriteHtml rewrites the src of an external script tag', () => {
  const html = '<script src="/x.js"></script>';
  assert.equal(rewriteHtml(html), '<script src="./x.js"></script>');
});

test('rewriteHtml rewrites url() inside a style block', () => {
  const html = '<style>.a{background:url(/x)}</style>';
  assert.equal(rewriteHtml(html), '<style>.a{background:url(./x)}</style>');
});

