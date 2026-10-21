function splitSuffix(value) {
  const match = /^([^?#]*)([?#].*)?$/.exec(value);
  return {
    path: match?.[1] ?? value,
    suffix: match?.[2] ?? '',
  };
}

function hasFileExtension(path) {
  const lastSegment = path.slice(path.lastIndexOf('/') + 1);
  return lastSegment.includes('.');
}

/**
 * Rewrite one root-absolute URL. `kind` is `"href"` for navigable links and
 * `"src"`/`"asset"` for resources. Only a single leading slash is rewritable:
 * protocol-relative URLs, fragments, and URLs with a scheme stay untouched.
 */
export function rewriteUrl(value, kind = 'href') {
  if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//')) {
    return value;
  }

  const { path, suffix } = splitSuffix(value);

  if (kind !== 'href') {
    return `./${path.slice(1)}${suffix}`;
  }

  if (path === '/' || path === '') {
    return `./index.html${suffix}`;
  }

  let target = path.replace(/\/+$/, '');
  if (target === '') {
    return `./index.html${suffix}`;
  }

  if (!hasFileExtension(target)) {
    target = `${target}.html`;
  }

  return `./${target.slice(1)}${suffix}`;
}

function rewriteAttribute(_match, prefix, quote, value) {
  const kind = /href/i.test(prefix) ? 'href' : 'src';
  return `${prefix}${quote}${rewriteUrl(value, kind)}${quote}`;
}

function rewriteCssUrl(match, quote, value) {
  if (!value.startsWith('/') || value.startsWith('//')) {
    return match;
  }
  return `url(${quote}${rewriteUrl(value, 'asset')}${quote})`;
}

/**
 * Rewrite root-absolute `href`, `src`, and CSS `url()` values in an HTML
 * string so the static build works from any directory (including file://).
 */
export function rewriteHtml(html) {
  return html
    .replace(
      /(\b(?:href|src)\s*=\s*)(["'])([^"']*)\2/gi,
      rewriteAttribute,
    )
    .replace(
      /url\(\s*(["']?)(\/(?!\/)[^"')]*?)\1\s*\)/gi,
      rewriteCssUrl,
    );
}
