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

function rewriteCssUrl(match, quote, value) {
  if (!value.startsWith('/') || value.startsWith('//')) {
    return match;
  }
  return `url(${quote}${rewriteUrl(value, 'asset')}${quote})`;
}

function rewriteChunk(chunk) {
  return chunk
    .replace(
      /(\s)(href|src)\s*=\s*(["'])(\/(?!\/)[^"']*)\3/gi,
      (_match, space, name, quote, value) => {
        const kind = name.toLowerCase() === 'href' ? 'href' : 'src';
        return `${space}${name}=${quote}${rewriteUrl(value, kind)}${quote}`;
      },
    )
    .replace(/url\(\s*(["']?)(\/(?!\/)[^"')]*?)\1\s*\)/gi, rewriteCssUrl);
}

/**
 * Rewrite root-absolute `href`, `src`, and CSS `url()` values in an HTML
 * string so the static build works from any directory (including file://).
 * Inline script bodies stay untouched because they contain JavaScript, not
 * markup. The opening tag of an external script is still rewritten.
 */
export function rewriteHtml(html) {
  const parts = html.split(/(<script\b[^>]*>[\s\S]*?<\/script>)/i);

  return parts
    .map((part, index) => {
      if (index % 2 === 1) {
        return part.replace(/^<script\b[^>]*>/i, (tag) => rewriteChunk(tag));
      }
      return rewriteChunk(part);
    })
    .join('');
}
