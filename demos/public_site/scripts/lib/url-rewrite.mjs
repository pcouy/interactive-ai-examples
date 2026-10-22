import path from 'node:path';

function splitSuffix(value) {
  const match = /^([^?#]*)([?#].*)?$/.exec(value);
  return {
    path: match?.[1] ?? value,
    suffix: match?.[2] ?? '',
  };
}

function hasFileExtension(pathname) {
  const lastSegment = pathname.slice(pathname.lastIndexOf('/') + 1);
  return lastSegment.includes('.');
}

function withDotSlash(relative) {
  if (relative === '') return './';
  return relative.startsWith('.') || relative.startsWith('/') ? relative : `./${relative}`;
}

/**
 * Rewrite one root-absolute URL. `kind` is `"href"` for navigable links and
 * `"src"`/`"asset"` for resources. `fromDir` is the dist-relative directory of
 * the page being rewritten (`""` for root pages, `"en"` for nested pages).
 * Only a single leading slash is rewritable: protocol-relative URLs,
 * fragments, and URLs with a scheme stay untouched.
 */
export function rewriteUrl(value, kind = 'href', fromDir = '') {
  if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//')) {
    return value;
  }

  const { path: urlPath, suffix } = splitSuffix(value);
  const baseDir = fromDir || '.';

  if (kind !== 'href') {
    const targetPath = urlPath.slice(1);
    if (targetPath === '') {
      const relative = path.posix.relative(baseDir, '.');
      return relative === '' ? `./${suffix}` : `${relative}/${suffix}`;
    }

    const relative = path.posix.relative(baseDir, targetPath);
    const rewritten = urlPath.endsWith('/') ? `${relative}/` : relative;
    return `${withDotSlash(rewritten)}${suffix}`;
  }

  if (urlPath === '' || urlPath === '/') {
    return `${withDotSlash(path.posix.relative(baseDir, 'index.html'))}${suffix}`;
  }

  let target = urlPath.slice(1);
  if (urlPath.endsWith('/')) {
    target = `${target}index.html`;
  } else if (!hasFileExtension(target)) {
    target = `${target}.html`;
  }

  return `${withDotSlash(path.posix.relative(baseDir, target))}${suffix}`;
}

function rewriteCssUrl(match, quote, value, fromDir) {
  if (!value.startsWith('/') || value.startsWith('//')) {
    return match;
  }
  return `url(${quote}${rewriteUrl(value, 'asset', fromDir)}${quote})`;
}

function rewriteChunk(chunk, fromDir) {
  return chunk
    .replace(
      /(\s)(href|src)\s*=\s*(["'])(\/(?!\/)[^"']*)\3/gi,
      (_match, space, name, quote, value) => {
        const kind = name.toLowerCase() === 'href' ? 'href' : 'src';
        return `${space}${name}=${quote}${rewriteUrl(value, kind, fromDir)}${quote}`;
      },
    )
    .replace(/url\(\s*(["']?)(\/(?!\/)[^"')]*?)\1\s*\)/gi, (match, quote, value) =>
      rewriteCssUrl(match, quote, value, fromDir),
    );
}

/**
 * Rewrite root-absolute `href`, `src`, and CSS `url()` values in an HTML
 * string so the static build works from any directory (including file://).
 * Inline script bodies stay untouched because they contain JavaScript, not
 * markup. The opening tag of an external script is still rewritten.
 */
export function rewriteHtml(html, fromDir = '') {
  const parts = html.split(/(<script\b[^>]*>[\s\S]*?<\/script>)/i);

  return parts
    .map((part, index) => {
      if (index % 2 === 1) {
        return part.replace(/^<script\b[^>]*>/i, (tag) => rewriteChunk(tag, fromDir));
      }
      return rewriteChunk(part, fromDir);
    })
    .join('');
}
