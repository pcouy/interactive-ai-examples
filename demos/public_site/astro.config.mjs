import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// Astro 7 uses Sätteri by default. `unified()` opts the Markdown pipeline back
// into remark/rehype so the demos can share the KaTeX setup used elsewhere.
export default defineConfig({
  // PUBLIC_SITE_URL overrides the default GitHub Pages URL for a custom domain.
  site: process.env.PUBLIC_SITE_URL ?? 'https://pcouy.github.io/interactive-ai-examples',
  output: 'static',
  trailingSlash: 'never',
  build: {
    format: 'file',
    inlineStylesheets: 'always',
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: false }]],
    }),
  },
});
