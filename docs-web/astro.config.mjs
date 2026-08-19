// @ts-check
import { defineConfig } from 'astro/config'
import starlight from '@astrojs/starlight'
import { autoImportComponents } from './plugins/auto-import-components.mjs'
import { baseInternalLinks } from './plugins/base-internal-links.mjs'

// The docs live at sushantgundla.com/context-grid, which is a path on the personal site rather
// than a host of its own, so `base` has to be set here and every internal link has to survive
// it. `trailingSlash: 'never'` keeps the URLs identical to the ones Mintlify served, so nothing
// that already links to a page breaks on the move.
const BASE = '/context-grid'

export default defineConfig({
  site: 'https://sushantgundla.com',
  base: BASE,
  trailingSlash: 'never',
  // Applies to MDX as well as Markdown: @astrojs/mdx inherits `markdown.remarkPlugins`.
  markdown: {
    remarkPlugins: [autoImportComponents],
    rehypePlugins: [[baseInternalLinks, { base: BASE }]],
  },
  integrations: [
    starlight({
      title: 'context-grid',
      description:
        'A lab for grounding pipelines: sweep parser x chunker x embedder x index x reranker ' +
        'over your own documents and score every combination on quality, latency and cost.',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/sushantgundla/context-grid',
        },
      ],
      // Mirrors docs-site/docs.json, which stays the source of truth for Mintlify while both
      // sites are up. `scripts/sync-navigation.mjs` regenerates this file from it.
      sidebar: (await import('./src/sidebar.mjs')).sidebar,
    }),
  ],
})
