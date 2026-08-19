import { defineCollection } from 'astro:content'
import { glob } from 'astro/loaders'
import { docsSchema } from '@astrojs/starlight/schema'

// Starlight's own `docsLoader()` hard-codes `src/content/docs`, so it cannot read pages that
// live anywhere else. Pointing Astro's glob loader at `../docs-site` instead keeps one copy of
// the MDX: Mintlify builds it from the repository, and so does this, with no sync step and no
// chance of the two drifting.
export const collections = {
  docs: defineCollection({
    loader: glob({ base: '../docs-site', pattern: '**/[^_]*.mdx' }),
    schema: docsSchema(),
  }),
}
