// Regenerates src/sidebar.mjs from ../docs-site/docs.json.
//
// Both sites read the same MDX, so they must also agree on the order and grouping of it. Rather
// than keep a second navigation by hand and let the two drift -- which is how `/reference/reports`
// stayed missing for two releases without anyone noticing -- this derives one from the other.
// Run it whenever docs.json changes; `npm run build` runs it first.
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const here = fileURLToPath(new URL('.', import.meta.url))
const config = JSON.parse(readFileSync(new URL('../../docs-site/docs.json', import.meta.url)))

// Starlight addresses the root page as the empty slug; Mintlify calls it `index`.
const toSlug = (page) => (page === 'index' ? '' : page)

const sidebar = config.navigation.groups.map((group) => ({
  label: group.group,
  items: group.pages.map((page) => ({ slug: toSlug(page) })),
}))

writeFileSync(
  new URL('../src/sidebar.mjs', import.meta.url),
  [
    '// Generated from ../docs-site/docs.json by scripts/sync-navigation.mjs. Do not hand-edit:',
    '// docs.json stays the one place navigation is decided while Mintlify is still serving.',
    `export const sidebar = ${JSON.stringify(sidebar, null, 2)}`,
    '',
  ].join('\n')
)

const pages = sidebar.reduce((total, group) => total + group.items.length, 0)
console.log(`sidebar: ${sidebar.length} groups, ${pages} pages`)
