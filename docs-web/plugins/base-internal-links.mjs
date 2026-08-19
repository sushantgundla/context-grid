import { visit } from 'unist-util-visit'

// The pages link to each other with root-absolute paths -- `[Installation](/installation)` --
// because on Mintlify the docs own the whole host. Here they live under /context-grid on a
// domain that also serves a personal site, so an unprefixed `/installation` leaves the docs
// entirely and lands on a page that does not exist. 219 links are affected.
//
// Rewriting them in the MDX would fix this site and break Mintlify, which is still serving. So
// the prefix is added at build time instead, and the source keeps working for both.
//
// Every root-absolute link is prefixed, without checking it names a real page: the `nav` job in
// .github/workflows/docs.yml already fails the build when an internal link points at a page that
// does not exist, so by the time this runs, they all do.
export function baseInternalLinks({ base }) {
  const prefix = base.replace(/\/$/, '')

  return function rewrite(tree) {
    visit(tree, 'element', (node) => {
      if (node.tagName !== 'a') return
      const href = node.properties?.href
      if (typeof href !== 'string') return

      // `//example.com` is protocol-relative and external, despite starting with a slash.
      if (!href.startsWith('/') || href.startsWith('//')) return
      if (href === prefix || href.startsWith(`${prefix}/`)) return

      node.properties.href = prefix + href
    })
  }
}
