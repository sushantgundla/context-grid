# context-grid docs site

An [Astro Starlight](https://starlight.astro.build) build of the documentation, served at
`https://sushantgundla.com/context-grid`.

## Where the content lives

Nowhere in this directory. The 38 pages are the MDX in `../docs-site`, and the navigation is
`../docs-site/docs.json`. Both are read directly:

- `src/content.config.mjs` points Astro's glob loader at `../docs-site`. Starlight's own
  `docsLoader()` hard-codes `src/content/docs` and cannot read pages kept elsewhere.
- `scripts/sync-navigation.mjs` regenerates `src/sidebar.mjs` from `docs.json`. `npm run build`
  runs it first.

One copy of the content, so this site and Mintlify cannot drift apart while both are up.

## The two build plugins

The pages were written for Mintlify and are still valid Mintlify input. Nothing in `../docs-site`
was changed to make them build here — the differences are handled at build time instead:

- `plugins/auto-import-components.mjs` injects imports for the thirteen components Mintlify
  supplies ambiently (`Note`, `ParamField`, `Card`, …), so no page needs an import line.
- `plugins/base-internal-links.mjs` prefixes root-absolute links with `/context-grid`. The pages
  link to each other as `/installation`, which is correct when the docs own the host and wrong
  here, where they sit on a path beside a personal site. 219 links are affected. `Card` does the
  same for its `href`, which arrives as a prop and never reaches the rehype tree.

## Commands

```bash
npm install
npm run dev      # local server with hot reload
npm run build    # static output in dist/
npm run preview  # serve dist/ exactly as it will be deployed
```

## Components

`src/components/` holds the thirteen. They are deliberately thin, and three are worth knowing
about:

- `ParamField` and `ResponseField` carry 201 of the 372 component uses. `ResponseField` is
  `ParamField` with the prop named `name` instead of `path`.
- `Tabs`/`Tab` render stacked labelled panels rather than a tab strip. Astro cannot enumerate
  slotted children, so real tabs would mean shipping JavaScript or reaching into Starlight's
  internals, and there are six `Tab` uses in the whole site.
