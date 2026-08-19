import { parse as parseJs } from 'acorn'
import { fileURLToPath } from 'node:url'

// The 38 pages under ../docs-site were written for Mintlify, which supplies components like
// <Note> and <ParamField> ambiently -- no page imports anything. Adding an import block to each
// one would work, but it would also stop them building on Mintlify, and both sites have to keep
// working until the move is finished.
//
// So the imports are injected here instead, into every MDX document as it is parsed. This is
// what `astro-auto-import` does; it is inlined because that package bails out unless it finds
// `@astrojs/mdx` already in the integrations array, and Starlight adds MDX itself, later.
//
// Paths are absolute because the documents live outside `srcDir`, so a relative specifier would
// resolve against ../docs-site and find nothing.
const componentDir = fileURLToPath(new URL('../src/components/', import.meta.url))

const COMPONENTS = [
  'Note',
  'Tip',
  'Warning',
  'ParamField',
  'ResponseField',
  'Accordion',
  'AccordionGroup',
  'Card',
  'CardGroup',
  'Steps',
  'Step',
  'Tabs',
  'Tab',
]

const source = COMPONENTS.map(
  (name) => `import ${name} from ${JSON.stringify(`${componentDir}${name}.astro`)};`
).join('\n')

const importsNode = {
  type: 'mdxjsEsm',
  value: '',
  data: {
    estree: {
      body: [],
      ...parseJs(source, { ecmaVersion: 'latest', sourceType: 'module' }),
      type: 'Program',
      sourceType: 'module',
    },
  },
}

export function autoImportComponents() {
  return function injectImports(tree, file) {
    // Plain Markdown has no JSX, and an ESM node in it is a syntax error rather than a no-op.
    if (file.basename?.endsWith('.md')) return
    tree.children.unshift(importsNode)
  }
}
