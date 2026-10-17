import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const plopDir = path.dirname(fileURLToPath(import.meta.url));
const NAME_PATTERN = /^[a-z][a-z0-9]*(-[a-z0-9]+)*$/;
const RESERVED = new Set(['demo', 'controls', 'slider', 'card', 'base', 'core', 'ui', 'styles']);

function toKebab(value) {
  return String(value)
    .trim()
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .toLowerCase()
    .replace(/^-+|-+$/g, '');
}

function isNameAvailable(name) {
  if (!NAME_PATTERN.test(name) || RESERVED.has(name)) {
    return false;
  }
  return !existsSync(path.join(plopDir, 'src', 'components', `drl-${name}`));
}

function readCli(argv) {
  const { values, positionals } = parseArgs({
    args: argv,
    options: {},
    strict: false,
    allowPositionals: true,
  });
  const threeD = values['3d'];
  return {
    rawName: positionals[0],
    is3d: threeD === true || threeD === 'true' || threeD === '',
  };
}

function buildActions(name, is3d) {
  const componentTemplate = is3d ? 'component-3d.hbs' : 'component.hbs';
  const tag = is3d ? '3D' : '2D';
  const data = { name, is3d };
  return [
    {
      type: 'add',
      path: `src/components/drl-${name}/drl-${name}.ts`,
      templateFile: `plop-templates/${componentTemplate}`,
      data,
    },
    {
      type: 'add',
      path: `src/core/${name}.ts`,
      templateFile: 'plop-templates/core.hbs',
      data,
    },
    {
      type: 'add',
      path: `src/core/${name}.test.ts`,
      templateFile: 'plop-templates/core.test.hbs',
      data,
    },
    {
      type: 'add',
      path: `playground/drl-${name}.html`,
      templateFile: 'plop-templates/page.hbs',
      data,
    },
    {
      type: 'modify',
      path: 'src/index.ts',
      pattern: /(\/\/ @demos-imports)/,
      template: `import './components/drl-${name}/drl-${name}.js';\n\n$1`,
      data,
    },
    {
      type: 'modify',
      path: 'playground/index.html',
      pattern: /(<!-- @demos-cards -->)/,
      template:
        `<a class="demo-card" href="./drl-${name}.html">\n` +
        `          <h3>drl-${name}</h3>\n` +
        `          <span class="demo-tag">${tag} demo</span>\n` +
        `        </a>\n` +
        '        $1',
      data,
    },
  ];
}

export default function (plop) {
  plop.setGenerator('demo', {
    description: 'Create a new DRL demo (2D SVG/D3 or 3D three.js).',
    prompts: [
      {
        type: 'input',
        name: 'name',
        message: 'Demo name (kebab-case):',
        validate: (input) => {
          const name = toKebab(input ?? '');
          if (!name) {
            return 'Enter a demo name.';
          }
          if (!NAME_PATTERN.test(name)) {
            return 'Use kebab-case, for example "grid-world".';
          }
          if (RESERVED.has(name)) {
            return `"${name}" is a reserved name.`;
          }
          if (!isNameAvailable(name)) {
            return `src/components/drl-${name}/ already exists.`;
          }
          return true;
        },
      },
      {
        type: 'confirm',
        name: 'is3d',
        message: '3D demo (three.js)?',
        default: false,
      },
    ],
    actions: (answers) => buildActions(toKebab(answers.name), Boolean(answers.is3d)),
  });

  const { rawName, is3d } = readCli(process.argv.slice(2));
  if (!rawName) {
    return;
  }

  const name = toKebab(rawName);
  if (!NAME_PATTERN.test(name) || RESERVED.has(name)) {
    console.error(`[demos] invalid name "${rawName}": use kebab-case, for example "grid-world".`);
    process.exit(1);
  }
  if (!isNameAvailable(name)) {
    console.error(`[demos] src/components/drl-${name}/ already exists.`);
    process.exit(1);
  }

  plop.setGenerator(rawName, {
    description: `Create the "${name}" demo.`,
    prompts: [],
    actions: () => buildActions(name, is3d),
  });
}
