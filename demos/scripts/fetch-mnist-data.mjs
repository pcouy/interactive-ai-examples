#!/usr/bin/env node
/**
 * Télécharge les quatre archives IDX MNIST dans `demos/public/mnist/`.
 *
 * Les archives ne sont pas versionnées : `npm install` lance ce script via le
 * crochet `postinstall` (mode `--optional`), et `npm run mnist:data` le lance
 * explicitement (mode strict).
 *
 * Le script n'utilise aucune dépendance : `fetch` global, `node:fs/promises` et
 * `node:crypto`. Chaque fichier est vérifié (taille puis MD5) après
 * téléchargement ; un fichier déjà présent avec la bonne taille et le bon MD5
 * est conservé sans être retéléchargé.
 *
 * Modes :
 * - défaut : toute erreur fait sortir avec un code non nul ;
 * - `--optional` : toute erreur affiche un avertissement et sort avec le code 0,
 *   pour ne jamais casser une installation.
 */

import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MNIST_BASE_URL, MNIST_FILES } from './mnist-data-manifest.mjs';

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const TARGET_DIRECTORY = resolve(SCRIPT_DIRECTORY, '..', 'public', 'mnist');
const OPTIONAL = process.argv.includes('--optional');

function md5(bytes) {
  return createHash('md5').update(bytes).digest('hex');
}

/** Vrai si le fichier existe déjà avec la taille et le MD5 attendus. */
async function isAlreadyValid(file) {
  try {
    const bytes = await readFile(join(TARGET_DIRECTORY, file.name));
    return bytes.length === file.size && md5(bytes) === file.md5;
  } catch {
    return false;
  }
}

/** Télécharge, vérifie puis écrit un fichier. Lance une erreur explicite sinon. */
async function download(file) {
  const url = `${MNIST_BASE_URL}${file.name}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`could not download ${url}: HTTP ${response.status} ${response.statusText}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length !== file.size) {
    throw new Error(
      `size mismatch for ${file.name}: expected ${file.size} bytes, received ${bytes.length}`,
    );
  }
  const digest = md5(bytes);
  if (digest !== file.md5) {
    throw new Error(`MD5 mismatch for ${file.name}: expected ${file.md5}, received ${digest}`);
  }
  await writeFile(join(TARGET_DIRECTORY, file.name), bytes);
}

async function main() {
  console.log(`mnist:data target directory: ${TARGET_DIRECTORY}`);
  await mkdir(TARGET_DIRECTORY, { recursive: true });
  let failures = 0;
  for (const file of MNIST_FILES) {
    if (await isAlreadyValid(file)) {
      console.log(`  ${file.name}: ok (already present, size and MD5 match)`);
      continue;
    }
    try {
      await download(file);
      console.log(`  ${file.name}: downloaded (${file.size} bytes, MD5 ${file.md5})`);
    } catch (error) {
      failures += 1;
      console.error(`  ${file.name}: FAILED - ${error.message}`);
    }
  }

  if (failures === 0) {
    console.log('mnist:data complete.');
    return 0;
  }
  if (OPTIONAL) {
    console.warn(
      `mnist:data could not fetch ${failures} file(s). Run "npm run mnist:data" to retry.`,
    );
    return 0;
  }
  console.error(`mnist:data failed for ${failures} file(s).`);
  return 1;
}

process.exitCode = await main();
