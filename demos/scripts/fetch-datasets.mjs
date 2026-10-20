#!/usr/bin/env node
/**
 * Télécharge les archives IDX de MNIST et de Fashion-MNIST dans
 * `demos/public/datasets/<id>/`.
 *
 * Ce script remplace progressivement `fetch-mnist-data.mjs`, qui reste le
 * téléchargeur historique de MNIST utilisé par `postinstall`. Il ne déclare
 * aucune dépendance : `fetch` global, `node:fs/promises`, `node:crypto`,
 * `node:path` et `node:url`.
 *
 * Pour chaque fichier, le script télécharge dans un temporaire `<nom>.tmp`,
 * vérifie la taille puis le MD5, puis renomme le temporaire en place. Un échec
 * supprime le temporaire : un téléchargement interrompu ne laisse jamais de
 * fichier tronqué sous le nom final. Un fichier déjà présent avec la bonne
 * taille et le bon MD5 est conservé sans être retéléchargé, sauf `--force`.
 *
 * Options :
 * - `--only <id>` : ne traite que le jeu donné ; répétable ou séparé par des
 *   virgules. Un identifiant inconnu fait échouer le script.
 * - `--dry-run` : affiche le plan sans télécharger ni rien écrire.
 * - `--force` : retélécharge même un fichier déjà valide.
 * - `--help` : affiche cette aide.
 *
 * Le script sort avec un code non nul si un téléchargement ou une vérification
 * échoue. Les fonctions pures (`parseArgs`, `buildTargetPath`, `describePlan`)
 * sont exportées pour les tests ; la CLI ne s'exécute que si ce fichier est le
 * point d'entrée du processus.
 */

import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { DATASET_SOURCE_IDS, DATASET_SOURCES, datasetFileUrl } from './dataset-sources.mjs';

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(SCRIPT_DIRECTORY, '..');

const USAGE = `Usage: node scripts/fetch-datasets.mjs [options]

Télécharge les archives IDX de MNIST et de Fashion-MNIST dans
public/datasets/<id>/.

Options:
  --only <id>   Ne traite que le jeu donné (répétable ou séparé par des virgules).
                Jeux connus : ${DATASET_SOURCE_IDS.join(', ')}
  --dry-run     Affiche le plan sans télécharger ni écrire.
  --force       Retélécharge même un fichier déjà présent et valide.
  --help        Affiche cette aide.`;

/**
 * Construit le chemin cible d'un fichier : `public/datasets/<id>/<name>`.
 *
 * L'identifiant et le nom de fichier doivent rester des composants de chemin
 * simples, sinon une erreur explicite est levée : la cible est sous
 * `public/datasets/`, jamais ailleurs.
 */
export function buildTargetPath(id, name) {
  for (const [label, value] of [
    ['dataset id', id],
    ['file name', name],
  ]) {
    if (value === '' || value.includes('/') || value.includes('\\')) {
      throw new RangeError(
        `fetch-datasets: invalid ${label} "${value}", expected a simple path component`,
      );
    }
  }
  return join(PROJECT_ROOT, 'public', 'datasets', id, name);
}

/** Analyse les arguments de la ligne de commande. */
export function parseArgs(argv) {
  const options = { only: [], dryRun: false, force: false, help: false };
  const ids = new Set();

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    switch (argument) {
      case '--help':
      case '-h':
        options.help = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      case '--force':
        options.force = true;
        break;
      case '--only': {
        const value = argv[index + 1];
        if (value === undefined) {
          throw new RangeError('fetch-datasets: --only requires a dataset id');
        }
        index += 1;
        for (const id of value.split(',')) {
          if (id !== '') {
            ids.add(id);
          }
        }
        break;
      }
      default:
        throw new RangeError(
          `fetch-datasets: unknown argument "${argument}", see --help for usage`,
        );
    }
  }

  for (const id of ids) {
    if (!(id in DATASET_SOURCES)) {
      throw new RangeError(
        `fetch-datasets: unknown dataset "${id}", expected one of ${DATASET_SOURCE_IDS.join(', ')}`,
      );
    }
  }
  options.only = [...ids];
  return options;
}

/** Empreinte MD5 hexadécimale d'un buffer. */
function md5(bytes) {
  return createHash('md5').update(bytes).digest('hex');
}

/** Vrai si le fichier cible existe déjà avec la taille et le MD5 attendus. */
async function isAlreadyValid(targetPath, file) {
  try {
    const bytes = await readFile(targetPath);
    return bytes.length === file.size && md5(bytes) === file.md5;
  } catch {
    return false;
  }
}

/** Décrit le plan d'exécution sans rien écrire. */
export function describePlan(options) {
  const ids = options.only.length > 0 ? options.only : DATASET_SOURCE_IDS;
  const lines = [];
  for (const id of ids) {
    lines.push(`# ${id}`);
    for (const file of DATASET_SOURCES[id].files) {
      const targetPath = buildTargetPath(id, file.name);
      lines.push(`  ${file.name} <- ${datasetFileUrl(id, file)}`);
      lines.push(`  -> ${targetPath} (${file.size} bytes, MD5 ${file.md5})`);
    }
  }
  return lines.join('\n');
}

/** Télécharge, vérifie puis écrit un fichier. Lance une erreur explicite sinon. */
async function downloadFile(id, file) {
  const url = datasetFileUrl(id, file);
  const targetPath = buildTargetPath(id, file.name);
  const tempPath = `${targetPath}.tmp`;
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== file.size) {
      throw new Error(`size mismatch: expected ${file.size} bytes, received ${bytes.length}`);
    }
    const digest = md5(bytes);
    if (digest !== file.md5) {
      throw new Error(`MD5 mismatch: expected ${file.md5}, received ${digest}`);
    }
    await writeFile(tempPath, bytes);
    await rename(tempPath, targetPath);
    return `downloaded (${file.size} bytes, MD5 ${file.md5})`;
  } catch (error) {
    await unlink(tempPath).catch(() => {});
    throw new Error(`could not fetch ${url}: ${error.message}`);
  }
}

/** Point d'entrée de la CLI. Renvoie le code de sortie du processus. */
export async function main(argv) {
  let options;
  try {
    options = parseArgs(argv);
  } catch (error) {
    console.error(error.message);
    console.error(USAGE);
    return 1;
  }
  if (options.help) {
    console.log(USAGE);
    return 0;
  }

  const ids = options.only.length > 0 ? options.only : DATASET_SOURCE_IDS;

  if (options.dryRun) {
    console.log(describePlan(options));
    return 0;
  }

  for (const id of ids) {
    await mkdir(join(PROJECT_ROOT, 'public', 'datasets', id), { recursive: true });
  }

  let failures = 0;
  for (const id of ids) {
    for (const file of DATASET_SOURCES[id].files) {
      const targetPath = buildTargetPath(id, file.name);
      if (!options.force && (await isAlreadyValid(targetPath, file))) {
        console.log(`  ${id}/${file.name}: ok (already present, size and MD5 match)`);
        continue;
      }
      try {
        const detail = await downloadFile(id, file);
        console.log(`  ${id}/${file.name}: ${detail}`);
      } catch (error) {
        failures += 1;
        console.error(`  ${id}/${file.name}: FAILED - ${error.message}`);
      }
    }
  }

  const total = ids.reduce((count, id) => count + DATASET_SOURCES[id].files.length, 0);
  const fetched = total - failures;
  if (failures === 0) {
    console.log(`datasets:data complete: ${fetched} file(s) verified under public/datasets/.`);
    return 0;
  }
  console.error(`datasets:data failed for ${failures} of ${total} file(s).`);
  return 1;
}

const isEntryPoint = import.meta.url === pathToFileURL(process.argv[1]).href;
if (isEntryPoint) {
  process.exitCode = await main(process.argv.slice(2));
}
