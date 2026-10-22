/**
 * Marque un répertoire de jeu de données avec un fichier `CACHEDIR.TAG`.
 *
 * Le format est décrit sur <https://bford.info/cachedir/> : la première ligne
 * est la signature exacte, suivie de commentaires libres. Les outils de
 * sauvegarde et d'indexation reconnaissent ce fichier et peuvent ignorer le
 * répertoire, ce qui évite de dupliquer les archives téléchargées.
 *
 * `writeCacheDirTag` crée le répertoire si besoin et n'écrit le fichier que
 * s'il manque ou diffère : une exécution répétée ne modifie pas sa date. Le
 * module n'a aucune dépendance et est partagé par tous les scripts de
 * téléchargement.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

/** Nom du fichier marqueur. */
export const CACHE_DIR_TAG_NAME = 'CACHEDIR.TAG';

/**
 * Contenu du marqueur. La signature doit rester la première ligne, sur
 * exactement 43 octets (sans le saut de ligne), pour être reconnue par les
 * outils.
 */
export const CACHE_DIR_TAG = `Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by the drl-demos dataset scripts.
# For information about cache directory tags, see:
#       https://bford.info/cachedir/
`;

/** Chemin du marqueur dans un répertoire donné. */
export function cacheDirTagPath(directory) {
  return join(directory, CACHE_DIR_TAG_NAME);
}

/**
 * Écrit `CACHEDIR.TAG` dans `directory`, en créant le répertoire au besoin.
 *
 * Renvoie `true` si le fichier a été écrit, `false` s'il était déjà à jour.
 */
export async function writeCacheDirTag(directory) {
  const path = cacheDirTagPath(directory);
  let current = null;
  try {
    current = await readFile(path, 'utf8');
  } catch {
    current = null;
  }
  if (current === CACHE_DIR_TAG) {
    return false;
  }
  await mkdir(directory, { recursive: true });
  await writeFile(path, CACHE_DIR_TAG);
  return true;
}
