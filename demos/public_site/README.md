# Jouer pour comprendre l'IA — site public

## Aperçu

Site statique Astro pour « Jouer pour comprendre l'IA ». Il présente deux démos
interactives : Deep Blue (recherche adversariale) et Frozen Lake (Q-learning).

- Contenu en français, dans des fichiers Markdown.
- Formules rendues par KaTeX au build.
- Thème clair et sombre, sans flash au chargement.
- Aucun appel réseau au runtime : les scripts et les polices viennent du dossier du site.

Le site consomme les bundles générés par `demos/`. `prebuild` construit ces bundles
automatiquement.

## Prérequis

- Node `>=22.22.0`. `sync-demo-assets.mjs` refuse une version plus ancienne.
- npm.

## Installation

```bash
cd demos
npm install
cd public_site
npm install
```

Le `postinstall` de `demos/` télécharge les jeux MNIST en mode optionnel. Le site public
n'en a pas besoin.

## Développement

```bash
npm run dev
```

Ouvrez l'URL affichée par Astro (par défaut `http://localhost:4321/`).

`predev` construit d'abord les démos (`npm --prefix .. run build`), puis lance
`scripts/sync-demo-assets.mjs`. Le script copie les bundles de `../dist/` vers
`public/demo/`, copie KaTeX vers `public/katex/` et régénère
`src/styles/palette.generated.css`.

`npm run dev` a besoin de `demos/node_modules`. Lancez `npm install` dans `demos/` avant le
premier `npm run dev`.

`npm` lit `predev` au démarrage. Un serveur de dev déjà lancé garde l'ancien `predev`.
Redémarrez `npm run dev` pour prendre en compte un nouveau `predev`.

`file://` ne fonctionne pas : les modules ES et `fetch` exigent un serveur HTTP. Utilisez
`npm run dev`, `npm run preview` ou un serveur statique local.

## Build

```bash
npm run build
```

Le build se déroule en trois étapes :

1. `prebuild` construit `demos/` (`npm --prefix .. run build`), puis lance
   `scripts/sync-demo-assets.mjs`. Le script copie les bundles, copie KaTeX, valide les
   13 couleurs des deux thèmes et vérifie les contrastes (minimum 4.5 pour `fg/bg`,
   `muted/bg` et `accent/bg`).
2. `astro build` rend les pages dans `dist/`. La configuration utilise
   `build.format: 'file'` et `inlineStylesheets: 'always'`.
3. `scripts/finalize-build.mjs` réécrit les URL absolues (`/deep-blue`) en URL relatives
   (`./deep-blue.html`). `scripts/verify-build.mjs` contrôle ensuite le résultat : fichiers
   requis, liens, balises de thème, KaTeX, balises de démo et absence d'URL absolue. Il
   vérifie aussi l'identité octet pour octet du bundle copié et de
   `demos/dist/drl-demos.js`, ainsi que la présence de quatre marqueurs :
   `--drl-theme-bg`, `Boucle agent-environnement`, `Dernière mise à jour` et `Rejeu`.

Le build échoue si un contrôle de `verify-build.mjs` échoue.

`precheck` reconstruit aussi les démos. `npm run verify` (`check` + `test` + `build`)
construit donc les démos deux fois.

## Dossier déployable

Le dossier à publier est `demos/public_site/dist/` :

| Chemin | Contenu |
|---|---|
| `index.html` | Page d'accueil |
| `deep-blue.html` | Démo Deep Blue |
| `frozen-lake.html` | Démo Frozen Lake |
| `demo/` | `drl-demos.js`, `drl-three.js`, `drl-index.js` |
| `katex/` | `katex.min.css` et `fonts/` |

Copiez le dossier complet. Ne séparez pas les pages des dossiers `demo/` et `katex/`.

## Hébergement dans un sous-chemin

Toutes les URL du build sont relatives : `./deep-blue.html`, `./demo/drl-demos.js`,
`./katex/katex.min.css`. Le dossier fonctionne donc à la racine du domaine ou dans un
sous-chemin, par exemple `https://exemple.fr/nested/path/`.

`build.format: 'file'` écrit toutes les pages à la racine de `dist/`, à la même
profondeur. Les liens relatifs restent valides quel que soit le préfixe.

L'URL d'entrée doit se terminer par `/` ou par `index.html` :

- `https://exemple.fr/nested/path/` fonctionne.
- `https://exemple.fr/nested/path/index.html` fonctionne.
- `https://exemple.fr/nested/path` peut renvoyer 404 : le serveur cherche un fichier sans
  slash final.

## Thème clair et sombre

Le thème est stocké dans `localStorage['drl-theme']`, avec la valeur `light` ou `dark`.

Un script inline dans `<head>` (`src/layouts/Base.astro`) lit la valeur et pose
`data-theme` sur `<html>` avant le premier rendu. Sans valeur stockée, le script suit
`prefers-color-scheme`. Le script inline évite le flash de thème.

Le bouton `[data-theme-toggle]` bascule le thème et écrit la valeur dans `localStorage`.
Il met à jour `aria-pressed` et son libellé.

Les composants reçoivent `--drl-theme-*` depuis `src/styles/palette.generated.css`. Le
fichier est généré par `scripts/sync-demo-assets.mjs` à partir de
`../src/styles/token-values.ts`. Il contient un bloc `:root` clair, un bloc
`:root[data-theme='dark']` et un bloc `@media (prefers-color-scheme: dark)` pour les
visiteurs sans préférence enregistrée.

Les pages utilisent les tokens `--site-*` (`--site-bg`, `--site-fg`, `--site-accent`…).
Ne codez pas de couleur en dur : `global.css` doit rester sans littéral de couleur.

## Rédiger du contenu

Les textes vivent dans `src/content/`, en Markdown :

| Fichier | Usage |
|---|---|
| `home.md` | Section « À propos » de l'accueil |
| `deep-blue.md` | Corps de la page Deep Blue |
| `frozen-lake.md` | Corps de la page Frozen Lake |

Les formules utilisent LaTeX : `$...$` en ligne, `$$...$$` en bloc. `remark-math` et
`rehype-katex` les rendent au build. Le HTML produit contient des éléments `<math>` et des
classes `.katex`.

`home.md` ne doit contenir ni `#` ni `##`. La page d'accueil fournit déjà le `<h1>` et le
`<h2>` de la section. Utilisez `###` pour les sous-titres.

Les pages de démo commencent au niveau `##`. Le layout `DemoPage.astro` fournit le `<h1>`
à partir du frontmatter.

Frontmatter des pages de démo :

```markdown
---
title: Frozen Lake · Q-learning tabulaire
description: Guidez un agent qui apprend seul à traverser un lac gelé.
---
```

`title` alimente le `<h1>` et le `<title>`. `description` alimente le paragraphe
d'introduction et la meta description.

Pour ajouter une formule, écrivez la syntaxe LaTeX entre les délimiteurs, sans espace
après le `$` d'ouverture :

```markdown
La valeur $Q(s,a)$ mesure l'intérêt du couple état-action.

$$ Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right] $$
```

## Ajouter une démo

1. Créez la démo dans `demos/` (`npm run new -- <nom>`) et construisez-la :
   `npm --prefix .. run build`.
2. Copiez les bundles et régénérez la palette : `npm run sync`.
3. Créez `src/content/<nom>.md` avec `title` et `description`, puis le corps à partir de
   `##`.
4. Créez `src/pages/<nom>.astro`. Importez le Markdown et `DemoPage.astro`.
5. Ajoutez la section `demo-surface` et l'élément `<drl-*>`.
6. Ajoutez le script du bundle, puis le script de collage :

   ```astro
   <Fragment slot="scripts">
     <script is:inline type="module" src="/demo/drl-demos.js"></script>
     <script is:inline type="module" set:html={glue}></script>
   </Fragment>
   ```

7. Gardez le script du bundle avant le script de collage. Le bundle enregistre les
   custom elements. Le script de collage lit ces éléments.
8. Lancez `npm run build`, puis `npm run verify`.

Écrivez les URL en absolu dans les sources (`/demo/drl-demos.js`). `finalize-build.mjs`
les convertit en relatif.

Le script de collage se trouve dans `src/scripts/`. Importez-le avec `?raw`, comme
`deep-blue.astro` et `frozen-lake.astro`.

## Scripts

| Commande | Rôle |
|---|---|
| `npm run sync` | Copie `../dist/*.js` vers `public/demo/`, copie KaTeX et régénère `palette.generated.css`. Il refuse un bundle plus ancien que `demos/src` ou que les fichiers de configuration des démos |
| `npm run dev` | Construit les démos, puis synchronise, puis sert Astro avec HMR (`predev`) |
| `npm run build` | `prebuild` + `astro build` + `finalize-build.mjs` + `verify-build.mjs` |
| `npm run preview` | Sert `dist/` en local |
| `npm run check` | Construit les démos, puis synchronise, puis `astro check` (`precheck`) |
| `npm test` | Tests Node de `scripts/url-rewrite.test.mjs` |
| `npm run verify` | `check` + `test` + `build` |
| `npm run smoke` | Contrôle HTTP du build (voir « Vérification »). Vérifie aussi les quatre marqueurs dans le bundle servi |
| `npm run verify:full` | `verify` + `smoke` |

## Vérification

```bash
npm run check       # types et diagnostics Astro
npm test            # réécriture d'URL
npm run build       # build complet + contrôles de sortie
npm run smoke       # contrôle HTTP de dist/
npm run verify      # check + test + build
npm run verify:full # verify + smoke
```

`npm run smoke` est un contrôle HTTP. Le script sert `dist/` à `/nested/site/` avec Node.
Il vérifie les pages, les assets, les URL relatives, le balisage du thème, le balisage
KaTeX et les balises de démo avec `fetch`.

`npm run smoke` exige un build à jour. Lancez-le après `npm run build`.

`npm run verify` échoue si un contrôle de `verify-build.mjs` échoue. Le message indique le
fichier et la règle en cause.

## Déploiement sur GitHub Pages

Le site se déploie avec GitHub Actions. Le workflow se trouve dans
`.github/workflows/deploy-pages.yml`, à la racine du dépôt.

Le workflow se déclenche sur un push vers la branche `main` et sur un lancement manuel
(`workflow_dispatch`). Il ne se déclenche pas sur une pull request ni sur un fork.

**Réglage initial.** Dans le dépôt GitHub, ouvrez **Settings** → **Pages** →
**Build and deployment**, puis choisissez **Source : GitHub Actions**. Sans ce réglage,
l'étape `configure-pages` échoue.

**Étapes du workflow.** Le workflow exécute ces actions :

1. Il installe `demos/` avec `npm ci --ignore-scripts --no-audit --no-fund`. L'option
   `--ignore-scripts` saute le `postinstall`, donc le téléchargement optionnel des jeux
   MNIST.
2. Il installe `demos/public_site/` avec `npm ci --no-audit --no-fund`.
3. Il lance `npm run build` dans `demos/public_site/`. Ce contrôle rapide construit les
   démos, synchronise les assets, lance le build Astro et le finaliseur, puis exécute
   `verify-build.mjs` (109 contrôles). Il saute `astro check`, les tests unitaires d'URL
   et le contrôle HTTP `smoke`.
4. Il envoie `demos/public_site/dist` avec `actions/upload-pages-artifact`.
5. Il publie le site avec `actions/deploy-pages@v5`.

**URL publiée.** L'URL a la forme `https://<owner>.github.io/<repo>/`. L'URL d'entrée doit
finir par `/` ou par `index.html` :

- `https://<owner>.github.io/<repo>/` fonctionne.
- `https://<owner>.github.io/<repo>/index.html` fonctionne.
- `https://<owner>.github.io/<repo>` peut renvoyer 404.

Le build réécrit les URL absolues en URL relatives. Le sous-chemin `/<repo>/` fonctionne
donc sans réglage `base` dans `astro.config.mjs`. Voir aussi « Hébergement dans un
sous-chemin ».

**Reproduction locale.** Pour produire le même dossier que le workflow :

```bash
cd demos
npm ci --ignore-scripts --no-audit --no-fund
npm run build
cd public_site
npm ci --no-audit --no-fund
npm run build
```

Le workflow envoie le dossier `demos/public_site/dist/`.

## Dépannage

**Bundle de démo absent.** `sync-demo-assets: Missing .../demos/dist/drl-demos.js`.
Lancez `npm --prefix .. run build`, puis `npm run sync`.

**Démo vide.** Vérifiez l'ordre des scripts : `/demo/drl-demos.js` avant le script de
collage. Vérifiez aussi que le tag `<drl-*>` correspond au composant enregistré.

**`$` affiché en clair.** La formule n'a pas été rendue. Vérifiez les délimiteurs
(`$...$`, `$$...$$`), l'absence d'espace après le `$` d'ouverture, et le passage du
Markdown par Astro.

**Flash de thème.** Le script inline de `Base.astro` doit rester dans `<head>`, avant la
première feuille de style. `verify-build.mjs` contrôle cet ordre.

**Couleurs sombres incorrectes.** `palette.generated.css` est périmé. Lancez `npm run sync`
puis `npm run build`. Si le problème persiste, vérifiez `drlPalette.dark` dans
`../src/styles/token-values.ts`.

**Panneaux de démo clairs en thème sombre.** Le chrome du site est sombre, mais les
panneaux de démo restent clairs et le texte sur les zones transparentes reste foncé. Les
panneaux MDP de Frozen Lake manquent. Cause : un `demos/dist/drl-demos.js` périmé a été
copié dans `public/demo/`. Corrigez avec :

```bash
npm --prefix .. run build
npm run sync
```

Redémarrez ensuite `npm run dev`. Le script `sync-demo-assets.mjs` refuse maintenant un
bundle plus ancien que le fichier le plus récent de `demos/src` ou que les fichiers de
configuration des démos. Le message demande de lancer `npm --prefix .. run build`.

**404 dans un sous-chemin.** Utilisez le dossier `dist/` complet et une URL d'entrée qui
finit par `/` ou `index.html`. Vérifiez que les fichiers sont copiés avec leur
arborescence (`demo/`, `katex/`).

**Échec en `file://`.** Les modules ES et `fetch` ne fonctionnent pas sous `file://`.
Servez le dossier en HTTP avec `npm run preview`.

**404 sur une police KaTeX.** `public/katex/fonts/` est incomplet. Lancez `npm run sync`,
puis `npm run build`. Vérifiez que le déploiement inclut `katex/fonts/`.

## Fichiers générés

Ces chemins sont générés et ignorés par Git. Ne les modifiez pas à la main :

- `public/demo/`
- `public/katex/`
- `src/styles/palette.generated.css`
- `dist/`
- `.astro/`

`npm run sync` recrée les trois premiers. `npm run build` recrée `dist/` et `.astro/`.
