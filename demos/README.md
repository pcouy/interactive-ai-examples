# drl-demos — Bibliothèque de démos interactives

Bibliothèque de composants web (WebComponents Lit + D3 + Three.js) qui alimente le site
statique de présentation. Le site ne consomme que les fichiers générés dans `demos/dist/`.

## Installation

```bash
cd demos
npm install
```

## Scripts

| Commande | Rôle |
|---|---|
| `npm run new -- <nom> [--3d]` | Génère une nouvelle démo (plop) |
| `npm run dev` | Vite : galerie playground + pages démo avec HMR (ouvre `/playground/index.html`) |
| `npm run build` | Vite : `dist/drl-demos.js`, `dist/drl-three.js`, `dist/drl-index.js` et `datasets/**` |
| `npm run test` / `npm run test:watch` | Vitest : core, tokens, garde de palette et composants |
| `npm run typecheck` | `tsc --noEmit` (strict) |
| `npm run lint` / `npm run format` | Biome |

## Jeux de données

La démo `drl-mnist-cnn` charge dix jeux servis sous `public/datasets/<id>/`. Les
commandes ci-dessous les téléchargent et les convertissent. Toutes écrivent dans
`public/datasets/<id>/`.

| Commande | Jeux |
|---|---|
| `npm run datasets:data` | MNIST, Fashion-MNIST |
| `npm run quickdraw:data` | QuickDraw |
| `npm run medmnist:data` | Sept jeux MedMNIST |

Voir `public/datasets/README.md` pour la liste des jeux, le format des fichiers, les
pourcentages de chargement et les licences.

## Créer une démo

```bash
npm run new -- grid-world          # démo 2D (SVG + D3)
npm run new -- robot-arm --3d      # démo 3D (canvas + Three.js)
npm run new                       # mode interactif (sans argument)
```

Le nom doit être en **kebab-case** (`^[a-z][a-z0-9]*(-[a-z0-9]+)*$`), sans collision avec une
démo existante ; `demo`, `controls`, `slider`, `card`, `base`, `core`, `ui`, `styles` sont
réservés.

La génération produit :

| Fichier | Contenu |
|---|---|
| `src/components/drl-<nom>/drl-<nom>.ts` | Squelette du composant Lit |
| `src/core/<nom>.ts` | Squelette de logique pure (sinusoïde factice) |
| `src/core/<nom>.test.ts` | Tests vitest du core (passent d'emblée) |
| `playground/drl-<nom>.html` | Page de dev isolée |

…et modifie automatiquement `src/index.ts` (import du composant à l'ancre
`// @demos-imports`) et `playground/index.html` (carte de galerie à l'ancre
`<!-- @demos-cards -->`). **Ne supprimez pas ces ancres** : le générateur s'en sert.

## Anatomie d'une démo

Chaque démo suit la même découpe en trois couches :

1. **Core** (`src/core/<nom>.ts`) — la logique (Q-learning, jeu, réseau de neurones…) :
   classes/fonctions pures, **aucun import DOM, Lit, D3 ou Three**. C'est la seule cible de
   Vitest. Modèle du squelette : une classe avec `tick(rate)`, `reset()` et des getters
   d'état ; les paramètres (ex. `learningRate`) sont passés en argument de `tick`, jamais
   lus depuis le composant.
2. **Composant** (`src/components/drl-<nom>/drl-<nom>.ts`) — la visualisation, qui hérite de
   `DemoElement` et délègue toute la logique au core.
3. **Page playground** (`playground/drl-<nom>.html`) — page de dev isolée qui charge
   `/src/index.ts` (source, pas `dist/`).

## Le contrat `DemoElement`

`src/base/demo-element.ts` fournit tout le chrome commun : boucle d'animation, boutons
play/pause/step/reset, réglage de vitesse. Le composant enfant n'implémente que la viz.

À implémenter :

| Membre | Rôle |
|---|---|
| `step()` | Un tick de la démo : avance le core puis redessine. Appelé ~30 fois/seconde × `speed` par la boucle, ou manuellement via le bouton Step (à l'arrêt). |
| `renderContent()` | Retourne le balisage de la visualisation (le `<svg>` ou le `<canvas>`). Abstrait. |
| `onReset()` | Optionnel — réinitialise le core et le rendu. Appelé par `reset()` qui coupe aussi la boucle. |
| `renderHeader()` | Optionnel — titre/description au-dessus des contrôles. |
| `firstUpdated()` | Optionnel — point d'entrée pour brancher D3 (2D) ou charger Three (3D). Le DOM n'existe pas avant. |

À ne PAS faire :

- ne pas surcharger `render()` : il est final et compose `<drl-controls>` + `renderContent()`.
- ne pas gérer sa propre boucle `requestAnimationFrame` : le base class s'en charge.
- ne pas instancier le core dans `renderContent()` : gardez-le dans un champ
  (`private readonly core = new MonCore()`).

Propriétés héritées, exposées comme attributs HTML : `running` (booléen, démarre la boucle
dès la connexion si défini) et `speed` (nombre > 0, ramené à 1 sinon ; 1 tick/seconde
nominal = 30 × speed).

## Démo 2D (SVG + D3)

- `renderContent()` retourne un `<svg viewBox="…">` dimensionné via les styles statiques.
- Dans `firstUpdated()`, sélectionnez le SVG depuis `this.renderRoot` et conservez les
  sélections D3 dans des champs ; les ticks suivants ne font que mettre à jour les
  attributs (pattern « enter once, update many » du squelette).
- D3 fait partie du bundle principal `drl-demos.js` : import statique `import { select }
  from 'd3'`, sans souci de poids.

## Templates SVG avec Lit (`svg` vs `html`)

Piège déjà rencontré sur `drl-game-board` : un `<svg>` **vide à l'écran** alors que ses
nœuds (`rect`, `circle`, `line`…) sont bel et bien présents dans l'inspecteur de DevTools.

Cause : Lit parse les templates `html` comme du **HTML**. Un template dont l'élément
racine est `<svg>` est correct (le parseur HTML bascule en « foreign content » et place
tout le sous-arbre dans le namespace SVG). En revanche, un template **imbriqué** — ou un
helper — dont la racine est un élément SVG direct (`<rect>`, `<circle>`, `<line>`…)
produit des éléments dans le **namespace HTML**. Insérés dans un vrai `<svg>`, ils
existent dans le DOM mais ne sont **jamais dessinés**. Le tag ``svg` `` de Lit corrige
cela en parsant le fragment dans le bon namespace.

Règle : **dès que la racine d'un template est un élément SVG, utilisez ``svg` `` et non
``html` ``** — et typez le retour en `SVGTemplateResult` (`TemplateResult` vaut
`TemplateResult<1>` et n'accepte pas un résultat SVG).

```ts
import { html, svg } from 'lit';
import type { SVGTemplateResult } from 'lit';

// Racine <svg> : html OK (le parseur HTML passe en « foreign content »).
renderContent() {
  return html`<svg viewBox="…">${this.pieces()}</svg>`;
}

// Racine = éléments SVG : tag svg obligatoire, retour typé SVGTemplateResult.
private pieces(): SVGTemplateResult {
  return svg`<rect x="0" y="0" width="10" height="10"></rect>`;
}
```

Symptômes de l'erreur, utiles au diagnostic : « Modifier en HTML/SVG » depuis l'inspecteur
fait réapparaître le dessin — reparcourir le balisage recrée les nœuds dans le namespace
SVG — mais **casse les événements** (`@click`…), car les `addEventListener` posés par Lit
sont perdus. Exemple corrigé :
`src/components/drl-game-board/drl-game-board.ts`.

## Démo 3D (canvas + Three.js)

- `renderContent()` retourne un `<canvas>` (ou le message d'erreur `webglError`).
- **Three uniquement via dynamic import** dans `firstUpdated()` :

  ```ts
  const THREE = await import('three');
  ```

  C'est ce qui permet à Rollup d'éjecter Three dans un chunk séparé (`drl-three.js`),
  téléchargé seulement par les pages contenant une démo 3D. Un import statique casserait
  cette garantie (et doublerait le bundle pour tous les visiteurs).
- Enveloppez l'init dans un `try/catch` (WebGL peut être indisponible) et basculez
  `webglError` pour afficher un message de repli.
- Nettoyez dans `disconnectedCallback()` : `geometry.dispose()`, `material.dispose()`,
  `renderer.dispose()`, puis appelez `super.disconnectedCallback()`.
- Typage sans import statique : `type ThreeModule = typeof import('three')`.

## Exposer des paramètres

Les paramètres sont des propriétés réactives Lit, avec un **attribut kebab-case** pour
l'usage déclaratif :

```ts
@property({ type: Number, attribute: 'learning-rate' })
learningRate = 0.1;
```

Le site peut alors écrire `<drl-grid-world learning-rate="0.1" speed="2">`. Dans le
playground, branchez un `<drl-slider>` (événement `drl-input`, `detail` numérique) sur la
propriété, comme dans la page générée :

```js
const demo = document.querySelector('drl-grid-world');
slider.addEventListener('drl-input', (event) => {
  demo.learningRate = event.detail;
});
```

## Kit UI commun

| Élément | Props / événements |
|---|---|
| `<drl-controls>` | `running`, `speed` ; émet `drl-play`, `drl-pause`, `drl-step`, `drl-reset`, `drl-speed` (detail numérique). Inclus automatiquement par `DemoElement`. |
| `<drl-slider>` | `label`, `min`, `max`, `step`, `value` ; émet `drl-input` (detail numérique). |
| `<drl-card>` | `heading` + slot ; conteneur de mise en page pour le site hôte. |

## Thème

### Une seule source de valeurs

`src/styles/token-values.ts` est la seule source des valeurs de couleur. Le fichier
exporte deux objets :

- `drlTokenValues` — 14 valeurs : 8 couleurs (`bg`, `surface`, `fg`, `muted`, `accent`,
  `accent2`, `arrow`, `border`) et 6 valeurs de mise en forme (`radius`, `font`,
  `fontSize`, `space1`, `space2`, `space3`). La feuille Lit (`src/styles/tokens.ts`) et
  la scène Three.js (`src/components/shared/landscape-scene.ts`) lisent cet objet.
- `drlPalette` — deux thèmes, `light` et `dark`. Chaque thème définit 13 couleurs. Le
  thème clair réutilise `drlTokenValues` ; le thème sombre fournit ses propres valeurs.

| Clé | Rôle | Clair | Sombre |
|---|---|---|---|
| `bg` | Fond de page | `#f8fafc` | `#0b1220` |
| `surface` | Fond des panneaux, cartes et barres de contrôle | `#ffffff` | `#111a2b` |
| `fg` | Texte principal | `#0f172a` | `#e2e8f0` |
| `muted` | Texte secondaire, légendes | `#64748b` | `#94a3b8` |
| `accent` | Accent principal : sélection, données positives | `#2563eb` | `#60a5fa` |
| `accent-2` | Second accent : cible opposée, données négatives | `#7c3aed` | `#a78bfa` |
| `arrow` | Flèches et repères directionnels | `#d32f2f` | `#f87171` |
| `border` | Bordures et séparateurs | `#e2e8f0` | `#1e293b` |
| `fg-2` | Texte secondaire contrasté | `#334155` | `#cbd5e1` |
| `muted-2` | Texte discret, détails secondaires | `#94a3b8` | `#8b98ad` |
| `warn` | État d'avertissement | `#b45309` | `#fbbf24` |
| `danger` | État d'erreur | `#b91c1c` | `#fca5a5` |
| `danger-2` | Erreur renforcée | `#dc2626` | `#f87171` |

`fg-2`, `muted-2`, `warn`, `danger` et `danger-2` servent aux pages playground. Aucun
composant ne les lit aujourd'hui.

### `--drl-theme-*` : l'entrée de thème

Chaque composant redéclare les tokens sur son `:host` :

```css
:host {
  --drl-accent: var(--drl-theme-accent, #2563eb);
}
```

La valeur de repli est la valeur claire de `drlPalette.light`.

`--drl-theme-*` est l'entrée de thème prise en charge. Poser `--drl-*` sur `:root` ne
suffit pas. Les propriétés personnalisées s'héritent, mais le `:host` de chaque composant
déclare déjà `--drl-*`. Une déclaration locale gagne contre une valeur héritée. Dans un
shadow root imbriqué, le composant garde donc sa valeur claire. `--drl-theme-*` n'est
jamais redéclaré : il traverse l'héritage et alimente le `var()` du `:host`.

Thème clair :

```css
:root {
  --drl-theme-bg: #f8fafc;
  --drl-theme-surface: #ffffff;
  --drl-theme-fg: #0f172a;
  --drl-theme-muted: #64748b;
  --drl-theme-accent: #2563eb;
  --drl-theme-accent-2: #7c3aed;
  --drl-theme-arrow: #d32f2f;
  --drl-theme-border: #e2e8f0;
  --drl-theme-fg-2: #334155;
  --drl-theme-muted-2: #94a3b8;
  --drl-theme-warn: #b45309;
  --drl-theme-danger: #b91c1c;
  --drl-theme-danger-2: #dc2626;
}
```

Thème sombre :

```css
:root[data-theme='dark'] {
  --drl-theme-bg: #0b1220;
  --drl-theme-surface: #111a2b;
  --drl-theme-fg: #e2e8f0;
  --drl-theme-muted: #94a3b8;
  --drl-theme-accent: #60a5fa;
  --drl-theme-accent-2: #a78bfa;
  --drl-theme-arrow: #f87171;
  --drl-theme-border: #1e293b;
  --drl-theme-fg-2: #cbd5e1;
  --drl-theme-muted-2: #8b98ad;
  --drl-theme-warn: #fbbf24;
  --drl-theme-danger: #fca5a5;
  --drl-theme-danger-2: #f87171;
}
```

### Pages playground

Les pages de `playground/` chargent `/src/styles/playground-tokens.ts` en module
différé. Le module injecte une balise `<style>` en tête de `<head>`, avec deux blocs
`:root` : le thème clair, puis le thème sombre sous `:root[data-drl-theme='dark']`. Les
pages écrivent `var(--drl-theme-*)`.

Le module s'exécute après l'analyse du HTML. Une page de dev peut donc afficher une frame
dans le thème clair avant l'injection du thème sombre. C'est attendu.

### Garde de palette

Le test `src/styles/palette-guard.test.ts` refuse toute couleur littérale dans le code. Il
parcourt `demos/src`, `demos/playground`, `demos/plop-templates`, `demos/scripts`,
`demos/public_site/src` et `demos/public_site/scripts`. Seuls deux fichiers échappent à la
règle : `src/styles/token-values.ts` et la fixture du parseur MNIST
(`src/components/drl-mnist-cnn/drl-mnist-cnn.test.ts`). Les dossiers `node_modules`,
`dist`, `.astro`, `.cache`, `.git` et `public` sont ignorés, ainsi que
`public_site/src/styles/palette.generated.css`.

```bash
npx vitest run src/styles/palette-guard.test.ts
```

`--drl-theme-accent` et `--drl-theme-accent-2` doivent rester en `#rrggbb`. `hexToRgb`
les analyse dans `drl-mnist-cnn` (palette des filtres de convolution). Une autre notation
(`rgb()`, `color-mix()`, nom de couleur) casse la palette : les deux couleurs retombent
sur le même bleu par défaut. `public_site/scripts/sync-demo-assets.mjs` valide aussi les
13 couleurs des deux thèmes au format `#rrggbb` et vérifie les contrastes.

### Composants qui lisent la couleur au montage

`drl-neuron`, `drl-mnist-cnn` et `drl-gradient-descent` lisent `--drl-accent` (et
`--drl-accent-2`) une seule fois avec `getComputedStyle`, puis gardent le résultat. Un
changement de thème après le montage ne met pas leurs couleurs à jour. Le site public
pose le thème avant le chargement des composants : le script inline de `Base.astro`
définit `data-theme` pendant l'analyse du `<head>`, et `palette.generated.css` en dérive
les `--drl-theme-*`.

## Intégrer au site statique

Construisez et vérifiez le paquet avant de le copier :

```bash
npm run typecheck   # tsc --noEmit, strict
npm run lint        # biome check .
npm run test        # vitest run
npm run build       # vite build
```

`npm run build` écrit dans `dist/` :

| Fichier | Rôle |
|---|---|
| `drl-demos.js` | Bundle principal : enregistre tous les composants |
| `drl-three.js` | Chunk Three.js, chargé par les démos 3D |
| `drl-index.js` | Chunk TensorFlow.js, chargé par l'entraîneur MNIST |
| `datasets/**` | Jeux de données copiés depuis `public/datasets/` |

Copiez **tous** les fichiers `.js` de `dist/` côte à côte, par exemple dans `assets/` :

```html
<script type="module" src="assets/drl-demos.js"></script>
<drl-perceptron learning-rate="0.1" speed="2"></drl-perceptron>
```

- Zéro CSS à ajouter : le Shadow DOM encapsule les styles.
- L'import du module enregistre tous les composants (effet de bord
  `customElements.define`).
- Les chunks paresseux sont obligatoires. `drl-demos.js` importe `./drl-three.js` et
  `./drl-index.js` à la demande. Copier seulement `drl-demos.js` casse les démos MNIST et
  3D : l'import dynamique renvoie une erreur 404.
- `drl-three.js` n'est téléchargé que si une démo 3D figure sur la page.

## Checklist avant de considérer une démo terminée

```bash
npm run typecheck   # strict
npm run lint
npm run test        # tests du core à jour et verts
npm run build
npm run dev         # vérifier playground/drl-<nom>.html en vrai rendu
```

## Conventions

- Préfixe `drl-` partout ; 1 démo = 1 dossier `src/components/drl-<nom>/`.
- Noms : composant/tag en kebab-case, classe en PascalCase (`grid-world` → `GridWorld`),
  core en `<Nom>Core`.
- Les ancres `// @demos-imports` et `<!-- @demos-cards -->` sont gérées par plop uniquement.
- SVG en Lit : tag ``svg` `` (et retour `SVGTemplateResult`) dès que la racine d'un
  template est un élément SVG — voir « Templates SVG avec Lit ».
- Code en anglais ; la logique métier vit dans `src/core/`, testable sans navigateur.
