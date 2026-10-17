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
| `npm run build` | Génère `dist/drl-demos.js` (+ chunk `drl-three.js` si une démo 3D existe) |
| `npm run test` / `npm run test:watch` | Vitest (logique pure dans `src/core/`) |
| `npm run typecheck` | `tsc --noEmit` (strict) |
| `npm run lint` / `npm run format` | Biome |

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

Tous les composants consomment les tokens CSS `src/styles/tokens.ts` : `--drl-bg`,
`--drl-surface`, `--drl-fg`, `--drl-muted`, `--drl-accent`, `--drl-accent-2`,
`--drl-border`, `--drl-radius`, `--drl-font`, `--drl-font-size`, `--drl-space-1/2/3`.
Les variables traversent le Shadow DOM : le site hôte les surcharge sans ajouter de CSS
au composant.

```css
drl-grid-world {
  --drl-accent: #d81b60;
  --drl-radius: 4px;
}
```

## Intégrer au site statique

Après `npm run build`, copiez **tous** les fichiers de `dist/` (aujourd'hui
`drl-demos.js`, plus `drl-three.js` dès qu'une démo `--3d` existe) côte à côte, par
exemple dans `assets/` :

```html
<script type="module" src="assets/drl-demos.js"></script>
<drl-perceptron learning-rate="0.1" speed="2"></drl-perceptron>
```

- Zéro CSS à ajouter : le Shadow DOM encapsule les styles.
- L'import du module enregistre tous les composants (effet de bord
  `customElements.define`).
- `drl-three.js` n'est téléchargé que si une démo 3D figure sur la page (dynamic import).

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
- Code en anglais ; la logique métier vit dans `src/core/`, testable sans navigateur.
