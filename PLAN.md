# Plan — Bibliothèque de démos interactives (WebComponents TS + D3 + Three.js)

## 1. Emplacement et périmètre

Nouveau projet Node dans `demos/` (à la racine du repo, à côté des fichiers du site). Le site statique consomme uniquement les fichiers générés dans `demos/dist/`.

## 2. Dépendances

| Rôle | Paquet |
|---|---|
| Runtime | `lit`, `d3`, `three` |
| Build | `vite` (mode librairie, ESM) |
| Dev | `typescript` (strict), `vitest`, `biome`, `plop` |

## 3. Configuration build (`vite.config.ts`)

- `build.lib` : entry unique `src/index.ts`, format **ESM uniquement**, nom `drl-demos`
- Aucune dépendance externe : Lit + D3 + core bundle dans `dist/drl-demos.js`
- **Three.js en dynamic import** → Rollup l'éjecte dans un chunk séparé, chargé seulement par les démos 3D (ESM seul le permet ; c'est pour ça qu'on ne fait pas d'IIFE)
- Importer `drl-demos.js` déclenche l'enregistrement de tous les composants (effet de bord `customElements.define`)

## 4. Architecture du code

```
demos/
├── plopfile.mjs
├── plop-templates/            # .hbs : composant, core, test, page playground, 3d variant
├── src/
│   ├── index.ts               # entry : import + define tous les composants
│   ├── base/demo-element.ts   # classe de base : running/speed, reset()/step(), contrôles communs
│   ├── ui/                    # <drl-controls> (play/pas/reset/vitesse), <drl-slider>, <drl-card>
│   ├── styles/tokens.ts       # CSS custom properties (thème surchargeable par le site hôte)
│   ├── core/                  # logique pure (Q-learning, jeux, NN) — cible de Vitest
│   └── components/drl-*/
└── playground/
    ├── index.html             # galerie (dev + aperçu du rendu réel)
    └── drl-<nom>.html        # page isolée par démo
```

**Conventions** : préfixe `drl-` ; 1 démo = 1 dossier ; les paramètres exposés sont des propriétés réactives Lit (attributs HTML : `<drl-gridworld learning-rate="0.5">`) ; D3 opère sur le SVG rendu par Lit dans `firstUpdated()` ; les démos 3D chargent Three via `await import('three')`.

## 5. Générateur Plop

`npm run new -- <nom> [--3d]` produit :
1. `src/components/drl-<nom>/drl-<nom>.ts` — squelette Lit (SVG + D3, ou canvas + Three si `--3d`), héritant de `DemoElement`
2. `src/core/<nom>.ts` + `<nom>.test.ts` — squelettes de logique pure
3. `playground/drl-<nom>.html` — page de dev isolée
4. Actions `modify` : ajout de l'import/define dans `src/index.ts` et d'une carte dans la galerie `playground/index.html`

## 6. Scripts npm

```
npm run new [-- --name <nom>]   # plop : nouvelle démo
npm run dev                     # vite : galerie + pages démo avec HMR
npm run build                   # dist/drl-demos.js (+ chunk three)
npm run test / test:watch       # vitest (core)
npm run typecheck               # tsc --noEmit
npm run lint / format           # biome
```

## 7. Étapes d'implémentation

1. **Bootstrap toolchain** : `demos/` + package.json, tsconfig strict, vite.config (lib ESM), biome, vitest
2. **Fondations UI** : `DemoElement`, `<drl-controls>`, tokens CSS, `src/index.ts`
3. **Générateur** : plopfile + templates + actions modify
4. **Validation bout-en-bout** : générer une démo jetable → dev → build → intégration dans une page statique hors playground (copie de `dist/`, balise `<script type="module">` + `<drl-*>`) → suppression de la démo jetable
5. Ensuite, chaque démo de la présentation devient : `npm run new -- <nom>` puis implémentation core + viz (hors périmètre de ce plan)

## 8. Intégration site statique (usage final)

```html
<script type="module" src="assets/drl-demos.js"></script>
<drl-perceptron learning-rate="0.1" speed="2"></drl-perceptron>
```

Zéro CSS à ajouter (Shadow DOM encapsule), thèmes optionnels via CSS variables, chunk Three.js téléchargé uniquement si une démo 3D figure sur la page.

