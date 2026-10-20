# Deep Blue — Utilisation et référence des interfaces

Ce document décrit l'**implémentation réelle** de la démo Deep Blue : APIs
publiques vérifiées dans le code, signatures exactes, valeurs par défaut,
événements. Les documents `00-overview.md` à `04-arbre.md` restent les
*spécifications d'intention* ; en cas d'écart, c'est ce document qui fait foi.
Le rôle narratif — rendre visible la recherche adversariale et l'évaluation
artisanale, préalable à l'apprentissage — est celui de la section 1.A.i de
`presentation.md`.

## 1. Ce qui est livré

```
demos/
├── src/core/
│   ├── contracts.ts              # Game, Evaluator, SearchAgent, TreeRecorder…
│   ├── games/                    # tic-tac-toe, connect4, board-view, types
│   └── search/                   # minimax, transposition, random-agent,
│                                 # evaluators/, tree-model, stepper
├── src/components/
│   ├── shared/create-game-component.ts   # factory + API des composants de jeu
│   ├── drl-game-board/           # plateau générique (BoardView → SVG)
│   ├── drl-tic-tac-toe/          # partie morpion
│   ├── drl-connect4/             # partie puissance 4
│   ├── drl-eval-tree/            # visualiseur d'arbre d'évaluation
│   └── drl-deep-blue-page/       # page complète (jeu + arbre)
├── playground/                   # pages de dev isolées (drl-deep-blue-page.html,
│                                 # drl-tic-tac-toe.html, drl-eval-tree.html, …)
└── src/index.ts                  # enregistrement de tous les composants
```

Couverture des spécifications : `01-contrats.md` → `src/core/contracts.ts` ;
`02-jeux.md` → `src/core/games/` + `drl-game-board` ; `03-algorithmes.md` →
`src/core/search/` ; `04-arbre.md` → `tree-model.ts`, `stepper.ts` +
`drl-eval-tree`. Les écarts tranchés à l'implémentation sont listés en
section 6.

Scripts (`demos/package.json`) :

| Commande | Rôle |
|---|---|
| `npm run dev` | Vite : galerie playground avec HMR (ouvre `/playground/index.html`) |
| `npm run build` | Bundle ES `dist/drl-demos.js` (+ chunks, dont `drl-three.js` si une démo 3D existe) |
| `npm run test` / `test:watch` | Vitest (core pur uniquement) |
| `npm run typecheck` | `tsc --noEmit` (strict) |
| `npm run lint` / `format` | Biome |
| `npm run new -- <nom>` | Générateur plop (hors périmètre Deep Blue) |

Intégration au site statique : après `npm run build`, copier tous les fichiers
de `demos/dist/` côte à côte, puis :

```html
<script type="module" src="assets/drl-demos.js"></script>
<drl-deep-blue-page game="connect4" depth="4" evaluator="windows"></drl-deep-blue-page>
```

L'import du module enregistre tous les composants. Deep Blue ne dépend ni de
Three ni d'un chunk séparé : D3 (arbre) et Lit sont dans le bundle principal
`drl-demos.js`.

## 2. Démarrage rapide (HTML)

### Page complète : `<drl-deep-blue-page>`

Une balise suffit : à gauche la partie interactive, à droite l'arbre
d'évaluation. Chaque recherche d'agent est rejouée pas à pas dans l'arbre ;
cliquer un nœud propose « Rejouer ce coup » (premier coup du chemin menant à
ce nœud, appliqué dans la partie).

| Attribut | Type | Défaut | Rôle |
|---|---|---|---|
| `game` | `'tic-tac-toe' \| 'connect4'` | `'tic-tac-toe'` | jeu affiché ; toute autre valeur retombe sur `'tic-tac-toe'` |
| `players` | string JSON | `'["human","agent"]'` | passé tel quel au composant de jeu (voir plus bas) |
| `depth` | number | `null` → défaut du jeu (morpion 6, puissance 4 : 4) | horizon de recherche en plies |
| `evaluator` | `'windows' \| 'rollout' \| 'random' \| 'custom'` | `'windows'` | passé tel quel au composant de jeu |
| `rollout-samples` | number | `10` | échantillons de rollout par évaluation ; transmis au jeu (champ « Échantillons de rollout », visible seulement si `evaluator="rollout"`) |
| `discount` | number | `1` | actualisation γ ∈ ]0, 1] ; pondère la valeur de chaque coup enfant à la remontée ; bornée à [0.01, 1] (retombe à `1` si invalide) ; champ « Actualisation (γ) » masqué quand `evaluator="random"` |
| `auto-fit` | boolean | `true` | case « Ajuster auto » de l'arbre cochée par défaut (recentre et ajuste le zoom à chaque mise à jour de l'arbre) |
| `transposition` | boolean | `false` | active la table de transposition |
| `pruning` | boolean | `false` | `false` = minimax pur (sans élagage) ; `true` = alpha-beta |
| `strict-pruning` | boolean | `false` | élagage strict (`> β` / `< α`) ; transmis au jeu **et** aux deux agents rejoués dans l'arbre, pour que l'arbre enregistré corresponde à la recherche jouée |

La page possède ses propres contrôles (le HTML de démo ne les contient plus) :
sous son titre et le sélecteur de jeu, elle affiche la rangée des modes
(Humain vs Agent / Agent vs Humain / Agent vs Agent) puis les paramètres de
recherche — profondeur, évaluateur, échantillons de rollout (si
`evaluator="rollout"`), actualisation (γ), élagage, élagage strict (sauf
`evaluator="random"`) et transposition. Quand `evaluator="custom"`, un bloc
pleine largeur apparaît sous la rangée de recherche (section 4.4). Elle passe
`show-search-fields=false` au
composant de jeu embarqué, donc la zone de jeu ne répète pas les champs
« Échantillons de rollout », « Actualisation » et « Élagage strict ». Le
sélecteur « Élagage » fixe un booléen : choisir minimax désactive l'alpha-bêta.

```html
<drl-deep-blue-page
  game="connect4"
  players='["agent","human"]'
  depth="4"
  evaluator="windows"
  transposition
></drl-deep-blue-page>
```

### Parties isolées : `<drl-tic-tac-toe>` / `<drl-connect4>`

Mêmes attributs que la page, plus `cell-size` (voir section 4.2). Exemple
reprenant le playground `demos/playground/drl-tic-tac-toe.html` :

```html
<drl-tic-tac-toe players='["human","agent"]' depth="6" evaluator="windows"></drl-tic-tac-toe>
<script type="module">
  const game = document.querySelector('drl-tic-tac-toe');
  game.addEventListener('drl-move-played', (e) => console.log(e.detail.move));
  game.addEventListener('drl-game-over', (e) => console.log(e.detail.outcome));
</script>
```

Les pages `demos/playground/drl-deep-blue-page.html`, `drl-tic-tac-toe.html`,
`drl-connect4.html`, `drl-eval-tree.html` sont des exemples complets
(événements journalisés, réglages pilotés en JS), servies par `npm run dev`.

## 3. Bibliothèque core (TypeScript)

Tout vit dans `demos/src/core/` : TypeScript strict, aucun import DOM/Lit/D3.
Les imports relatifs utilisent l'extension `.js`.

### 3.1 Le contrat `Game`

```ts
export interface Game<S, M> {
  readonly winScore: number;                    // sentinelle de victoire (= 1)
  initialState(): S;
  legalMoves(state: S): readonly M[];           // ordre stable = ordre DFS
  apply(state: S, move: M): S;                  // immuable, jamais in place
  undo(state: S, move: M): S;
  currentPlayer(state: S): PlayerId;            // PlayerId = number (0 | 1 ici)
  winner(state: S): GameOutcome;                // PlayerId | 'draw' | null
  stateKey(state: S): string;                   // clé canonique (TT)
  boardView(state: S): BoardView;
  pieceCount?(state: S): number;                // optionnel : nombre de pièces posées (tour intra-partie)
}
```

`pieceCount?(state)` est optionnel : il donne le nombre de pièces posées,
c'est-à-dire le tour intra-partie (une pièce par coup dans les deux jeux). Le
moteur l'enregistre dans le champ `pieces` des entrées de table de
transposition (section 3.4), pour la colonne « Pièces » du navigateur
(section 4.2). Un jeu qui ne l'implémente pas laisse le champ absent.

Deux instances livrées :

| Instance | État | Coup | `stateKey` |
|---|---|---|---|
| `ticTacToe` (`games/tic-tac-toe.ts`) | `TicTacToeState` : `cells` (9 cellules) + `currentPlayer` | `{ index: 0..8 }` | plus petite des 8 symétries (rotation/réflexion) + joueur au trait |
| `connect4` (`games/connect4.ts`) | `Connect4State` : `grid` 6×7, `currentPlayer`, `lastMove`, `history` | `{ col: 0..6 }` | plus petite des deux formes (normale / miroir) + joueur au trait |

Sentinelles exportées par `contracts.ts` : `WIN_SCORE = 1`, `LOSS_SCORE = -1`,
`DRAW_SCORE = 0`, `WIN_SCORE_MARGIN = 0.95` (borne exclusive des scores
heuristiques), plus `noopRecorder` (recorder no-op pour une recherche sans
visualisation). `makeBoardView(rows, cols, cells, { highlight?, clickable? })`
(`games/board-view.ts`) construit une `BoardView` défensive (copies) ;
`games/types.ts` réexporte contrats et board-view.

### 3.2 Agent minimax / alpha-beta

```ts
import { createMinimaxAgent } from './core/search/minimax.js';

const agent = createMinimaxAgent(ticTacToe, {
  depth: 6,                      // horizon en plies
  evaluator: ticTacToeHeuristic, // appelé aux feuilles non terminales
  prune: true,                   // false = minimax pur (sans élagage)
  strictPruning: false,          // true = coupe seulement sur > β / < α
  recorder: noopRecorder,        // ou tree.recorder pour la visualisation
  transposition,                 // optionnel : TranspositionTable
});
```

`MinimaxOptions<S, M>` : `{ depth, evaluator, prune, recorder,
transposition?, strictPruning?, rng? }`.
Un seul moteur : alpha-beta est minimax avec fenêtre `[α, β]` resserrée. Les
fils sont visités dans l'ordre de `legalMoves` (DFS strict, déterministe),
ce qui rend le stepper possible. Un nœud est « maximisant » si son joueur au
trait est `rootPlayer` ; les scores émis par le moteur restent du point de vue
de `rootPlayer` (pas de négation négamax).

`strictPruning: false` (défaut) coupe sur `bestScore ≥ β` (max) ou `≤ α`
(min). Les ex æquo sont départagés au hasard dès que leurs valeurs sont
exactes : c'est le cas en **minimax pur** (`prune: false`, tous les enfants
évalués) comme en **élagage strict** (`> β` / `< α`). Le tirage est uniforme
(échantillonnage réservoir) et, sans `rng` fourni, utilise `Math.random`.
L'**alpha-bêta classique** (`prune: true`, `strictPruning: false`) reste au
contraire déterministe sans `rng` : ses ex æquo peuvent être un artefact de
fail-soft, et retenir le premier coup visité rend deux arbres comparables à
l'identique. `strictPruning` est aussi passé à `isEntryUsable`, qui refuse
alors une entrée bornée dont le score égale simplement la borne.

Consommation step-chunkée (motif des tests `minimax.test.ts`) :

```ts
agent.start(state);                      // reset implicite, peut être rappelé
while (agent.step() === 'running') { /* une itération = un nœud visité */ }
const move = agent.bestMove();           // null tant que pas 'done'
const stats = agent.stats();
```

`SearchStats` (tous champs en lecture seule) : `nodesVisited`, `nodesPruned`,
`ttHits`, `ttMisses`, `leavesEvaluated`, `elapsedSteps`, `effectiveDepth`
(profondeur certifiée de la valeur de racine ; `SOLVED_DEPTH` si la racine est
entièrement résolue). Un `step()` de l'agent
correspond à un `yield` du générateur interne : c'est l'unité de travail que
le composant multiplie par un budget (voir 4.2).

Agent aléatoire (baseline jouable) :

```ts
import { createRandomAgent } from './core/search/random-agent.js';
const agent = createRandomAgent(game);          // option : { rng?: () => number }
```

Un seul `step()` puis `'done'` ; coup légal uniforme ; RNG mulberry32 seedé
(6203) par défaut, donc reproductible.

### 3.3 Évaluateurs

Un `Evaluator<S, M>` expose `id: string` et
`evaluate(state, ctx: EvalContext<S, M>): number`, avec
`EvalContext = { rootPlayer, game }`. Il n'est **jamais** appelé sur un état
terminal.

| Évaluateur | `id` | Valeur |
|---|---|---|
| `ticTacToeHeuristic` | `'zero'` | 0 (le morpion se résout au terminal) |
| `connect4Heuristic` | `'windows'` | comptage des 69 fenêtres de 4 cases ; 1/2/3 pions valent 1/10/50, signé `rootPlayer`, divisé par 1000, borné à ±0.9499 |
| `createRandomRolloutEvaluator({ rng?, samples?, maxPlies? })` | `'random-rollout'` | moyenne de `samples` playouts aléatoires (±1/0) ; défauts : graine 4157, `samples = 1`, `maxPlies = 200` |

### 3.3.1 Évaluateur personnalisé (JavaScript)

`compileCustomEvaluator(source, game, gameId)` compile le **corps** d'une
fonction `evaluate(state, ctx)` écrit à la main et renvoie
`{ ok: true, evaluator }` ou `{ ok: false, error }`. `gameId` est un
`CustomGameId` : `'tic-tac-toe'` ou `'connect4'`. Le corps est compilé une
seule fois avec `new Function('state', 'ctx', '"use strict";\n' + source)`,
jamais à chaque évaluation.

Le corps reçoit deux arguments :

| Argument | Contenu |
|---|---|
| `state` | l'état du jeu : `{ cells, currentPlayer }` (morpion) ou `{ grid, currentPlayer, lastMove, history }` (puissance 4) |
| `ctx` | `rootPlayer` et `opponent` (joueurs), `rows` et `cols` (dimensions), `game` (le `Game`), puis `lines` (morpion : les 8 triplets gagnants) ou `windows` (puissance 4 : les 69 quadruplets de 4 cases) |

Le corps doit retourner un **nombre fini**. Le wrapper borne chaque résultat à
`[-0.9499, 0.9499]` (`CUSTOM_BOUND`) ; un résultat non fini ou une exception
levée pendant la recherche retombe sur `0`.

À la compilation, une sonde exécute le corps sur `game.initialState()` pour
`rootPlayer` 0 puis 1. Une erreur de syntaxe, une exception à la sonde, ou un
résultat non numérique/non fini à la sonde renvoie `{ ok: false, error }`
(`"L'évaluateur doit retourner un nombre fini"` dans ce dernier cas) : le corps
n'entre jamais en recherche tant qu'il n'est pas valide.

`defaultCustomSource(gameId)` renvoie le corps par défaut du jeu actif :
l'implémentation `windows` (comptage des fenêtres) pour le puissance 4, et
`return 0` pour le morpion (heuristique nulle : le moteur résout la partie au
terminal).

Le composant de jeu expose la propriété `customEvaluator` (sans attribut),
utilisée quand `evaluator="custom"` ; la page la renseigne (section 4.4).

**Sécurité** — `new Function` s'exécute dans le realm de la page : le corps peut
lire les globales, le stockage, les cookies et le réseau. Ce n'est **pas** un
bac à sable. Réservé aux démos locales : ne collez pas de code non fiable.

### 3.4 Table de transposition

```ts
import {
  createTranspositionTable,
  isEntryUsable,
  SOLVED_DEPTH,
} from './core/search/transposition.js';

const tt = createTranspositionTable();
// tt.store(key, entry) | tt.probe(key) | tt.clear() | tt.size()
// tt.beginTurn() | tt.currentTurn() | tt.snapshot()
```

`TranspositionEntry` : `{ score, depth, exact, samples?, pieces? }`. Le score est vu du
**joueur au trait** dans l'état (perspective intrinsèque), pas du joueur
racine : la clé est `game.stateKey(state)` seul, sans préfixe de joueur, et
l'entrée sert d'un tour à l'autre. `store` conserve l'entrée la plus profonde
(`entry.depth ≥ existante`). `depth` est la **profondeur effective certifiée**
(nombre de plies pour lesquels `score` est exact) ; une position résolue au
terminal porte le sentinelle `SOLVED_DEPTH` (`Infinity`). `samples` n'existe
que pour les feuilles rollout (cache à profondeur 0). `pieces` est le nombre de
pièces sur le plateau à cette position ; le moteur le renseigne à chaque
`store` seulement si le `Game` implémente `pieceCount`, sinon le champ reste
absent.

`isEntryUsable(entry, remainingDepth, alpha, beta, maximizing, currentTurn, strict = false)` :
hit valable ssi `entry.depth > remainingDepth` et (`exact` ou borne compatible
avec la fenêtre du nœud courant — `score ≥ beta` si maximisant, `score ≤ alpha`
sinon). À profondeur **égale**, l'entrée n'est réutilisée que si elle porte le
tour courant (`entry.turn === currentTurn`) : les tours précédents ne
fournissent que du strictement plus profond. C'est ce strictement-plus-profond
qui augmente la profondeur effective. Seule exception : la feuille rollout à
profondeur 0 (cache-feuille), qui étend la moyenne mémorisée et met à jour
`samples`.
`currentTurn` sert donc à la fois à cette clause de profondeur égale et à
l'option `probeTurns: 'previous'`, disponible pour un agent qui refuserait en
plus les entrées du tour courant. Le rejoué Deep Blue n'utilise pas cette
option : il applique la politique par défaut `'all'`, lit la table partagée en
lecture seule (`createReadOnlyOverlay`) et écrit ses propres entrées dans une
table temporaire privée, si bien que la table réelle n'est jamais polluée.

### 3.5 Instrumenter une recherche : `TreeModel` et `createStepper`

```ts
import { TreeModel, createModelRecorder } from './core/search/tree-model.js';
import { createStepper } from './core/search/stepper.js';

const model = new TreeModel<S>();
const agent = createMinimaxAgent(game, { depth, evaluator, prune: true,
                                         recorder: createModelRecorder(model) });
const stepper = createStepper(agent, model);
stepper.start(state);
stepper.stepIn();      // ou stepOver() / stepOut()
stepper.current();     // clé du nœud courant, ou null
stepper.node();        // TreeNode<S> | null
stepper.path();        // pile de contexte (clés racine → courant)
stepper.isDone();
```

`TreeModel<S>` : `insert(VisitEvent)`, `applyValue(ValueEvent)`, `root()`,
`node(key)`, `children(key)`, `has(key)`, `resolved(key)`, `size()`,
`lastVisited()`, `clear()`. Fusion par `nodeKey` (un nœud revisité met à jour
`kind`/score sans dupliquer) ; les nœuds `pruned`/`tt-hit` restent sans
descendants. `createModelRecorder(model)` retourne le `TreeRecorder<S>` à
brancher sur n'importe quel agent instrumenté.

## 4. Référence des composants web

Tous les événements sont des `CustomEvent` avec `bubbles: true, composed:
true`. Les composants de jeu et l'arbre héritent de `DemoElement`, qui fournit
les attributs `running` (boolean, réflexif) et `speed` (number > 0, défaut 1),
les contrôles play/pause/step/reset (`drl-play`, `drl-pause`, `drl-step`,
`drl-reset`, `drl-speed`) et la boucle ~30 ticks/s × `speed`.

### 4.1 `<drl-game-board>`

Rendu SVG d'une `BoardView`, sans aucune logique de jeu. Utilisé en pleine
taille (jeu interactif) et en miniature 24 px (nœuds de l'arbre).

| Attribut / propriété | Type | Défaut | Notes |
|---|---|---|---|
| `view` | `BoardView \| null` | `null` | propriété seulement (pas d'attribut) ; rendu vide si `null` ou grille invalide |
| `cell-size` | number | `48` | ramené à 4 minimum |
| `piece-style` | `'discs' \| 'marks'` | `'discs'` | disques pleins (joueur 0 accent / joueur 1 accent-2) ou croix/rond |

Événement : `drl-cell-click` — `detail: { row: number, col: number }`, émis
uniquement pour les coordonnées listées dans `view.clickable` (survol
surligné, curseur pointer). `highlight` dessine un cadre accent. Le mapping
cellule→coup est la responsabilité de l'hôte.

### 4.2 `<drl-tic-tac-toe>` / `<drl-connect4>`

Deux instanciations de la factory `createGameComponent` (section 5.4) ;
leur API commune est `GameComponentApi<S, M>` (alias
`DrlTicTacToeApi` / `DrlConnect4Api`).

| Attribut | Type | Défaut (morpion / P4) | Rôle |
|---|---|---|---|
| `players` | string JSON | `'["human","agent"]'` | tableau de 2 kinds : `"human"`/`"agent"` ou `{ "kind": … }` ; entrée invalide → défaut |
| `depth` | number | 6 / 4 | horizon des agents ; non fini → défaut du jeu (`sanitizeDepth`) |
| `evaluator` | string | `'windows'` | `'windows'` \| `'rollout'` \| `'random'` \| `'custom'` (autre valeur → `'windows'`) |
| `customEvaluator` | `Evaluator<S, M> \| null` | `null` | propriété seule (pas d'attribut) ; évaluateur utilisé quand `evaluator="custom"` (repli sur l'heuristique du jeu si `null`) |
| `transposition` | boolean | `false` | table de transposition pour les agents minimax |
| `strict-pruning` | boolean | `false` | élagage strict (`> β` / `< α`) ; case à cocher « Élagage strict » dans la barre d'outils (masquée quand `evaluator="random"`) |
| `pruning` | boolean | `true` | `false` = minimax pur (sans élagage) ; `true` = alpha-beta |
| `rollout-samples` | number | `1` | échantillons de rollout par évaluation ; champ « Échantillons de rollout » de la barre d'outils (visible si `evaluator="rollout"`) |
| `discount` | number | `1` | actualisation γ ∈ ]0, 1] ; pondère la valeur de chaque coup enfant à la remontée ; bornée à [0.01, 1] (retombe à `1` si invalide) ; champ « Actualisation (γ) » de la barre d'outils (masqué quand `evaluator="random"`) |
| `show-search-fields` | boolean | `true` | `false` masque de la barre d'outils les champs « Échantillons de rollout », « Actualisation » et « Élagage strict » |
| `cell-size` | number | 72 / 48 | taille des cases du plateau |

Propriétés en lecture : `game`, `title`, `pieceStyle`, `playerNames`
(`readonly [string, string]`), `playerKinds` (kinds effectifs), `currentState`
(`S | null`), `history` (`readonly S[]`), `gameOutcome`, `seriesScore`
(`SeriesScore` : `{ wins: readonly [number, number], draws: number }`),
`transpositionTable` (`TranspositionTable`), `searchStats`
(`SearchStats | null`). Propriétés mutables : `recorder` (`TreeRecorder<S> |
null`, sans attribut) — le brancher avant le coup d'agent pour alimenter un
`<drl-eval-tree>` ; `customEvaluator` (`Evaluator<S, M> | null`, sans attribut)
— l'évaluateur utilisé quand `evaluator="custom"` (repli sur l'heuristique du
jeu si `null`).

Méthodes : `applyMove(move: M): boolean` (coup légal uniquement, refuse si
partie finie ou recherche en cours), `clearTranspositionTable()` (vide la table
et les mini-plateaux mémorisés) et `reset()` (héritée). `step()` est géré
par la boucle `DemoElement` : chaque tick consomme
`min(256, max(1, round(speed × 32)))` itérations de l'agent en cours, joue
`bestMove()` à `'done'`. Changer `players`/`depth`/`evaluator`/`transposition`/
`strictPruning` annule la recherche en attente.

Événements (charges exactes) :

| Événement | `detail` |
|---|---|
| `drl-move-played` | `{ move: M, state: S, player: PlayerId }` — `state` = état **après** le coup |
| `drl-search-started` | `{ state: S, agent: SearchAgent<S, M>, player: PlayerId }` |
| `drl-game-over` | `{ outcome: GameOutcome }` |
| `drl-move-undone` | `{ plies: number, state: S }` — s'il y a un humain, remonte jusqu'à un état où c'est à lui de jouer (typiquement 2 plies en humain vs agent), sinon 1 ply |
| `drl-invalid-move` | `{ row: number, col: number }` — clic hors `clickableCells` ou coup illégal |
| `drl-game-reset` | `{ state: S }` — état initial reconstruit |

Série de parties : quand les deux joueurs sont `agent`, la barre de statut
ajoute une ligne `Série — <nom0> <w0> · <nom1> <w1> · Nulles <draws>`, dont
`seriesScore` porte le détail. Le tally s'incrémente à chaque fin de partie.
`reset()` et un changement de `players` remettent la série à zéro ; la relance
automatique (`autoRestart`) ne la remet pas à zéro, donc la série s'accumule
entre les parties relancées. Annuler le coup final d'une partie terminée retire
son résultat de la série (le compteur ne passe pas sous zéro) ; si les agents
terminent ensuite la même partie, elle n'est comptée qu'une fois.

Table de transposition : quand `transposition` est actif, le bouton
« Parcourir la TT » de la barre d'outils ouvre un dialogue « Table de
transposition ». Il liste les entrées de `snapshot()` : Position (mini-plateau),
Clé, Score, Profondeur eff., Éch., Exact, Tour et Pièces (le nombre de pièces,
ou `–` si le jeu n'implémente pas `pieceCount`). Les en-têtes numériques
(Score, Profondeur eff., Éch., Exact, Tour, Pièces) trient la liste : premier
clic croissant, second clic sur le même en-tête décroissant, l'en-tête actif
portant `▲` ou `▼` ; sans tri, l'ordre est celui de `snapshot()` (par clé). Le
tableau affiche 50 lignes par page, avec « Précédent », « Page X / Y » et
« Suivant » (désactivés aux bornes). Un filtre par clé restreint les lignes.
L'ordre appliqué est filtre, puis tri, puis tranche de page ; un changement de
filtre ou de tri revient à la page 1, et l'ouverture du dialogue remet à zéro le
filtre, le tri et la page. La note de bas de dialogue affiche la plage affichée
et le total retenu.

### 4.3 `<drl-eval-tree>`

Visualiseur d'arbre branché sur n'importe quel agent instrumenté. La classe
`DrlEvalTree<S>` est générique : côté TypeScript, manipulez-la en
`DrlEvalTree<unknown>` (c'est ce que fait `drl-deep-blue-page`).

| Attribut / propriété | Type | Défaut | Rôle |
|---|---|---|---|
| `max-depth` | number | `∞` (`Number.POSITIVE_INFINITY`) | profondeur **d'affichage** ; au-delà, nœuds collapsés « +N » |
| `node-size` | string CSS | `'3rem'` | largeur des cartes nœuds (mesurée, repli 48 px) |
| `stepper` | boolean | `true` | affiche les boutons Entrer / Passer / Sortir |
| `piece-style` | `'discs' \| 'marks' \| null` | `null` → auto : `'marks'` si plateau ≤ 4×4, sinon `'discs'` | style des miniatures |

Lecture : `model` (`TreeModel<S>`), `recorder` (`TreeRecorder<S>` prêt à
brancher sur un agent). Méthodes publiques :

| Méthode | Effet |
|---|---|
| `attach(game)` | fournit le `Game` (rendu `game.boardView` des nœuds) |
| `startSearch(agent, state)` | vide le modèle, crée le stepper, démarre l'agent sur `state` |
| `clearTree()` | vide modèle, stepper et agent lié |
| `setViewRoot(key \| null)` | re-racine l'affichage (clé inconnue ignorée) ; émet `drl-root-changed` |

Événement : `drl-root-changed` — `detail: { nodeKey: string | null }` (clic
nœud, double-clic fond, bouton « Racine = état courant », fil d'Ariane).

Contrôles intégrés : pas-à-pas débogueur (`stepper`), zoom d3-zoom (extent
0.15–4, slider + « Ajuster » + « 100 % »), champ « Prof. max », fil d'Ariane,
panneau α/β du nœud de détail (score, fenêtre, coupure et coup déclencheur),
barre de statut (nœuds, masqués, curseur, état). `step()` applique la
primitive sélectionnée (in/over/out) ; `running`/`speed` hérités pilotent la
replay continue. Rendu plafonné à 4000 nœuds affichés (au-delà : nœuds
compacts comptés).

L'arbre n'a **aucun nouvel attribut** pour l'élagage strict : il lit
`window`/`cut` dans les événements et affiche les détails quel que soit le
mode du moteur.

### 4.4 `<drl-deep-blue-page>`

Composition jeu + arbre, câblage interne : à chaque `drl-search-started` du
composant de jeu, la page crée un agent parallèle (mêmes options, recorder =
`tree.recorder`) et appelle `tree.startSearch(agent, state)` ; à
`drl-game-reset` et `drl-move-undone` elle vide l'arbre. Elle relaie aussi le
**Step** de la partie vers l'arbre : quand l'arbre est en pause, `drl-step` du
composant de jeu déclenche un pas de l'arbre (primitive `in` par défaut) ;
l'arbre garde ses propres contrôles, toujours actifs. Exports utiles :
`firstMoveOnPath(model, nodeKey): unknown` (premier coup du chemin
racine→nœud, `null` si racine ou clé inconnue) et
`DeepBlueGameId = 'tic-tac-toe' | 'connect4'`. Attributs : voir section 2 —
`strict-pruning` y est transmis à la fois au composant de jeu et aux deux
agents rejoués, pour que l'arbre enregistré reflète la recherche réellement
jouée. Elle n'hérite pas de `DemoElement` : les contrôles de lecture sont ceux
des composants enfants.

Quand `evaluator="custom"`, la page affiche sous la rangée de recherche un bloc
pleine largeur « Évaluateur personnalisé (JavaScript) » : une aide mémoire des
arguments, un avertissement local (rappel complet : section 3.3.1), un
`textarea` contenant le **corps** de la
fonction `evaluate(state, ctx)` (contrat : section 3.3.1), les boutons
**Appliquer** et **Réinitialiser**, et une ligne d'erreur. **Appliquer** compile
le brouillon et l'applique : une erreur de syntaxe, une exception à la sonde sur
l'état initial, ou un résultat non fini s'affiche alors dans la ligne d'erreur et
le dernier évaluateur valide reste actif. **Réinitialiser** restaure la source par
défaut du jeu et la recompile. Les brouillons et les évaluateurs compilés sont
conservés **par jeu** : changer de jeu restaure le brouillon et l'évaluateur de
ce jeu, et revenir à `custom` retrouve l'état précédent. Appliquer vide l'arbre
rejoué et change `customEvaluator`, ce qui annule la recherche en attente dans le
composant de jeu. L'évaluateur personnalisé sert à la fois la recherche en direct
et celle de l'arbre rejoué, avec la même profondeur, la même actualisation, le
même élagage et le même élagage strict.

## 5. Étendre la bibliothèque

### 5.1 Ajouter un `Game`

Implémenter les 9 membres de `Game<S, M>` (section 3.1) et vérifier les
invariants avec le harnais générique :

```ts
import { describeGameContract } from './core/testing/game-contract.js';
describeGameContract(monJeu, 'mon-jeu'); // options : { seed, corpusSize, minPly, maxPly }
```

- états immuables, `undo(apply(s, m), m) === s`, `currentPlayer` alterné ;
- `legalMoves` exhaustif et rejouable ; ordre **stable** (il détermine l'ordre
  DFS de la recherche et le rendu de l'arbre) ;
- `stateKey` canonique : transpositions équivalentes → même clé, états d'une
  même trajectoire → clés distinctes ; incluez le joueur au trait ;
- `boardView` cohérent avec la grille (`makeBoardView`) ; `highlight` pour le
  dernier coup (Connect4) ; `clickable` est rempli par le composant de jeu.

`core/testing/playouts.ts` fournit `createRng`, `randomState(s)`,
`randomPlayout`, `pick` pour construire des corpus de test.

### 5.2 Ajouter un `Evaluator`

Un objet `{ id, evaluate(state, ctx) }`. Contrat : jamais appelé sur un
terminal ; résultat du point de vue de `ctx.rootPlayer` ; strictement dans
`] -WIN_SCORE_MARGIN, +WIN_SCORE_MARGIN [` (voir `clampHeuristicScore`).
Stochastique accepté (rollout) : RNG seedé recommandé.

### 5.3 Ajouter un `SearchAgent`

Implémenter `{ start, step, bestMove, stats }`. Invariants que le
visualiseur suppose :

- ordre de visite **DFS, un nœud par `step()`**, fils dans l'ordre de
  `legalMoves` — c'est lui que rejoue le stepper ;
- chaque visite notée au recorder avec un `VisitEvent` (`nodeKey =
  game.stateKey(state)`, `parentKey`, `move`, `state`, `depth` = plies depuis
  la racine, `kind` parmi `'expand' | 'pruned' | 'tt-hit' | 'leaf' |
  'terminal'`) ; les scores via `ValueEvent` (`exact` uniquement quand la
  valeur du nœud est définitive) ;
- scores du point de vue de `rootPlayer`, sentinelles du contrat (±1 / 0) ;
- `start` rappelable à tout moment (abandon propre de la recherche en cours) ;
  `bestMove()` = `null` hors `'done'` et coup ∈ `legalMoves(state initial)`.

### 5.4 Composant de partie pour un jeu custom

```ts
import { createGameComponent, type GameComponentConfig } from './components/shared/create-game-component.js';

const config: GameComponentConfig<MonEtat, MonCoup> = {
  game: monJeu,
  title: 'Mon jeu',
  playerNames: ['Blancs', 'Noirs'],
  defaultDepth: 4,             // défaut `depth` et repli `sanitizeDepth`
  defaultCellSize: 48,         // défaut `cell-size`
  pieceStyle: 'discs',         // 'discs' | 'marks'
  clickableCells(state) { … }, // coordonnées cliquables pour un humain
  moveForCell(row, col, state) { … }, // coordonnée → coup, ou null
  heuristic: monEvaluateur,    // évaluateur du mode 'windows'
};
const MonJeuElement = createGameComponent(config);
customElements.define('drl-mon-jeu', MonJeuElement);
```

Le composant obtenu hérite de toute la mécanique (humain/agent, undo,
statistiques, événements de la section 4.2, branchement `recorder`). Les
helpers exportés réutilisables : `createGameAgent(game, { mode, depth,
fallbackDepth, transposition, recorder, heuristic })` (mode `'random'` →
`createRandomAgent`, `'rollout'` → rollout + minimax, sinon `heuristic` +
minimax, `prune: true`), `parsePlayersConfig`, `parseEvaluatorMode`,
`sanitizeDepth`, `sameMove`, et les configs prêtes à l'emploi
`TIC_TAC_TOE_GAME_CONFIG` / `CONNECT4_GAME_CONFIG`. Un `Game` custom peut
implémenter `pieceCount?(state)` pour alimenter la colonne « Pièces » du
navigateur de table de transposition ; sans lui, la colonne affiche `–`.

## 6. Notes d'implémentation et limites

Écarts et clarifications par rapport aux spécifications :

- **`players`** — la spec `02-jeux.md` décrivait un JSON
  `[PlayerConfig, PlayerConfig]` avec agents injectables et `stepBudget`.
  L'implémentation n'accepte que des kinds (`"human"`/`"agent"` ou
  `{ "kind": … }`) : les agents sont construits en interne par
  `createGameAgent`. `PlayerConfig.stepBudget` existe toujours dans
  `contracts.ts` mais n'est pas consommé ; le budget par tick dérive de
  `speed` (32 itérations/unité, plafond 256).
- **`ValueEvent.depth`** — contrairement à `VisitEvent.depth` (plies depuis
  la racine), le `depth` des `ValueEvent` porte la profondeur **restante**
  jusqu'à l'horizon au moment de l'évaluation.
- **Table de transposition** — clé `game.stateKey(state)` seule (plus de
  préfixe `rootPlayer:`) : les entrées sont partagées entre tours et entre
  recherches démarrées par des joueurs différents. Les scores sont mémorisés du
  point de vue du **joueur au trait** (perspective intrinsèque). La validité
  d'un hit distingue la direction du nœud (`maximizing`), `exact` signant une
  valeur non coupée, et `depth` une **profondeur certifiée** (`SOLVED_DEPTH`
  pour une position résolue au terminal). Un hit exige une profondeur
  **strictement supérieure** ; à profondeur égale, seule une entrée du tour
  courant est consommée (une entrée d'un tour antérieur doit être plus
  profonde). Exception : la feuille rollout à profondeur 0, dont le cache
  étend la moyenne mémorisée.
- **Persistance de l'arbre** — la spec 04 prévoyait un modèle vidé « à chaque
  coup ». L'implémentation le vide au **démarrage de la recherche suivante**
  (`startSearch`) et sur reset/undo ; après un coup sans recherche (tour
  humain), l'arbre précédent reste affiché.
- **`node-size`** — chaîne CSS (et non nombre), défaut `'3rem'` ; les valeurs
  par jeu (morpion `'4rem'`, puissance 4 `'3rem'`) sont appliquées par
  `drl-deep-blue-page`, pas par l'attribut lui-même.
- **`max-depth`** — champ numérique (la spec mentionnait un slider) ; vide =
  illimité, minimum 1.
- **Évaluateur morpion** — évaluateur dédié `id: 'zero'` (et non « le même
  fichier qui retourne 0 ») : même sémantique, autre objet.

Limites connues :

- **Mode `evaluator="random"`** — `createRandomAgent` n'accepte pas de
  recorder : l'agent parallèle de l'arbre n'émet aucun événement et
  `<drl-eval-tree>` reste vide. La partie se joue, la visualisation non.
- **Modes `rollout` / `random` : rejeu de l'arbre** — l'arbre est construit
  par un agent **parallèle** (nouvelle instance) et non par l'agent qui joue
  réellement. Avec un évaluateur stochastique, l'égalité des deux recherches
  n'est garantie aujourd'hui que parce que les deux instances utilisent les
  mêmes graines par défaut (4157 / 6203) et le même état de départ ; rien
  dans le contrat ne l'impose (un `rng` injecté divergerait).
- **Volumétrie de l'arbre** — 4000 nœuds affichés au maximum (collapsés en
  nœuds compacts `+N` au-delà de `max-depth` ou de ce plafond) ; canevas de
  460 px avec zoom 0.15–4 : les recherches très larges se lisent par
  zoom/navigation, pas en un écran.
- **`bestMove()` après `'done'`** peut être `null` (racine sans coup légal) ;
  le composant de jeu vérifie la légalité avant de jouer et ignore sinon le
  coup.
- **Visualiseur — frères élagués masqués** — l'arbre ne dessine que le premier
  frère `pruned` d'un nœud ; les autres frères élagués ne sont pas dessinés
  mais restent dans le modèle de l'arbre, donc le compteur « Nœuds » de la barre
  de statut du visualiseur les compte. La statistique « coupures », elle, est
  une statistique du moteur affichée dans le panneau de jeu
  (`SearchStats.nodesPruned`) : elle compte aussi ces frères masqués. Le
  compteur « Masqués » du visualiseur ne compte que les nœuds repliés par
  `max-depth` (ou par le plafond de 4000 nœuds affichés). Le curseur du stepper
  peut se poser sur un nœud masqué, qui n'a alors aucune surbrillance.
