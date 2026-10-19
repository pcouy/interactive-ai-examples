# Deep Blue — Visualiseur d'arbre d'évaluation

`<drl-eval-tree>` : visualiseur **branchable** sur n'importe quel
`SearchAgent` instrumenté (contrat `01-contrats.md`). Il consomme la
séquence d'événements `TreeRecorder` émise pendant la recherche et
reconstruit l'arbre en mémoire, au fur et à mesure, dans l'ordre **DFS
réel de l'algorithme** — pas une construction abstraite par niveaux.

## Structure du composant

```
src/components/drl-eval-tree/drl-eval-tree.ts   // composant Lit
src/core/search/tree-model.ts                   // arbre en mémoire (pur)
src/core/search/tree-model.test.ts
```

Le composant ne connaît ni minimax ni TicTacToe : il consomme des
`VisitEvent`/`ValueEvent` génériques. Le **modèle d'arbre** (`tree-model.ts`,
core pur) :

- insertion incrémentale : un `VisitEvent` crée un nœud sous `parentKey` ;
- chaque nœud porte : `state` (pour le rendu `BoardView`), `move`, `depth`,
  `kind` (`expand`/`pruned`/`tt-hit`/`leaf`/`terminal`), le meilleur score
  courant, `exact`, l'ordre DFS (séquence d'insertion), la fenêtre `[α, β]`
  effective et, pour un nœud coupé, le `CutInfo` de la coupure ; une valeur
  exacte efface un `cut` devenu obsolète ;
- un nœud peut être **visité plusieurs fois** (transpositions, révisions de
  score) : le modèle fusionne par `nodeKey` ; la position dans l'arbre reste
  la **première** insertion (l'arbre est un arbre de coups, pas un DAG).

## Rendu

- **d3-hierarchy** `tree()` (tidy tree), orientation **verticale**, racine
  en haut.
- Chaque nœud = **miniature fidèle** du plateau : le même composant
  `<drl-game-board>` (`02-jeux.md`) à échelle réduite, alimenté par
  `game.boardView(node.state)` — pas de rendu dédié, pas de duplication.
  Densité réglable par `node-size` (voir Contrôles).
- Sur chaque nœud s'ajoutent, quand disponibles :
  - le **score courant** (meilleur connu), en couleur (accent si positif
    pour le joueur racine, muted si négatif) ;
  - le **meilleur coup enfant** (arc accent parent→enfant) ;
  - `exact` (valeur définitive) = score en gras ; valeur provisoire = score
    en italique/muted ;
  - badges de type : `pruned` = miniature **grisée** avec hachure,
    `tt-hit` = icône raccourci, `terminal` = bordure pleine ;
  - sur un nœud *pruned*, l'emplacement du score affiche une **étiquette de
    coupure** compacte (`> β` / `≥ β` / `< α` / `≤ α`) colorée avec
    `--drl-arrow`, qui rend la comparaison lisible d'un coup d'œil ;
  - l'arête menant du nœud coupé à son **fils déclencheur** (`triggerKey`)
    est **pointillée et rouge** (`--drl-arrow`) : on voit quelle branche a
    provoqué la coupure, pas seulement qu'elle a eu lieu ;
  - chaque nœud porte un **tooltip composé** (score, α/β, raison de coupure
    et coup déclencheur).
- Les nœuds *pruned* et *tt-hit* n'ont pas de descendants : placeholder
  grisé (point pédagogique : *ce que l'algorithme ne calcule pas*).
- Étiquette des arêtes : le coup (ex. nom de colonne), discret.

## Contrôles

### Profondeur d'affichage

Slider `max-depth` : nœuds au-delà de N plies **collapsés** en un nœud
compact (compteur « + descendants »). Distinct de la profondeur de
recherche (`depth` du moteur) : l'affichage peut être plus court que
l'exploration.

### Zoom & pan

- Slider `zoom` + boutons `fit` (ajuster à la vue) / `100%`.
- Pan par drag sur le fond SVG (d3-zoom), pinch trackpad supporté.
- Mémorisation de la position lors des step (pas de recentrage sauvage) ;
  bouton `fit` pour se recentrer.

### Navigation par clic

- **Clic sur un nœud** : ce nœud devient la **racine de l'affichage** —
  l'arbre est re-layouté depuis lui (le reste de l'arbre reste en mémoire).
  Fil d'Ariane (breadcrumbs) des ancêtres pour ne pas se perdre.
- **Bouton « racine = état courant »** : retour à la racine réelle de la
  recherche (l'état courant de la partie).
- Double-clic sur une zone vide : retour racine également.

### Détails alpha-bêta

Un panneau sous le fil d'Ariane décrit la fenêtre du **nœud de détail** :
racine d'affichage explicite si elle est posée, sinon le curseur du stepper,
sinon le dernier nœud visité, sinon la racine de recherche. Il affiche
l'étiquette du nœud, α, β, le score et son exactitude, puis, pour un nœud
coupé, une ligne rouge du type `meilleur +0,50 > β +0,20 · déclenché par le
coup 4`.

Le panneau est `aria-live="polite"` : le stepper le met à jour à chaque pas
sans interrompre la lecture en cours. Il fonctionne dans les deux modes
d'élagage — les détails sont affichés même quand `strictPruning` est faux —
car ils décrivent la fenêtre effective, pas une option d'affichage.

### Contrôles de type débogueur

Le clou de la démo : **stepper le développement de l'arbre**, nœud par
nœud, dans l'ordre réel de l'algorithme. La recherche rejoue depuis
l'état courant de la partie (le modèle d'arbre se construit au rythme
des pas, indépendamment de la recherche « réelle » qui peut être finie
depuis longtemps).

Quatre contrôles, sémantique de débogueur :

| Contrôle | Sémantique |
|---|---|
| **Step-in** | Visite le prochain fils non visité du nœud « courant » ; descend dedans (nœud courant = ce fils). 1 pas = 1 unité de travail de l'agent. |
| **Step-over** | Termine l'évaluation du nœud courant (tous ses fils restants, récursivement) et passe au nœur frère suivant ; le nœud courant devient le frère. |
| **Step-out** | Termine le nœud courant **et tous ses frères**, remonte : le nœud courant devient le parent. |
| **Run / pause** | Relance la consommation en continu (vitesse via `speed` du `DemoElement`). |

**Machine à états du stepper** : le stepper maintient une **pile de
contexte** (nœud courant + index de fils). Le moteur de recherche est
exécuté **paresseusement** : l'arbre n'est développé que jusqu'au nœud
courant, jamais au-delà. Toutes les implémentations d'agents doivent
pouvoir se consommer ainsi (c'est l'ordre DFS garanti par le contrat
`SearchAgent`).

Cette sémantique est **refinable** (précisée plus tard dans la démo) : ce
document fixe les quatre actions et la pile, les raccourcis clavier et la
présentation exacte pourront être ajustés.

**Reset du stepper** : tout nouvel état racine (nouveau coup joué, clic
sur nœud root, `drl-reset`) réinitialise la pile.

## Branchement (contrat d'utilisation)

```ts
const agent = new MinimaxAgent(game, {
  depth: 6,
  evaluator,
  recorder: treeComponent.recorder,  // ← branché ici
});
```

Le composant expose :

- `recorder: TreeRecorder` — implémentation qui alimente le modèle ;
- `attach(game)` — informe le rendu du `Game` (pour `boardView`) ;
- Événement `drl-root-changed` quand la racine d'affichage change.

Plusieurs sources possibles : la recherche d'un agent peut être
**enregistrée** (replay) ou **consommée à la demande** (stepper). C'est le
même flux d'événements ; le stepper contrôle la consommation, la vue « run »
la consomme en continu.

## Couplage avec le jeu interactif

La page de démo place jeu interactif et arbre côte à côte :

- racine de l'arbre = état courant de la partie ;
- à chaque coup joué (humain ou agent) : nouvelle recherche, nouvel arbre
  (le précédent est abandonné — le modèle est vidé, pas conservé en
  mémoire) ;
- l'option « rejouer le coup choisi par l'agent » : clic sur un nœud de
  l'arbre → jouer ce coup dans la partie (exploration pédagogique —
  « et si on jouait ce coup-là ? »).

## Paramètres exposés (attributs HTML)

| Attribut | Type | Défaut | Rôle |
|---|---|---|---|
| `max-depth` | number | `Infinity` | profondeur d'affichage (collapsage) |
| `node-size` | number | 3rem c4 / 4rem t3 | taille des miniatures (densité) |
| `stepper` | bool | `true` | active les contrôles débogueur |

## Tests

- `tree-model.test.ts` : insertion d'une séquence DFS connue → structure
  attendue (profondeurs, parents, ordre) ; fusion par `nodeKey` ;
  `ValueEvent` met à jour sans dupliquer ; nœuds pruned sans descendants.
- Stepper : machine à états testée en pur — step-in/over/out sur un arbre
  construit, vérification de la pile de contexte à chaque pas.
- Rendu : pas de test navigateur ; layout d3 testé via le modèle
  (géométrie exclue du scope).

## Non-buts

- Pas d'animation de transition entre états de l'arbre (v1 statique,
  transition d3 possible plus tard).
- Pas de vue DAG/transpositions complète (la fusion par `nodeKey` montre
  les raccourcis, pas le graphe complet).
- Pas d'édition des poids de l'heuristique depuis la vue.
- Contrôles de stepper : pas de raccourcis clavier en v1.