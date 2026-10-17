# Deep Blue — Environnements de jeu

Implémentation du contrat `Game` (`01-contrats.md`) pour TicTacToe et
Connect4, plus le rendu de plateau `<drl-game-board>` partagé entre le jeu
interactif et les nœuds de l'arbre.

Deux environnements sont fournis pour montrer la **généricité** de la
recherche : le même moteur minimax/alpha-beta doit fonctionner sur les deux,
sans adaptation. TicTacToe est petit (résoluble en entier), Connect4 est
grand (l'heuristique artisanale devient nécessaire) : c'est la progression
pédagogique de la section 1.A.i.

## Core — `src/core/games/`

```
src/core/games/
├── types.ts        // réexporte les contrats Game, BoardView
├── tic-tac-toe.ts  // TicTacToeState, moves, apply/undo, winner, key
├── tic-tac-toe.test.ts
├── connect4.ts     // Connect4State, moves (colonnes), gravity, winner
├── connect4.test.ts
└── board-view.ts   // BoardView pour chaque état (grille générique)
```

### TicTacToe

- **État** : `cells: (PlayerId|null)[9]` + `currentPlayer`. Trois états
  suffisent, deux joueurs (`PlayerId = 0 | 1`).
- **Coups** : `Move = { index: 0..8 }`.
- **Terminal** : alignement de 3 (8 lignes gagnantes), grille pleine
  → `'draw'`, sinon `null`.
- **`stateKey`** : canonisation. Rotation/réflexion du plateau produisent des
  **transpositions légitimes** : la clé canonique est la plus petite des
  8 symétries (bits du plateau XOR) — c'est le point pédagogique de la table
  de transposition (« 8 fois moins de calcul »).
- **Rendu** : `BoardView` 3×3, croix/rond.

### Connect4

- **État** : grille 6×7 (6 lignes × 7 colonnes), pions empilés par gravité,
  `currentPlayer`.
- **Coups** : `Move = { col: 0..6 }` — colonnes non pleines.
- **Terminal** : alignement de 4 (vertical, horizontal, diagonales ×2),
  grille pleine → `'draw'`, sinon `null`.
- **`stateKey`** : canonisation par symétrie miroir gauche/droite seulement
  (pas de rotation : la gravité casse la symétrie verticale). Pédagogique :
  « pourquoi la transposition est-elle moins efficace sur Connect4 ? »
- **Rendu** : `BoardView` 6×7, disques jaunes/rouges (tokens du thème),
  avec `highlight` = dernier coup joué.

### Règles de validation (tests)

Pour chaque jeu, une **génération d'états aléatoires** (playouts aléatoires)
sert de base aux tests de propriétés :

- `legalMoves` : chaque coup est rejouable (case vide / colonne non pleine),
  et réciproquement toute action jouable apparaît (exhaustivité).
- `apply`/`undo` : identité (testable par playout aléatoire inversé).
- `winner` : cohérent (état terminal ssi plus de coups légaux ou gagnant).
- `stateKey` : les transpositions par symétrie partagent une clé ; les
  états distincts n'en partagent aucune sur corpus aléatoire.

## Composants — `src/components/`

### `<drl-game-board>`

Rendu **paramétrable** d'un `BoardView`, à n'importe quelle échelle —
utilisé en pleine taille pour le jeu interactif et en miniature pour chaque
nœud de l'arbre (`04-arbre.md`).

- Entrée : `view: BoardView` (propriété reactive), `cellSize: number`
  (défaut thème).
- Rendu : un SVG par case + marqueurs ; **pas** de logique de jeu —
  le composant ne connaît ni TicTacToe ni Connect4, il n'affiche que
  `BoardView`.
- Événement : `drl-cell-click { detail: { row, col } }`.
- `clickable` : curseur pointer + surlignage au survol.
- `highlight` : cadre accent sur les cellules concernées.

Le mapping cellule→coup (`row, col` → `Move`) est fait par l'hôte :
pour Connect4, une colonne entière est cliquable (toutes les cellles de la
colonne) et `drl-cell-click` est réduit à `col` côté hôte.

### `<drl-tic-tac-toe>` / `<drl-connect4>`

Chaque jeu expose son propre composant de **partie interactive**. Ils
partagent la même mécanique de jeu (seuls le `Game` et le rendu diffèrent) :
une factory commune construit le composant à partir d'un `Game` concret
(pédagogie : la généricité du contrat, visible dans le code).

Chaque composant gère :

- `players: [PlayerConfig, PlayerConfig]` — chacun `'human'` ou `'agent'`
  (voir `01-contrats.md`).
- **Coup humain** : clic sur cellule → `game.apply` → re-render. Coup illégal
  ignoré (retour visuel bref).
- **Coup d'agent** : boucle de `step()` de l'agent connecté, `stepBudget`
  itérations (via le `speed` du `DemoElement`) jusqu'à `done`, puis
  `apply(bestMove)`.
- **Undo** : bouton dédié (annule le dernier coup ; en mode agent vs agent,
  annule jusqu'au dernier coup **humain** joué). `game.undo`.
- **Historique** : pile d'états (les états sont immuables — la pile est
  triviale).
- Fin de partie : bandeau résultat, `drl-reset` du `DemoElement` repart
  de `initialState()`.

### Interaction avec l'arbre

Chaque partie interactive peut se brancher au visualiseur (`04-arbre.md`) :
à chaque coup d'agent, le `TreeRecorder` de l'agent est redirigé vers
l'arbre, racine = état courant de la partie. Le même état (`S`) est partagé
immuablement entre jeu et arbre — pas de sérialisation.

## Paramètres exposés (attributs HTML)

| Attribut | Type | Défaut | Rôle |
|---|---|---|---|
| `players` (JSON) | `[PlayerConfig, PlayerConfig]` | `["human","agent"]` | qui joue chaque côté |
| `depth` | number | jeu-dépendant (TTT: 6, C4: 4) | horizon de recherche des agents |
| `evaluator` | string | `"windows"` | heuristique (voir `03-algorithmes.md`) |
| `transposition` | bool | `false` | active la table |
| `speed` | number | 1 | itérations d'agent par tick |

## Tests

- Core : propriétés ci-dessus (exhaustives, sur playouts aléatoires).
- Rendu : tests visuels hors périmètre (pas de tests navigateur) ; le
  mapping cellule→coup est testé au niveau composant (une seule fonction
  pure, testable).

## Non-buts

- Pas d'IA au-delà de la recherche (`SearchAgent`) — pas de MCTS dans cette
  démo (interface prête, implémentation postérieure).
- Pas de mode 3D / pas de mode en ligne.
- Pas de sauvegarde de partie (reset suffit).