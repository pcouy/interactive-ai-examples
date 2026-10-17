# Deep Blue — Contrats d'interface

Ce document fixe les interfaces entre les couches de la démo. C'est le document
de référence : `02-jeux.md`, `03-algorithmes.md` et `04-arbre.md` dérivent tous
de ces contrats. Les contrats sont conçus comme une **bibliothèque réutilisable**
(Q-learning, MCTS, AlphaStar-likes…) : aucun type ci-dessous ne mentionne
minimax, TicTacToe ou SVG.

Tous les types vivent dans le core (`demos/src/core/`), en TypeScript strict,
sans aucun import DOM/Lit/D3/Three. Les composantsLit ne consomment ces types
que par leurs bordures.

## Principes transverses

1. **Immutabilité des états.** `apply` retourne un nouvel état ; jamais de
   mutation en place. Permet l'annulation (`undo`), le partage d'états entre
   jeu et arbre, et des clés de hash stables.
2. **Step-chunk par générateurs.** Toute recherche est un générateur qui
   produit une unité de travail par itération (`yield`). Le composant
   consomme N itérations par `step()` selon `speed`. Aucun code ne boucle
   sans `yield`.
3. **Perspective fixe.** Tous les scores sont exprimés **du point de vue du
   joueur qui fait la recherche** (`rootPlayer`). Pas de négation implicite
   au rendu : un score positif signifie « bon pour rootPlayer ».
4. **Scores bornés.** Les scores de fin de partie sont des sentinelles
   conventionnelles : `WIN = +1`, `LOSS = -1`, `DRAW = 0` (au signe près pour
   rootPlayer), et les scores heuristiques sont bornés à
   `] -WIN_SCORE_MARGIN, +WIN_SCORE_MARGIN [` avec
   `WIN_SCORE_MARGIN = 0.95` pour ne jamais masquer un résultat exact.
   Convention : tout état terminal a un score hors de la plage heuristique.

## `Game` — environnement

```ts
export interface Game<S, M> {
  /** Sentinelles de score (voir « Scores bornés »). */
  readonly winScore: number;   // score d'une victoire (ex: 1)
  /** État initial standard. */
  initialState(): S;
  /** Coups légaux dans un état. Ordre stable (préférence de rendu). */
  legalMoves(state: S): readonly M[];
  /** État successeur. Ne modifie pas `state`. */
  apply(state: S, move: M): S;
  /** Inverse d'apply. Retourne l'état parent. */
  undo(state: S, move: M): S;
  /** Joueur au trait dans cet état. */
  currentPlayer(state: S): PlayerId;
  /** Résultat si terminal, sinon null. */
  winner(state: S): PlayerId | 'draw' | null;
  /** Clé canonique : deux états équivalents partagent la même clé (TT). */
  stateKey(state: S): string;
  /** Rendu paramétrable : données du plateau (grille + pions). */
  boardView(state: S): BoardView;
}

export type PlayerId = number;
```

`S` (état) et `M` (coup) sont paramétrés par jeu : les algorithmes sont
génériques sur `(S, M)`.

### `BoardView` — contrat de rendu

Le rendu (`02-jeux.md`) est paramétrable par jeu et **identique** pour le jeu
interactif et les nœuds de l'arbre (miniature fidèle, échelle près).

```ts
export interface BoardView {
  /** Grille : lignes × colonnes. */
  readonly rows: number;
  readonly cols: number;
  /** Contenu case par case : null = vide. */
  readonly cells: ReadonlyArray<readonly (PlayerId | null)[]>;
  /** Coordonnées de cellules mise en évidence (ex: dernier coup). */
  readonly highlight?: ReadonlyArray<{ row: number; col: number }>;
  /** Cellules cliquables en mode humain (index de coup en retour). */
  readonly clickable?: ReadonlyArray<{ row: number; col: number }>;
}
```

Un seul composant `<drl-game-board>` consomme `BoardView`, à n'importe quelle
échelle (jouer en grand / miniature de nœud). Le clic remonte l'indice de la
cellule : la correspondance cellule→coup est responsabilité du composant hôte
(pour Connect4 : colonne).

## `Move` — représentation

Aucun contrat au-delà de `M` générique ; chaque jeu documente sa forme
(index de colonne pour Connect4, index de case pour TicTacToe).

## `Evaluator` — fonction d'évaluation

```ts
export interface Evaluator<S> {
  /**
   * Valeur heuristique d'un état NON terminal, du point de vue de rootPlayer.
   * Peut être asynchrone/échantillonnée (Monte-Carlo) : renvoie un estimateur.
   */
  evaluate(state: S, ctx: EvalContext): number;
  /** Équivalent step-chunké : consomme un budget de travail. */
  readonly id: string; // nom affiché (viz)
}

export interface EvalContext {
  readonly rootPlayer: PlayerId;
  readonly game: Game<S, M>;
}
```

Un évaluateur **n'est jamais appelé sur un état terminal** — le moteur de
recherche traite les terminaux lui-même (scores exacts). C'est la propriété
qui permet le bornage ci-dessus.

Un `Evaluator` peut être **complètement aléatoire / stochastique**
(Monte-Carlo) : dans ce cas, appeler `evaluate` avec un budget croissant
améliore l'estimation (voir `03-algorithmes.md`).

## `SearchAgent` — moteur de recherche step-chunké

```ts
export interface SearchAgent<S, M> {
  /** Lance la recherche depuis `state`. Reset implicite. */
  start(state: S): void;
  /**
   * Une unité de travail (un nœud visité). Renvoie l'état de la recherche :
   *  - 'running' : il reste du travail,
   *  - 'done' : le meilleur coup est disponible via bestMove().
   */
  step(): SearchStatus;
  /** Meilleur coup après 'done'. */
  bestMove(): M | null;
  /** Statistiques : nœuds explorés, coupures, hits/misss TT… */
  stats(): SearchStats;
}

export type SearchStatus = 'running' | 'done';
```

Invariants :

- `start` peut être rappelé à tout moment (nouveau coup joué) : la recherche
  précédente est abandonnée proprement, ses enregistrements d'arbre restent
  dans le recorder jusqu'à reset explicite.
- `bestMove` est `null` tant que la recherche n'est pas `done` (et l'affiche
  le choix *en cours* : voir `TreeRecorder`).

## `TreeRecorder` — instrumentation de la recherche

Chaque agent est **instrumenté** : à chaque unité de travail, il notifie ses
événements à un `TreeRecorder`. C'est la seule collaboration entre la couche
algorithme et la couche visualisation.

```ts
export interface TreeRecorder<S> {
  /** Notifie que `state` (fils de `parentKey`) vient d'être visité. */
  onVisit(event: VisitEvent<S>): void;
  /** Mise à jour de la meilleure valeur courante d'un nœud en cours d'éval. */
  onValue(event: ValueEvent): void;
}

export interface VisitEvent<S> {
  readonly nodeKey: string;      // = game.stateKey(state)
  readonly parentKey: string | null; // null = racine
  readonly move: unknown;        // coup menant au nœud
  readonly state: S;
  readonly depth: number;       // plies depuis la racine de recherche
  readonly kind: 'expand' | 'pruned' | 'tt-hit' | 'leaf' | 'terminal';
}

export interface ValueEvent {
  readonly nodeKey: string;
  readonly score: number;       // meilleur score connu pour ce nœud (min/max)
  readonly depth: number;      // profondeur résolue
  readonly exact: boolean;     // true = valeur définitive du nœud
}
```

Règles :

- L'ordre des `onVisit` est l'ordre DFS de la recherche : c'est lui que
  reproduit le mode « step » du visualiseur (voir `04-arbre.md`).
- Un nœud *pruned* est enregistré comme `kind: 'pruned'` **sans** explorer
  ses descendants : le visualiseur affiche un placeholder grisé.
- Un nœud `tt-hit` peut être résolu sans `onVisit` de ses fils.
- Le recorder n'est pas dans le chemin critique : un recorder « no-op »
  (viz désactivée) ne coûte quasiment rien (compteur off).

## `Player` — qui joue

Le jeu interactif est paramétré par un tableau de `Player` :

```ts
export type PlayerKind = 'human' | 'agent';

export interface PlayerConfig<S, M> {
  readonly kind: PlayerKind;
  /** Ignoré pour 'human'. Pour 'agent' : agent démarré à chaque nouveau coup. */
  readonly agent?: SearchAgent<S, M>;
  /** Budget de travail par coup (itérations de step()), pour 'agent'. */
  readonly stepBudget?: number;
}
```

## Diagramme de dépendances

```
Game ──────┐
           │       ┌──────────► TreeRecorder ──► <drl-eval-tree>
           ▼       │
      SearchAgent ─┤  (minimax / alpha-beta / random)
           ▲       │
           │       │
        Evaluator ┘  (heuristic / random-rollout / tt-memo)
```

## Tests

Toute implémentation de ces contrats doit passer une suite de propriétés
génériques (testées sur chaque environnement) :

- `apply∘undo = identité` sur tout état et coup légal.
- `legalMoves` ne contient que des coups menant à des états non terminaux
  selon `winner` cohérent.
- `stateKey` : deux états équivalents (transpositions) partagent une clé ;
  deux états différents n'en partagent jamais (au moins par construction).
- Pour tout agent : `bestMove ∈ legalMoves(rootState)` une fois `done`.