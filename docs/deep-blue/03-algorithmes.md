# Deep Blue — Algorithmes d'évaluation

Implémentation des `SearchAgent` et `Evaluator` du contrat (`01-contrats.md`).
Toutes les implémentations sont **step-chunkées par générateurs** : aucun
appel ne boucle sans `yield`. Toutes sont **instrumentées** : chaque unité de
travail est notifiée à un `TreeRecorder`.

Emplacement : `demos/src/core/search/`.

```
src/core/search/
├── minimax.ts           // moteur minimax + alpha-beta (un seul moteur)
├── minimax.test.ts
├── random-agent.ts     // agent 100% aléatoire (baseline jouable)
├── random-agent.test.ts
├── evaluators/
│   ├── heuristic.ts    // évaluation artisanale (comptage de fenêtres)
│   ├── heuristic.test.ts
│   └── random-rollout.ts // évaluation MC (playouts aléatoires) + tests
└── transposition.ts     // table de transposition + tests
```

## Moteur minimax / alpha-beta (un seul moteur)

Minimax et alpha-beta sont **le même moteur** : alpha-beta est minimax avec
une fenêtre `[alpha, beta]` initialisée à `[-∞, +∞]` pour la racine. La
pédagogie exige de montrer que le même code, fenêtre ouverte ou resserrée,
produit le même `bestMove` mais pas le même arbre. Le moteur est paramétré :

```ts
export interface MinimaxOptions {
  readonly depth: number;          // horizon en plies
  readonly evaluator: Evaluator<S>; // heuristique aux feuilles
  readonly prune: boolean;         // false = minimax pur
  readonly recorder: TreeRecorder<S>;
  readonly transposition?: TranspositionTable;
}
```

Ordre DFS **strict et déterministe** : les fils sont visités dans l'ordre de
`legalMoves`. C'est ce choix qui rend le mode débogueur du visualiseur
possible (`04-arbre.md`) et qui rend les coupures reproductibles.

### Sémantique de step

Une itération de `step()` = un nœud visité (`onVisit`) avec, dans l'ordre :

1. `expand` — premier fils non encore visité du nœud courant ;
2. `terminal` — état terminal (score exact immédiat) ;
3. `leaf` — état non terminal à l'horizon (score heuristique) ;
4. `pruned` — fils coupé par la fenêtre (pas de récursion).

`onValue` est émis à chaque amélioration du meilleur score du nœud courant
(`exact: true` seulement au retour de récursion complet du nœud).

### Alpha-beta : la coupure comme événement

Quand `score ≥ beta` (ou `≤ alpha` au niveau adversaire), les frères
restants sont enregistrés `kind: 'pruned'` **sans récursion**. Le
visualiseur affiche ces nœuds grisés (`04-arbre.md`) — c'est le point
pédagogique central : *montrer ce qu'alpha-beta ne calcule pas*.

### Profondeurs par défaut

| Jeu | Défaut | Raison |
|---|---|---|
| TicTacToe | 6 plies | Arbre complet dès le 2e coup (~mille nœuds) — raisonnable |
| Connect4 | 4 plies | ~50k nœuds sans coupure ; alpha-beta le réduit à ~1-3k — démontrable en direct |

Le budget est réglable (attribut `depth`), l'utilisateur peut pousser et
observer (via la viz) le coût combinatoire.

## `random-agent` — agent 100% aléatoire (baseline jouable)

Un agent qui joue un coup **légal uniforme** :

- pas de recherche, pas d'arbre ;
- exposé au même contrat `SearchAgent` que minimax : `step()` → `done`
  en une itération, `bestMove` = coup uniformément aléatoire ;
- sert de baseline jouable : « l'algorithme le plus bête possible dans le
  même contrat ».

C'est le contrepoint pédagogique : minimax contre random ~100% de victoire
(à explorer avec la démo).

## `evaluators/heuristic.ts` — évaluation artisanale de Connect4

**Comptage de fenêtres** (nécessaire au-delà de l'horizon court) :

- pour chaque fenêtre de 4 cases adjacentes du plateau (69 fenêtres),
  si la fenêtre ne contient aucun pion adverse :
  - 4 pions propres : `+∞` (gagné — ne devrait jamais être vu, cf. terminal),
  - 3 pions : `+50`,
  - 2 pions : `+10`,
  - 1 pion : `+1`,
  - 0 pion : `0`.
- score = somme pondérée, signé du point de vue de `rootPlayer` ;
- **bornage** : `clamp(-0.95, +0.95)` (cf. `01-contrats.md`).

Poids **codés en dur** (constantes nommées) — pas de sliders : c'est le
point narratif (l'humain a écrit ces poids à la main, ils sont visibles et
contestables). Le fichier est écrit pour être **lu pendant la démo** :
commentaire pédagogique explicatif en tête.

Pour TicTacToe : l'évaluation est triviale (0 aux feuilles non terminales
suffit — l'information est dans le terminal). Le même `heuristic.ts` est
utilisé mais retourne 0 : pédagogique (montrer qu'un jeu ne nécessite pas
d'heuristique complexe à petit horizon).

## `evaluators/random-rollout.ts` — Monte-Carlo comme évaluateur

Évaluateur stochastique aux feuilles :

- `evaluate(state)` : lance un **playout aléatoire uniforme** jusqu'au
  terminal (borné à un plafond de plies anti-explosion), retourne le
  résultat du point de vue de `rootPlayer` : `+1`/`-1`/`0` ;
- **budget d'échantillonnage** : l'évaluateur accumule les résultats de
  plusieurs playouts par feuille (paramètre `samples`) et retourne la
  moyenne — la précision monte avec le budget.

**Dualité MC** — l'utilisateur peut brancher :

1. **`random-agent` seul** (pas de minimax) : agent jouable faible ;
2. **minimax + `random-rollout`** en `Evaluator` : minimax échantillonne les
   feuilles (approximation de l'évaluation exacte).

Le contraste est le point pédagogique : `random-rollout` est une *vraie*
fonction d'évaluation (boîte noire, échantillonnée), pas une heuristique
lisible — et son coût unitaire est réglable (`samples`).

## `transposition.ts` — table de transposition

Couche de **mémoïsation** optionnelle, activable sur le moteur minimax/
alpha-beta (indépendamment de `prune`) :

- clé : `game.stateKey(state)` (canonique par symétries, `02-jeux.md`) ;
- valeur : `TranspositionEntry` :
  ```ts
  interface TranspositionEntry {
    score: number;
    depth: number;    // plies résolues pour cette valeur
    exact: boolean;   // false = borne (coupure alpha-beta), pas valeur exacte
    bestMove: Move | null;
  }
  ```
- hit valide ssi `entry.depth ≥ remaining depth` et (`entry.exact` ou la
  fenêtre est compatible) ;
- hit → `onVisit({ kind: 'tt-hit' })` : le nœud est résolu sans visiter ses
  fils. Le visualiseur l'affiche comme un raccourci (`04-arbre.md`).

**Pédagogie visée** : « deux positions différentes mènent au même sous-arbre ;
on ne le calcule qu'une fois » — visible dans la viz via le compteur
`hits/misses` exposé dans `SearchStats`.

## `SearchStats`

Chaque agent expose ses statistiques de recherche (consommées par le
visualiseur) :

```ts
export interface SearchStats {
  readonly nodesVisited: number;
  readonly nodesPruned: number;
  readonly ttHits: number;
  readonly ttMisses: number;
  readonly leavesEvaluated: number;
  readonly elapsedSteps: number;
}
```

## Tests

- `minimax.test.ts` : `bestMove` du moteur = `bestMove` d'une implémentation
  de référence exhaustive (naïve, non génératrice) sur corpus d'états
  aléatoires TicTacToe — propriété : **prune on/off même coup**, arbre
  plus petit.
- Alpha-beta : **propriété d'exactitude** — sur chaque position testée,
  le score final (pas seulement `bestMove`) est identique avec/sans
  coupure, sur ~100 positions aléatoires TicTacToe.
- `transposition.test.ts` : avec TT, `nodesVisited` strictement inférieur,
  score final identique ; `ttHits > 0` sur positions avec symétries.
- `random-rollout.test.ts` : `evaluate` retourne toujours `[-1, 1]`,
  0 espérance sur position symétrique initiale (approximation, seuil
  probabiliste).
- `random-agent.test.ts` : coup toujours légal, distribution uniforme
  (test du chi² sur large échantillon).