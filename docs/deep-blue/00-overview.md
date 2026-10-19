# Deep Blue — Spécification de la démo minimax / alpha-beta

## Rôle dans la présentation

Cette démo illustre la section **1.A.i** de `presentation.md` (*Deep Blue*) :
avant l'apprentissage automatique, les meilleurs programmes de jeux étaient des
**moteurs de recherche dans l'arbre des coups possibles**, guidés par une
**fonction d'évaluation écrite à la main**. C'est cette fonction d'évaluation —
et l'ingénierie humaine qu'elle exige — qui constitue la *limite* narrée par la
sous-section suivante : c'est elle que l'apprentissage tabulaire puis les
réseaux de neurones viendront remplacer.

La démo doit donc rendre visibles, en interaction directe :

1. **La recherche adversariale** — l'arbre des coups, exploré en profondeur.
2. **L'évaluation** — une valeur numérique attachée à chaque état exploré.
3. **Les optimisations** — alpha-beta réduit l'exploration, les tables de
   transposition réutilisent les calculs : deux manières d'aller plus loin dans
   l'arbre avec le même budget de calcul.
4. **La limite** — au bout de l'horizon, une heuristique artisanale prend le
   relais ; on peut la voir, la juger, la contester.

Chaque composant doit rester **réutilisable dans d'autres démos** (Q-learning,
MCTS, etc.) : les contrats du document `01-contrats.md` sont conçus comme une
bibliothèque, pas comme des détails internes de la démo.

## Vue d'ensemble

```
┌────────────────────────────────────────────────────────────────┐
│ <drl-deep-blue-page>                                           │
│                                                                │
│  ┌──────────────────┐    ┌─────────────────────────────────┐  │
│  │ Jeu interactif   │    │ Visualiseur d'arbre d'évaluation │  │
│  │ <drl-game-board> │◄───┤ <drl-eval-tree>                 │  │
│  └──────────────────┘    └─────────────────────────────────┘  │
│            ▲                      ▲                             │
│            │                      │                           │
│  ┌─────────┴──────────────────────┴─────────────────────────┐ │
│  │ Environnement de jeu (Game)   Agent (minimax / αβ / MC)   │ │
│  │            └── TreeRecorder ───┘   Evaluator ──┘         │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

- La couche **jeu** (`02-jeux.md`) expose des environnements réutilisables
  (TicTacToe, Connect4) et un rendu de plateau paramétrable.
- La couche **algorithmes** (`03-algorithmes.md`) expose des agents
  step-chunkés et des évaluateurs interchangeables.
- La couche **visualisation** (`04-arbre.md`) consomme les enregistrements
  d'exploration produits par n'importe quel agent instrumenté.
- `01-contrats.md` fixe les interfaces entre ces couches.

## Documents

| Document | Contenu |
|---|---|
| `00-overview.md` | Ce document : intentions, vue d'ensemble, glossaire |
| `01-contrats.md` | Interfaces `Game`, `Move`, `Evaluator`, `Agent`, `TreeRecorder` |
| `02-jeux.md` | Environnements TicTacToe et Connect4, rendu de plateau, modes de jeu |
| `03-algorithmes.md` | Minimax, alpha-beta, Monte-Carlo aléatoire, tables de transposition |
| `04-arbre.md` | Visualiseur d'arbre : contrôles débogueur, layout, zoom, navigation |

## Contraintes techniques

Rappel du cadre `demos/README.md` :

- 3 couches par démo : **core pur** (testable Vitest), **composant Lit**,
  **page playground**. Toute la logique de jeu et de recherche vit dans
  `src/core/` — aucun import DOM, Lit, D3 ou Three.
- Le core est **step-chunké dès la conception** : aucun appel ne doit bloquer
  le thread. Toutes les boucles de recherche sont des générateurs itérables
  consommés par le `step()` du composant.
- Composants en préfixe `drl-`, kebab-case, un dossier par composant.
- Code en anglais, logique métier dans `src/core/`, paramètres passés en
  argument de `tick`/`step` et jamais lus depuis le composant.
- Tests Vitest obligatoires pour tout le core : règles du jeu, propriétés des
  algorithmes (voir chaque document).

## Glossaire

- **État / position** — situation de plateau complète, immuable. Un état
  détermine le joueur courant et l'ensemble des coups légaux.
- **Coup / `Move`** — action légale dans un état. Deux états distincts peuvent
  produire le même coup (c'est ce qui rend le *hashing* des tables de
  transposition non trivial).
- **Arbre de recherche** — arbre des coups alternés des deux joueurs, enraciné
  à l'état courant de la partie.
- **Feuille** — état auquel on cesse d'explorer : soit la partie est terminée,
  soit on atteint la profondeur d'horizon.
- **Fonction d'évaluation / `Evaluator`** — heuristique qui attribue une valeur
  numérique à un état (du point de vue d'un joueur fixé).
- **Horizon** — profondeur maximale d'exploration configurée.
- **Minimax** — recherche exhaustive en profondeur : chaque joueur maximise
  (ou minimise) l'évaluation à son niveau.
- **Alpha-beta** — minimax + élagage : on démontre que certaines sous-branches
  ne peuvent pas influencer la décision, et on les coupe.
- **Playout / rollout** — partie simulée jusqu'au bout (ici : aléatoire).
- **Table de transposition** — mémoire associative état → (score, profondeur)
  qui évite de recalculer des positions déjà résolues ; la profondeur mémorisée
  est une profondeur **certifiée** (nombre de plies pour lesquels le score est
  exact), ce qui permet de réutiliser une position au-delà de l'horizon
  courant.
- **Nœud coupé / *pruned*** — sous-arbre non exploré parce qu'alpha-beta a
  prouvé son inutilité. Rendu grisé dans la visualisation, jamais omis.
- **Ply / profondeur** — un *ply* = un demi-coup. La profondeur se compte en
  plies depuis la racine de la recherche.