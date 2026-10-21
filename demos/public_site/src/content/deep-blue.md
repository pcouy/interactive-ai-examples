---
title: Deep Blue · recherche adversariale arborescente
description: |
    Jouez au morpion ou au puissance 4 contre un agent qui explore l'arbre des
    coups. Explorez le processus de décision de l'agent.
---

## Le jeu

La partie gauche de l'écran vous permet de jouer une partie de morpion ou
puissance 4 contre un agent utilisant une recherhe arborescente (algorithme
minimax), ou de faire jouer 2 agents l'un contre l'autre.

La partie droite de l'écran vous permet de développer et explorer l'arbre de
recherche qui a permit à l'agent de décider un coup, à partir de la position
actuelle du jeu.

Les contrôles dans la zone du haut permettent de paramétrer l'algorithme de
recherche.

Le paramètre "Prof. max" dans la partie droite contrôle uniquement l'affichage
de l'arbre développé. Cliquer sur une position du jeu dans l'arbre permet
d'afficher uniquement le sous-arbre à partir de cette position.

## Minimax

Minimax suppose que chaque joueur joue au mieux. Il développe les coups jusqu'à une profondeur limite, puis attribue une valeur aux feuilles. La valeur remonte vers la racine : l'agent maximise son score, l'adversaire le minimise. Une position terminale a une valeur connue : victoire, défaite ou partie nulle.

$$ V(s) = \begin{cases} E(s) & \text{si } s \text{ est terminal} \\ \max_{a \in A(s)} V(s_a) & \text{si c'est à l'agent de jouer} \\ \min_{a \in A(s)} V(s_a) & \text{sinon} \end{cases} $$

## Élagage alpha-bêta

L'arbre complet devient vite immense. Alpha-bêta coupe les branches qui ne peuvent pas changer la décision. Il transporte deux bornes : la meilleure valeur garantie pour le maximiseur et pour le minimiseur. La recherche se restreint à la fenêtre $[\alpha, \beta]$. Le résultat reste identique à minimax, mais l'arbre visité est plus petit.

## La fonction d'évaluation

Aux feuilles de l'arbre (lorsqu'on explore jusqu'à la profondeur maximale sans
rencontrer une fin de partie), l'agent ne connaît pas l'issue de la partie. Une fonction d'évaluation estime la position : alignements, menaces, cases libres. Son choix change le style de jeu.

L'évaluateur `rollout` joue quelques parties aléatoires à partir de chaque
feuille pour estimer la valeur.

## Le facteur d'actualisation

Le paramètre $\gamma$ pondère la valeur des coups enfants à la remontée. Un gain immédiat vaut alors plus qu'un gain lointain. Avec $\gamma = 1$, tous les gains gardent le même poids.

On remarque qu'avec $\gamma = 1$ l'agent "abandonne" dès que la recherche
arborescente prouve que vous pouvez forcer une victoire. Avec $\gamma < 1$, par
exemple $\gamma = 0.99$, l'agent préferera toujours un coup qui lui permet de
retarder sa défaite, ou de gagner plus rapidement.

### Ce que vous pouvez essayer

1. Changez la profondeur de recherche et regardez l'arbre grandir.
2. Comparez minimax et alpha-bêta : le coup reste le même, mais le nombre de nœuds visités baisse.
3. Activez la table de transposition et parcourez ses entrées pour voir les positions déjà évaluées.
