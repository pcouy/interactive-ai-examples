---
title: Deep Blue · recherche adversariale
description: Jouez au morpion ou au puissance 4 contre un agent qui explore l'arbre des coups. Comparez minimax et alpha-bêta.
---

## Le jeu

Cette page rejoue une partie à deux joueurs : morpion ou puissance 4. Vous jouez contre un agent, ou vous regardez deux agents s'affronter. Chaque coup de l'agent provient d'une recherche dans l'arbre des positions.

## Minimax

Minimax suppose que chaque joueur joue au mieux. Il développe les coups jusqu'à une profondeur limite, puis attribue une valeur aux feuilles. La valeur remonte vers la racine : l'agent maximise son score, l'adversaire le minimise. Une position terminale a une valeur connue : victoire, défaite ou partie nulle.

$$ V(s) = \begin{cases} E(s) & \text{si } s \text{ est terminal} \\ \max_{a \in A(s)} V(s_a) & \text{si c'est à l'agent de jouer} \\ \min_{a \in A(s)} V(s_a) & \text{sinon} \end{cases} $$

## Élagage alpha-bêta

L'arbre complet devient vite immense. Alpha-bêta coupe les branches qui ne peuvent pas changer la décision. Il transporte deux bornes : la meilleure valeur garantie pour le maximiseur et pour le minimiseur. La recherche se restreint à la fenêtre $[\alpha, \beta]$. Le résultat reste identique à minimax, mais l'arbre visité est plus petit.

## La fonction d'évaluation

Aux feuilles, l'agent ne connaît pas l'issue de la partie. Une fonction d'évaluation estime la position : alignements, menaces, cases libres. Son choix change le style de jeu. L'évaluateur `rollout` joue quelques parties aléatoires pour estimer la valeur.

## Le facteur d'actualisation

Le paramètre $\gamma$ pondère la valeur des coups enfants à la remontée. Un gain immédiat vaut alors plus qu'un gain lointain. Avec $\gamma = 1$, tous les gains gardent le même poids.

### Ce que vous pouvez essayer

1. Changez la profondeur de recherche et regardez l'arbre grandir.
2. Comparez minimax et alpha-bêta : le coup reste le même, mais le nombre de nœuds visités baisse.
3. Activez la table de transposition et parcourez ses entrées pour voir les positions déjà évaluées.
