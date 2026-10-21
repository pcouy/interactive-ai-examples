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

Une fin de partie vaut $+1$ en cas de victoire, $-1$ en cas de défaite, $0$ en
cas de partie nulle.

$$ V(s) = \begin{cases} E(s) & \text{si } s \text{ est terminal} \\ \max_{a \in A(s)} V(s_a) & \text{si c'est à l'agent de jouer} \\ \min_{a \in A(s)} V(s_a) & \text{sinon} \end{cases} $$

$s$ est un état, $V(s)$ est sa valeur. $E(s)$ est $+1$, $0$ ou $-1$ (état
final). $s_a$ est l'état du jeu après avoir joué le coup $a$ depuis la position
$s$.

## Élagage alpha-bêta

L'arbre complet devient vite immense. Alpha-bêta coupe les branches qui ne peuvent pas changer la décision. Il transporte deux bornes : la meilleure valeur garantie pour le maximiseur et pour le minimiseur. La recherche se restreint à la fenêtre $[\alpha, \beta]$. Le résultat reste identique à minimax, mais l'arbre visité est plus petit.

## La fonction d'évaluation

Pour un jeu comme le morpion, l'élagage alpha-bêta suffit pour pouvoir explorer
tout l'arbre en un temps raisonnable. Cependant même pour puissance 4 (bien plus
petit que les échecs), cela ne suffit plus. On limite donc la profondeur
maximale de recherche à quelques coups, pour contenir le temps de calcul et
l'espace mémoire nécessaire.

Aux feuilles de l'arbre (lorsqu'on explore jusqu'à la profondeur maximale sans
rencontrer une fin de partie), l'agent ne connaît pas l'issue de la partie. Une fonction d'évaluation estime la position : alignements, menaces, cases libres. Son choix change le style de jeu.

Le choix `windows` (par défaut) correspond à :
- Toujours 0 dans les états non finaux pour le morpion
- Une fonction d'évaluation "simple" pour puissance 4

L'évaluateur `rollout` joue quelques parties aléatoires à partir de chaque
feuille pour estimer la valeur.

Le choix `custom` vous permet d'essayer de développer votre propre fonction
d'évaluation.

La fonction d'évaluation est une des principales faiblesses des méthodes "à la
Deep Blue". Il faut une connaissance experte du jeu et de ses stratégies, et
peut-être encore plus compliqué : il faut pouvoir traduire ces stratégies et
intuitions en code. C'est un travail long et difficile, qui ne fonctionne que
pour un seul jeu. La recherche arborescente peut fonctionner avec de nombreux
jeux, mais la fonction d'évaluation doit être re-développée de zéro pour chaque
jeu.

## Le facteur d'actualisation

Le paramètre $\gamma$ pondère la valeur des coups enfants à la remontée. Un gain immédiat vaut alors plus qu'un gain lointain. Avec $\gamma = 1$, tous les gains gardent le même poids.

On remarque qu'avec $\gamma = 1$ l'agent "abandonne" dès que la recherche
arborescente prouve que vous pouvez forcer une victoire. Avec $\gamma < 1$, par
exemple $\gamma = 0.99$, l'agent préferera toujours un coup qui lui permet de
retarder sa défaite, ou de gagner plus rapidement.

## Table de transposition (TT)

L'arbre de recherche peut encore être réduit en remarquant que certaines
positions sont équivalentes, par symétrie ou rotation, et en réutilisant les
valeurs déjà calculées pour des positions équivalentes.

En plus de réduire la taille de la recherche, cette table permet également
d'obtenir une version rudimentaire de l'apprentissage par renforcement. En
utilisant des règles appropriées pour mettre à jour et utilser les valeurs stockées,
l'agent peut apprendre automatiquement des valeurs qui remplacent la fonction
d'évaluation.

### Ce que vous pouvez essayer

1. Changez la profondeur de recherche et regardez l'arbre grandir.
2. Comparez minimax et alpha-bêta : le coup reste le même, mais le nombre de nœuds visités baisse.
3. Activez la table de transposition et parcourez ses entrées pour voir les positions déjà évaluées.
