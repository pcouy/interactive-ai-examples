---
title: Frozen Lake · Q-learning tabulaire
description: Guidez un agent qui apprend seul à traverser un lac gelé. Réglez l'exploration, l'apprentissage, la glissade et le rejeu.
---

## Environnements *gridworld*

Les environnements *gridworld* sont souvent utilisés pour tester rapidement des
intuitions en apprentissage par renforcement ou pour illustrer des algorithmes.
De manière générale, on peut appliquer des algorithmes d'apprentissage par
renforcement dès qu'un problème peut être modélisé par un agent dans un
environnement :

- L'agent observe un état $s$ de l'environnement
- L'agent choisit une action $a$
- L'environnement produit "en réponse" à l'action un nouvel état $s'$ et une
  récompense $r$

Les *gridworld* sont des environnements dans lesquels chaque état est une case.
L'agent observe l'identifiant de la case, et peut choisir parmi 4 actions
correspondant aux 4 directions. Certaines cases peuvent, lorsque l'agent arrive
dessus, mettre fin à la tentative, donner une récompense, ou les deux à la fois.

## Le lac gelé

Il s'agit d'un environnement *gridworld* classique.

Un agent se déplace sur une grille gelée. Chaque case est solide (pas de
récompense) ou cache un trou (fin de tentative sans récompense). La case
d'arrivée contient une récompense de 1 unité. Si l'agent tombe, la partie s'arrête. Il doit trouver un chemin sûr sans connaître la carte.

## Apprendre par essais et erreurs

L'agent ne reçoit pas la règle du jeu. Il agit, observe la récompense et met à
jour la valeur $Q(s,a)$ du couple état-action. Une récompense positive renforce
l'action ; une chute la pénalise. Les flèches indiquent ensuite le meilleur
déplacement connu. Vous pouvez survoler des cases avec la souris pour voir la
valeur que l'agent accorde à chaque action depuis cette case.

## Exploration et exploitation

S'il choisit toujours la meilleure action connue, l'agent peut rester bloqué. Epsilon fixe la probabilité de choisir une action au hasard. Une valeur élevée favorise la découverte ; une valeur faible favorise l'exploitation.

## Glissade et actualisation

La probabilité de glissade est à zéro par défaut. En l'augmentant, l'agent a une
probailité que l'action qu'il a choisit le mêne à une autre case voisine que
celle "prévue". Cela permet de modéliser des environnements qu'on ne peut pas
observer parfaitement (contrairement aux jeux auquels Deep Blue peut jouer).


L'agent essaye d'apprendre la "valeur" $Q$ de chaque action, qui correspond à la
récompense future espérée suite à ce choix d'action. Lorsque l'agent fait
l'expérience d'une "transition" $(s, a) \rightarrow (s', r)$ il peut en déduire
la valeur $Q(s, a)$ de l'action $a$ depuis l'état $s$ :

$$ Q_\text{observé}(s,a) = r + \gamma \max_{a'} Q(s', a') $$

Le paramètre $\gamma$ est le même facteur d'actualisation que celui présenté sur
le [page consacrée à la recherche arborescente](/deep-blue). Sa valeur $\gamma
< 1$ traduit une préférence pour une récompense plus proche dans le futur.

À chaque pas, l'agent corrige sa valeur avec l'erreur entre l'estimation pour la
transition observée et l'estimation courante :

$$ \text{Erreur} = Q(s,a) - Q_\text{observé}(s, a)  $$

$$ Q(s,a) \leftarrow Q(s,a) + \alpha * \text{Erreur} $$

Le taux d'apprentissage $\alpha$ contrôle la taille de la correction. Un taux trop grand rend l'apprentissage instable ; un taux trop petit le rend lent.

## Le rejeu

La mémoire de rejeu garde les transitions passées. Il réapprend à partir d'un échantillon aléatoire, ce qui casse la corrélation entre les pas. La taille de l'échantillon règle le nombre de transitions utilisées.

Cela permet également de faciliter la propagation de la valeur aux états
antérieurs.

### Ce que vous pouvez essayer

1. Augmentez epsilon et observez l'agent explorer des chemins nouveaux.
2. Baissez le taux d'apprentissage : les valeurs se stabilisent plus lentement.
3. Ajoutez de la glissade, puis activez le rejeu pour stabiliser l'apprentissage.
