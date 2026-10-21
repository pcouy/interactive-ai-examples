---
title: Frozen Lake · Q-learning tabulaire
description: Guidez un agent qui apprend seul à traverser un lac gelé. Réglez l'exploration, l'apprentissage, la glissade et le rejeu.
---

## Le lac gelé

Un agent se déplace sur une grille gelée. Chaque case est solide ou cache un trou. La case d'arrivée contient un bonbon. Si l'agent tombe, la partie s'arrête. Il doit trouver un chemin sûr sans connaître la carte.

## Apprendre par essais et erreurs

L'agent ne reçoit pas la règle du jeu. Il agit, observe la récompense et met à jour la valeur $Q(s,a)$ du couple état-action. Une récompense positive renforce l'action ; une chute la pénalise. Les flèches indiquent ensuite le meilleur déplacement connu.

## Exploration et exploitation

S'il choisit toujours la meilleure action connue, l'agent peut rester bloqué. Epsilon fixe la probabilité de choisir une action au hasard. Une valeur élevée favorise la découverte ; une valeur faible favorise l'exploitation.

## Glissade et actualisation

Le lac est glissant : la probabilité de glissade dévie parfois l'agent vers une case voisine. Le facteur d'actualisation $\gamma$ règle le poids du futur. Plus il est proche de 1, plus l'agent valorise les récompenses lointaines.

## La règle de mise à jour

À chaque pas, l'agent corrige sa valeur avec l'erreur entre la récompense observée et l'estimation courante :

$$ Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right] $$

Le taux d'apprentissage $\alpha$ contrôle la taille de la correction. Un taux trop grand rend l'apprentissage instable ; un taux trop petit le rend lent.

## Le rejeu

La mémoire de rejeu garde les transitions passées. Il réapprend à partir d'un échantillon aléatoire, ce qui casse la corrélation entre les pas. La taille de l'échantillon règle le nombre de transitions utilisées.

### Ce que vous pouvez essayer

1. Augmentez epsilon et observez l'agent explorer des chemins nouveaux.
2. Baissez le taux d'apprentissage : les valeurs se stabilisent plus lentement.
3. Ajoutez de la glissade, puis activez le rejeu pour stabiliser l'apprentissage.
