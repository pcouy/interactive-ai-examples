Deux démonstrations interactives pour voir, manipuler et comparer deux familles d'algorithmes qui comptent dans l'histoire de l'IA. Tout se passe dans votre navigateur. Aucun serveur ne reçoit vos clics, aucune donnée ne quitte votre machine.

### Deep Blue : chercher avant de jouer

La première démonstration rejoue une partie de morpion ou de puissance 4. L'agent explore l'arbre des coups possibles, puis remonte les valeurs jusqu'à la racine. Vous pouvez comparer minimax et l'élagage alpha-bêta, changer la profondeur de recherche ou la fonction d'évaluation. L'arbre affiché montre les nœuds visités, les coupures et la table de transposition. Le score d'une position dépend de la profondeur restante et du facteur d'actualisation $\gamma$ : un gain immédiat vaut plus qu'un gain lointain.

### Frozen Lake : apprendre par essais

La seconde démonstration place un agent sur un lac gelé. Il doit atteindre un bonbon sans tomber dans un trou. Cette fois, personne ne lui donne la carte : il apprend la valeur $Q(s,a)$ de chaque couple état-action en explorant, en se trompant et en recommençant. Les curseurs règlent l'exploration, le taux d'apprentissage et la mémoire de rejeu.

### Un même fil

La recherche et l'apprentissage répondent à la même question : « comment choisir le prochain coup ? » La recherche calcule une valeur à partir des règles du jeu. L'apprentissage estime cette valeur à partir de l'expérience. Les deux pages montrent cette différence avec les mêmes contrôles, sans rien envoyer sur le réseau.
