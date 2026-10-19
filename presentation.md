# Conférence interactive sur le Deep Reinforcement Learning / IA non générative

L'objet de cette présentation est de donner des clés de compréhension des
techniques d'IA participant à "l'arbre généalogique" du _Deep Reinforcement
Learning*, qui désigne une famille d'algorithmes et de méthodes, notamment mises
en oeuvre dans mon projet MaRLio. Cette discipline est née en 2014 d'un ensemble
d'innovations ayant permis de combiner l'apprentissage par renforcement avec le
deep learning, produisant un algorithme capable d'apprendre à jouer à 7 jeux
Atari différents à partir de captures d'écran brutes, sans adaptations
spécifiques à chaque jeu. Depuis la publication de cet article fondateur par
Deep Mind, cette discipline a connu un développement rapide. Elle joue aujourd'hui
un rôle essentiel dans les dernières étapes de l'entrainement des LLM actuels.
C'est d'un certain point de vue ce qui permet de transformer un "simple modèle
de langage" en une forme d'intelligence artificielle.

Pour comprendre ce qu'est le Deep Reinforcement Learning, il faudra d'abord
comprendre séparément l'apprentissage par renforcement et les réseaux de
neurones. Plusieurs démonstrations interactives permettront de voir les concepts
essentiels en évitant au maximum les mathématiques compliquées. Nous pourrons
alors proposer construire ensemble une approche "naïve" tentant d'associer ces
deux champs de l'IA. 

Les atouts respectifs de ces deux disciplines sont complémentaires. Leur
combinaison a débloqué plusieurs verrous et permis d'appliquer l'apprentissage
par renforcement à des problèmes ou jeux jusqu'ici trop complexes avec de
nombreux succès, dont certains très médiatiques. Une sélection de quelques unes
de ces réussites permettra de présenter les problèmes qui ont été rencontrés et
des innovations qui ont permis de les surmonter.

Si le temps le permet, quelques démonstrations supplémentaires viendront
conclure la présentation en montrant des systèmes d'apprentissage par
renforcement "rudimentaires" (mais dont les principes de base sont toujours à la
fondation des versions modernes) qui manifestent des problèmes d'alignement
analogues à ceux qui ont fait les titres ces derniers mois.

## Sommaire

1. Généalogie du *Deep Reinforcement Learning*
    1. Comprendre l'apprentissage par renforcement
        1. [Deep Blue](https://fr.wikipedia.org/wiki/Deep_Blue) : Démo interactive
             sur Tic-Tac-Toe et Puissance 4
        2. La fonction d'évaluation comme limite, l'apprentissage automatique
             tabulaire comme solution, généralisation à un agent dans un environnement
        3. Q-learning tabulaire : démo interactive avec des Gridworlds
        4. L'explosion combinatoire comme limite, l'observation engineering comme
             solution
    2. Comprendre les réseaux de neurones
        1. Le perceptron (démo interactive)
        2. La descente de gradient : démo avec l'analogie de la balle qui roule dans
             un paysage vallonné
        3. Les réseaux de neurones : Perceptron multi couche
           [Démo](https://playground.tensorflow.org)
        4. Classifieur de chiffre MNIST (démo interactive)
    3. Comment les deux fonctionnent ensemble
        - Limites de l'observation-engineering pour le RL : quel problème on cherchait à
            résoudre ? (la généralisation entre des observations similaires)
        - Quel autre outil ? (les réseaux de neurones)
2. Progrès rapides du *Deep Reinforcement Learning* (à réordonner par
     chronologie)
    1. DQN -> Rainbow : progrès rapides sur certains jeux, stagnation sur d'autres
    1. AlphaGo -> AlphaZero -> MuZero
    2. Curiosité intrinsèque : apprendre à explorer
    3. Politiques adverses (*fun fact* plus qu'exemple de progrès)
    5. DiscoRL : Apprendre à apprendre (meta-RL)
    6. AlphaStar : La ligue d'agents
    7. Retour d'expérience sur MaRLio : exploration vs exploitation
3. L'alignement comme enjeu du RL bien avant les modèles frontières
    1. [AI Safety Gridworlds](https://arxiv.org/abs/1711.09883) (démos
         interactives ?)
    2. Retour d'expérience sur MaRLio : de la difficulté de définir la bonne
         récompense
