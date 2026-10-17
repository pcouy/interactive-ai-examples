# Histoire de l'IA avant les LLM
## Conférence interactive — 45 min, public général, en français

## 1. Vision

- **Support** : site web statique, sans étape de build, fonctionnant hors-ligne
  (présentation depuis un fichier local). Sommaire + navigation précédent/suivant,
  une page par étape, plusieurs démos possibles par page.
- **Créneau** : 75 min — 45 min de conférence, le reste pour l'interaction et les questions.
- **Public** : grand public. Histoire d'abord, aucune équation à l'écran ;
  la profondeur technique vit dans les démos et les annexes, pas dans le discours.
- **Deux fils directeurs** tissés à travers toute la narration :
  1. **Savoir codé à la main contre apprentissage à partir des données** —
     symbolique, recherche arborescente, systèmes experts d'un côté ;
     perceptron, rétropropagation, RL de l'autre. Les deux branches convergent dans AlphaGo.
  2. **Hype contre limites réelles** — chaque vague sur-promet ;
     la vraie limite était le calcul et les données, pas les idées.
     Se clôt avec la « leçon amère » de Sutton (2019).

> Remarque d'équilibre : la conférence s'appuie beaucoup sur les jeux — c'est
> un choix défendable, les jeux étant le banc d'essai de l'époque. L'étape 6
> (« IA du quotidien ») répond explicitement à cette objection.

## 2. La frise en 8 étapes (44 min)

### 1. « Turing et le rêve fondateur » (1943–1956) — ~6 min
- Jeu d'imitation et **machine-enfant** : Turing a posé les DEUX côtés du débat
  (programmer l'intelligence vs la faire grandir par apprentissage).
- Objections auxquelles il répond lui-même : théologique, gödelienne,
  Lady Lovelace (« les machines ne font que ce qu'on leur dit »),
  conscience (Jefferson). Échos ultérieurs : Chambre chinoise (Searle, 1980),
  effet ELIZA.
- Contexte : neurones formels de McCulloch & Pitts (1943), règle de Hebb (1949),
  *Intelligent Machinery* (1948) et ses machines non organisées — des réseaux
  de neurones proto. Dartmouth (1956) intégré ici comme acte fondateur.
- **Démo : ELIZA** — le public expérimente le test se faisant manipuler.

### 2. « Le perceptron : la première machine qui apprend » (1958–1969) — ~6 min
- Rosenblatt (1958) et sa **règle du perceptron** : nudge des poids uniquement
  en cas d'erreur — ce n'est **pas** la descente de gradient.
- **ADALINE** (Widrow & Hoff, 1960) : la règle delta, vraie descente de gradient
  sur l'erreur quadratique — le pont honnête vers la rétropropagation (1986).
- Sur écran : la frontière qui **ne peut pas séparer XOR** → Minsky & Papert
  (*Perceptrons*, 1969) → **hiver n°1** avec ses causes : rapport Lighthill (1973),
  coupes DARPA.
- **Démo : entraîneur de frontière de décision** — bascule règle perceptron /
  descente de gradient + preset XOR.

### 3. « Deep Blue : la force brute » (1950–1997) — ~5 min
- Shannon (1950) formalise le jeu d'échecs par recherche.
- **Samuel (1959)** : son programme de dames apprend son évaluation par
  auto-apprentissage — la graine du RL, plantée 40 ans plus tôt.
- Minimax, **élagage alpha-bêta**, fonction d'évaluation artisanée
  (avec le GM Joel Benjamin), matériel dédié (~200 M positions/s).
- Les controverses Kasparov (1996–97) : le « coup 37 » du jeu 2,
  accusation d'intervention humaine, refus de revanche d'IBM.
- **Démo : explorateur alpha-bêta** — nœuds élagués en surbrillance.

### 4. « Les systèmes experts : le savoir en boîtes » (1965–1993) — ~5 min
- DENDRAL (1965), MYCIN (années 70), XCON (économie réelle pour DEC),
  projet japonais de Cinquième Génération (1982).
- Modes d'échec : **goulot d'étranglement de l'acquisition de connaissances**,
  fragilité hors du domaine prévu.
- **Hiver n°2** avec ses causes propres — effondrement du marché des machines
  Lisp, faillite de la bulle des systèmes experts. Ne jamais fusionner les deux hivers.
- **Démo : moteur de règles** — chaînage avant + trace « pourquoi ? ».

### 5. « L'apprentissage par renforcement » (1988–1989) — ~6 min
- Bellman et la programmation dynamique → **TD** (Sutton, 1988) →
  **Q-learning** (Watkins, 1989). Récit pur, sans équations.
- **Démo : gridworld Q-learning** — LA démo centrale : curseurs ε/α/γ,
  carte de chaleur des valeurs, flèches de politique, murs éditables.

### 6. « Les années discrètes » (1980–2012) — ~4 min — sans démo (respiration)
- L'IA se rebaptise « machine learning » : SVM, forêts aléatoires, boosting,
  HMM, réseaux bayésiens (Pearl — prix Turing 2011).
- **L'IA du quotidien que tout le monde a vécue** : filtres anti-spam,
  PageRank, détection de visages — réponse à l'objection « trop de jeux ».
- L'underground neuronal survit : Néocognitron (Fukushima, 1980) →
  CNN de LeCun (1989) → LSTM (1997) → réseaux de croyance profonds (2006).
- Les ingrédients de 2012 : **GPUs + ImageNet (2009) → AlexNet**, le pivot.
  Puis parole (2010–12), word2vec, GANs, ResNet, teaser « attention »
  sans entrer dans les LLM.

### 7. « TD-Gammon → DQN : la récompense tardive » (1992–2013) — ~6 min
- 1992 : TD-Gammon (Tesauro) apprend le backgammon en jouant contre lui-même —
  niveau expert. **Un succès prématuré.**
- Pourquoi 20 ans pour généraliser : la **triade mortelle** (bootstrap +
  hors-politique + approximation de fonction) — **montrée, pas nommée** :
  on regarde l'apprentissage exploser à l'écran.
- 2013 : DQN (Mnih et al.) et ses deux astuces — **mémoire de rejeu** et
  **réseau cible** — rendent enfin l'idée stable.
- **Démo : divergence vs stabilisation** — la plus démonstrative de la conférence.

### 8. « AlphaGo et après » (2016–2024) — ~6 min
- AlphaGo = **MCTS (2006) + réseaux profonds + RL** : la branche force brute
  (étape 3) et la branche apprentissage (étapes 5–7) se rejoignent.
  Le coup 37 contre Lee Sedol. Puis AlphaGo Zero (sans données humaines),
  AlphaZero, MuZero (qui apprend les règles).
- **AlphaFold** : flèche **thématique** — le même labo pointe l'apprentissage
  vers la science. Supervisé + attention, **pas** de RL ni recherche arborescente.
- Clôture : **double Nobel 2024** (Physique : Hopfield & Hinton ;
  Chimie : Hassabis & Jumper) + la « leçon amère » de Sutton (2019).
- **Démo : explorateur MCTS** — arbre de recherche en direct, jouer contre lui.

## 3. Corrections historiques intégrées (à ne pas perdre en route)

1. **Perceptron ≠ descente de gradient.** La règle de Rosenblatt ajuste seulement
   sur erreur. Le pont honnête est ADALINE (1960), puis la rétropropagation (1986).
2. **DQN n'était pas la première combinaison réseau + RL.** TD-Gammon (1992)
   précède de 20 ans ; l'écart s'explique par l'instabilité de l'approximation
   de fonction, corrigée par le rejeu et le réseau cible.
3. **AlphaGo → AlphaFold est une flèche thématique**, pas une filiation technique.
4. **Deux hivers, deux causes distinctes** (promesses symboliques années 70 ;
   effondrement systèmes experts/Lisp fin des années 80). Ne jamais les fusionner.
5. **Ordre chronologique respecté** : la révolution deep learning (2012) précède
   DQN (2013) — DQN n'est possible qu'après AlexNet.
6. La conférence s'appuie sur les jeux par choix assumé, corrigé par l'étape 6.

## 4. Démos — spécifications et vérification

| # | Démo | Cœur | Vérification | Taille |
|---|------|------|--------------|--------|
| 1 | ELIZA | Script docteur français, interface de chat | Conversations prédéfinies → réponses sensées | S |
| 2 | Perceptron | Points au clic, presets séparable/XOR, bascule règle perceptron / delta, frontière + courbe d'erreur | Converge sur données séparables ; les deux règles échouent sur XOR | M |
| 3 | Alpha-bêta | Arbre abstrait fixe (D3), élagage animé, curseur de profondeur, compteur de nœuds | Même valeur à la racine que minimax complet | M |
| 4 | Moteur de règles | ~20 règles « identifier l'animal », chaînage avant, trace « pourquoi ? » | Faits donnés → animaux corrects | S |
| 5 | Gridworld Q | 10×10 éditable, curseurs ε/α/γ, carte de chaleur, flèches de politique, courbe de récompense | Q tabulaire résout un gridworld connu en un nombre borné d'épisodes | M+ |
| 6 | Divergence | Même gridworld, connaissances compressées → explosion ; rejeu + réseau cible → stabilisation. **Plan B garanti : étoile de Baird** | Divergence visuelle sans correctifs, stabilité avec | M+ |
| 7 | MCTS | Morpion humain vs MCTS, arbre en direct (visites, taux de victoire, UCB1 en surbrillance), budget de simulations réglable | MCTS à fort budget ne perd jamais ; coup optimal en positions connues | L |
| 8 | Frise maîtresse | D3, 1943–2024, événements colorés par fil (symbolique / apprentissage / hivers), infobulles, liens vers les pages | Tous les événements cliquables renvoient au bon endroit | M |

Risque principal : la démo 6 exige un réglage fin pour être spectaculaire ;
l'étoile de Baird (divergence canonique, déterministe) sert de repli garanti.
Décision au moment du build.

## 5. Annexes (zéro minute de conférence)

- Frise chronologique complète et interactive (aussi colonne vertébrale du sommaire)
- Objections et Chambre chinoise
- Ère symbolique : Logic Theorist, GPS, SHRDLU, Shakey/A*
- Hype et hivers : les deux cycles, courbes de financement
- Triade mortelle : note technique pour les curieux
- Watson (2011) : le pont dans la mémoire collective entre Deep Blue et AlphaGo
- La « leçon amère » (Sutton, 2019) : traduction et commentaire
- L'IA du quotidien des années discrètes (spam, PageRank, visages)

## 6. Architecture technique

```
histoire_IA/
├── index.html                  # Accueil : hero + frise maîtresse (D3)
├── pages/01-turing.html … 08-alphago.html
├── annexes/                    # pages annexes ci-dessus
├── assets/css/style.css
├── assets/js/demos/            # eliza, perceptron, minimax, rules, gridworld, divergence, mcts
├── assets/js/timeline.js       # frise maîtresse (D3)
├── assets/js/data/            # timeline.js, rules.js — <script>, jamais fetch()
└── assets/vendor/d3.v7.min.js # vendu localement, CDN en repli
```

- **Aucune étape de build** ; chrome de page dupliqué en HTML statique (15 pages,
  chrome minime). Fonctionne depuis `file://` — données en `<script>`, pas de fetch.
- **Conventions** : prose et interface en français ; identifiants et commentaires
  de code en anglais ; typographie française (« ») ; polices système (hors-ligne) ;
  d'abord projecteur 16:9, responsive ensuite.
- **Tests** : fichiers Node zéro-dépendance (`node tests/*.js`) pour la
  correction algorithmique (perceptron, alpha-bêta, Q tabulaire, UCT, chaînage avant).

## 7. Jalons

1. **M1 — Squelette** : git init, style, accueil + frise, 8 pages avec prose
   complète (brouillons), annexes en texte → site lisible tôt.
2. **M2 — Gridworld Q-learner** (cœur réutilisé par M3).
3. **M3 — Divergence** (la plus risquée, planifiée tôt avec temps de réglage).
4. **M4 — Perceptron** · **M5 — MCTS** · **M6 — Alpha-bêta** ·
   **M7 — ELIZA + moteur de règles**.
5. **M8 — Finition** : prose finale, annexes, checklist répétition hors-ligne
   (file://, contraste projecteur, plein écran, test du déroulé complet).

## 8. Checklist de répétition

- [ ] Ouvrir chaque page depuis `file://` sans réseau
- [ ] Chaque démo atteignable en 2 clics depuis l'accueil
- [ ] Contraste et tailles lisibles en projection
- [ ] Déroulé chronométré sur les 8 étapes (cible : ≤ 45 min)
- [ ] Plan B prêt si le lieu n'a pas de réseau (site 100 % local)
