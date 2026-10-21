# Ressources de données (`demos/public/datasets/`)

Ce dossier contient les jeux de données servis tels quels par Vite (`public/` est copié
dans `dist/`). La démo `drl-mnist-cnn` choisit un jeu dans son registre
(`src/core/dataset-registry.ts`). Le chargeur navigateur télécharge ensuite les fichiers
depuis `datasets/<id>/` et les décompresse.

Dix jeux sont sélectionnables. Deux utilisent le format IDX classique : quatre archives
par jeu. Huit utilisent le format segmenté : des fragments IDX décrits par un
`index.json`.

## Jeux de données

| Identifiant | Jeu | Format | Classes | Apprentissage | Test | Défaut | Plafond |
|---|---|---|---|---|---|---|---|
| `mnist` | MNIST | IDX | 10 | 60 000 | 10 000 | 10 % | 100 % |
| `fashion-mnist` | Fashion-MNIST | IDX | 10 | 60 000 | 10 000 | 10 % | 100 % |
| `quickdraw` | QuickDraw | segmenté | 15 | 1 597 997 | 399 507 | 2 % | 100 % |
| `octmnist` | OCTMNIST | segmenté | 4 | 108 309 | 1 000 | 10 % | 100 % |
| `pneumoniamnist` | PneumoniaMNIST | segmenté | 2 | 5 232 | 624 | 10 % | 100 % |
| `breastmnist` | BreastMNIST | segmenté | 2 | 624 | 156 | 10 % | 100 % |
| `tissuemnist` | TissueMNIST | segmenté | 8 | 189 106 | 47 280 | 10 % | 100 % |
| `organamnist` | OrganAMNIST | segmenté | 11 | 41 052 | 17 778 | 10 % | 100 % |
| `organcmnist` | OrganCMNIST | segmenté | 11 | 15 367 | 8 216 | 10 % | 100 % |
| `organsmnist` | OrganSMNIST | segmenté | 11 | 16 384 | 8 827 | 10 % | 100 % |

Les colonnes « Défaut » et « Plafond » donnent le pourcentage de fragments chargé par le
navigateur. Tous les jeux autorisent 100 %.

MNIST et Fashion-MNIST comptent 60 000 images d'apprentissage et 10 000 images de test.
Fashion-MNIST pèse environ 31 Mo.

QuickDraw réunit quinze catégories : `book`, `moon`, `cat`, `umbrella`, `cake`, `dog`,
`fish`, `flower`, `star`, `house`, `tree`, `sun`, `bicycle`, `pizza` et `apple` (livre,
lune, chat, parapluie, gâteau, chien, poisson, fleur, étoile, maison, arbre, soleil, vélo,
pizza, pomme). Le convertisseur garde tous les dessins, sans échantillonnage. Il répartit
chaque catégorie en 80 % d'apprentissage et 20 % de test avec une graine fixe (42) : la
répartition est déterministe. Les sources brutes pèsent 1 566 044 336 octets (environ
1,57 Go). La sortie pèse 569 087 508 octets, en 2 011 fragments et deux galeries de
150 images.

Les sept jeux MedMNIST fusionnent les divisions `train` et `val` officielles dans la
fraction d'apprentissage. La division `test` officielle sert à la validation et à la
galerie. La sortie complète pèse 253 534 769 octets, en 509 fragments. Le jeu ChestMNIST
est exclu : il est multi-étiquette, et la démo utilise une sortie softmax mono-étiquette.

## Commandes

| Commande | Jeux | Rôle |
|---|---|---|
| `npm run datasets:data` | MNIST, Fashion-MNIST | Télécharge les archives IDX dans `public/datasets/<id>/`. |
| `npm run quickdraw:data` | QuickDraw | Télécharge les quinze `.npy`, les convertit en fragments IDX et écrit `public/datasets/quickdraw/`. |
| `npm run medmnist:data` | Sept jeux MedMNIST | Télécharge les sept `.npz`, les convertit en fragments IDX et écrit `public/datasets/<id>/`. |

Le crochet `postinstall` ne télécharge que MNIST.

Les trois commandes sont idempotentes : elles vérifient taille et MD5, reprennent un
téléchargement interrompu et ne réécrivent pas les fichiers existants. Un fichier déjà
présent et valide est conservé, sauf `--force`. Un échec supprime le temporaire : aucun
fichier tronqué ne reste sous son nom final.

Les convertisseurs suppriment l'archive brute (`.npy` ou `.npz`) après une conversion
réussie. L'option `--keep-raw` conserve ces archives. Un échec les conserve aussi. Si les
sorties sont complètes et que l'archive brute a disparu, une nouvelle exécution ne
télécharge rien ; si une sortie manque, elle re-télécharge l'archive.

## Fichiers générés

Chaque jeu écrit ses fichiers sous `public/datasets/<id>/`.

Un jeu IDX écrit quatre archives :

- `train-images-idx3-ubyte.gz` et `train-labels-idx1-ubyte.gz` ;
- `t10k-images-idx3-ubyte.gz` et `t10k-labels-idx1-ubyte.gz`.

Un jeu segmenté écrit :

- `index.json` (version 2), le descripteur lu par le navigateur ;
- `<split>-labels-idx1-ubyte.gz`, les libellés complets de chaque fraction ;
- `<split>-c<C>-s<NN>.idx.gz`, les images par classe et par fragment de 1 000 ;
- `train-gallery.idx.gz` et `train-gallery-labels-idx1-ubyte.gz` pour la galerie
  d'apprentissage ;
- `gallery.idx.gz` et `gallery-labels-idx1-ubyte.gz` pour la galerie de validation.

Tous ces fichiers sont générés par les trois commandes ci-dessus. Les fichiers `.gz` et
chaque `index.json` sont ignorés par Git, tout comme le dossier `.cache/`. Le manifeste
`public/datasets/mnist/model-v1.json` reste versionné. La source de vérité sur un clone
est la commande, pas les fichiers.

## Galeries par split

`index.json` décrit deux galeries : `galleries.train` et `galleries.test`. Chacune
contient 10 images par classe. Pour QuickDraw, cela fait 150 images par galerie. L'onglet
« Exemples » et le sélecteur d'exemples du panneau principal partagent ces fichiers et
permettent de passer d'une source à l'autre sans recharger les fragments.

## Chargement dans le navigateur

Le chargeur (`src/components/drl-mnist-cnn/sharded-assets.ts`) ne télécharge que les
fragments nécessaires au pourcentage choisi. Il prend les effectifs et les galeries dans
`index.json`. Les téléchargements sont limités à quatre simultanément.

Pour un jeu segmenté, `N = max(1, round(pourcent / 100 × fragmentsParClasse))` fragments
sont gardés par classe, plafonnés au nombre disponible.

### Mode segmenté

Au-delà de 60 000 images dans la plus grande des deux fractions — apprentissage ou
validation —, l'entraînement et la validation passent en mode segmenté. Un round contient
un fragment par classe sélectionnée. Le navigateur garde au plus 60 fragments décodés en
mémoire (environ 47 Mo), précharge le round suivant et libère les fragments consommés.
Chaque fragment est aussi mis en cache compressé dans la Cache API
(`drl-datasets-v<version>-<jeu>`), avec un repli en mémoire si le quota ou
l'API n'est pas disponible.

À 2 % sur les quinze classes QuickDraw, le navigateur demande 30 fragments
d'apprentissage (30 000 images) et 15 fragments de validation : la plus grande fraction
reste sous le seuil et le jeu complet est matérialisé. À 10 %, quinze classes demandent
environ 150 000 images d'apprentissage : les rounds segmentés prennent le relais. Cinq
classes à 10 % n'en demandent qu'environ 50 000 : le jeu complet reste matérialisé.

## Sélection des classes

QuickDraw affiche une case par catégorie, les cinq premières cochées par défaut. Deux
classes minimum doivent rester cochées. La largeur de sortie, les poids, les effectifs et
les galeries suivent la sélection. Les fragments des classes décochées ne sont pas
téléchargés.

## Sources et cache

Les archives brutes sont mises en cache sous `demos/.cache/` (dossier `quickdraw/` ou
`medmnist/`) puis supprimées après conversion, sauf avec `--keep-raw`. Le dossier
`.cache/` est ignoré par Git.

- MNIST : miroir S3 PyTorch (`ossci-datasets`), voir `public/datasets/mnist/README.md`.
- Fashion-MNIST : dépôt `zalandoresearch/fashion-mnist`.
- QuickDraw : `https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<categorie>.npy`.
- MedMNIST : `https://zenodo.org/records/10519652/files/<nom>.npz?download=1`.

## Licences

| Jeu | Licence | Attribution |
|---|---|---|
| MNIST | Voir la notice MNIST | Yann LeCun, Corinna Cortes, Christopher J.C. Burges |
| Fashion-MNIST | MIT | Zalando Research |
| QuickDraw | CC BY 4.0 | Google, Quick, Draw! |
| MedMNIST (sept jeux) | CC BY 4.0 | MedMNIST (Zenodo 10519652) |

Attribution :

- MNIST : Yann LeCun, Corinna Cortes, Christopher J.C. Burges. Voir
  `public/datasets/mnist/README.md`.
- Fashion-MNIST : Zalando Research.
- QuickDraw : Google, Quick, Draw!.
- MedMNIST : MedMNIST (Zenodo 10519652).
