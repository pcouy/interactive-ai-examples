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
| `quickdraw` | QuickDraw | segmenté | 5 | 490 571 | 122 645 | 2 % | 10 % |
| `octmnist` | OCTMNIST | segmenté | 4 | 108 309 | 1 000 | 10 % | 10 % |
| `pneumoniamnist` | PneumoniaMNIST | segmenté | 2 | 5 232 | 624 | 10 % | 10 % |
| `breastmnist` | BreastMNIST | segmenté | 2 | 624 | 156 | 10 % | 10 % |
| `tissuemnist` | TissueMNIST | segmenté | 8 | 189 106 | 47 280 | 10 % | 10 % |
| `organamnist` | OrganAMNIST | segmenté | 11 | 41 052 | 17 778 | 10 % | 10 % |
| `organcmnist` | OrganCMNIST | segmenté | 11 | 15 367 | 8 216 | 10 % | 10 % |
| `organsmnist` | OrganSMNIST | segmenté | 11 | 16 384 | 8 827 | 10 % | 10 % |

Les colonnes « Défaut » et « Plafond » donnent le pourcentage de fragments chargé par le
navigateur. Le plafond limite la mémoire utilisée. Un jeu IDX n'a pas de plafond propre :
sa valeur vaut 100 %.

MNIST et Fashion-MNIST comptent 60 000 images d'apprentissage et 10 000 images de test.
Fashion-MNIST pèse environ 31 Mo.

QuickDraw réunit les cinq catégories `book`, `moon`, `cat`, `umbrella` et `cake` (livre,
lune, chat, parapluie, gâteau). Le convertisseur garde tous les dessins, sans
échantillonnage. Il répartit chaque catégorie en 80 % d'apprentissage et 20 % de test
avec une graine fixe (42) : la répartition est déterministe. La sortie pèse
179 618 070 octets, en 622 fichiers.

Les sept jeux MedMNIST fusionnent les divisions `train` et `val` officielles dans la
fraction d'apprentissage. La division `test` officielle sert à la validation et à la
galerie. La sortie complète pèse 253 234 151 octets, en 509 fragments. Le jeu ChestMNIST
est exclu : il est multi-étiquette, et la démo utilise une sortie softmax mono-étiquette.

## Commandes

| Commande | Jeux | Rôle |
|---|---|---|
| `npm run datasets:data` | MNIST, Fashion-MNIST | Télécharge les archives IDX dans `public/datasets/<id>/`. |
| `npm run quickdraw:data` | QuickDraw | Télécharge les cinq `.npy`, les convertit en fragments IDX et écrit `public/datasets/quickdraw/`. |
| `npm run medmnist:data` | Sept jeux MedMNIST | Télécharge les sept `.npz`, les convertit en fragments IDX et écrit `public/datasets/<id>/`. |

Le crochet `postinstall` ne télécharge que MNIST.

Les trois convertisseurs sont idempotents. Chacun vérifie la taille et le MD5 de chaque
fichier téléchargé. Un fichier déjà présent et valide est conservé, sauf `--force`. Un
échec supprime le temporaire : aucun fichier tronqué ne reste sous son nom final.

## Fichiers générés

Chaque jeu écrit ses fichiers sous `public/datasets/<id>/`.

Un jeu IDX écrit quatre archives :

- `train-images-idx3-ubyte.gz` et `train-labels-idx1-ubyte.gz` ;
- `t10k-images-idx3-ubyte.gz` et `t10k-labels-idx1-ubyte.gz`.

Un jeu segmenté écrit :

- `index.json`, le descripteur lu par le navigateur ;
- `<split>-labels-idx1-ubyte.gz`, les libellés complets de chaque fraction ;
- `<split>-c<C>-s<NN>.idx.gz`, les images par classe et par fragment de 1 000 ;
- `gallery.idx.gz` et `gallery-labels-idx1-ubyte.gz`, la galerie.

Tous ces fichiers sont générés par les commandes ci-dessus. Les archives `.gz` et le
dossier `.cache/` sont ignorés par Git. Le descripteur `index.json` est généré lui aussi ;
seul celui de QuickDraw est aujourd'hui versionné.

## Chargement dans le navigateur

Le chargeur (`src/components/drl-mnist-cnn/sharded-assets.ts`) ne télécharge que les
fragments nécessaires au pourcentage choisi. Il prend les effectifs et la galerie dans
`index.json`. Les fichiers décodés sont mis en cache en mémoire. Les téléchargements sont
limités à quatre simultanément.

Pour un jeu segmenté, `N = max(1, round(pourcent / 100 × fragmentsParClasse))` fragments
sont gardés par classe, plafonnés au nombre disponible. À 2 %, le navigateur récupère 15
fragments pour QuickDraw (environ 4 Mo).

## Sources et cache

Les archives brutes sont mises en cache sous `demos/.cache/` (dossier `quickdraw/` ou
`medmnist/`). Les convertisseurs QuickDraw et MedMNIST ne suppriment rien du cache : vous
pouvez supprimer `demos/.cache/` après conversion.

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
