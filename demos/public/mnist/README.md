# Ressources MNIST (`demos/public/mnist/`)

Ce dossier contient les ressources servies telles quelles par Vite (`public/` est copié
dans `dist/`). Le chargeur navigateur
`src/components/drl-mnist-cnn/mnist-assets.ts` récupère les archives IDX et les
décompresse.

## Fichiers attendus

Les quatre archives IDX compressées ne sont **pas versionnées** dans le dépôt. Elles
sont téléchargées à la demande dans ce dossier :

| Fichier | Octets | MD5 |
|---|---|---|
| `train-images-idx3-ubyte.gz` | 9 912 422 | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
| `train-labels-idx1-ubyte.gz` | 28 881 | `d53e105ee54ea40749a09fcbcd1e9432` |
| `t10k-images-idx3-ubyte.gz` | 1 648 877 | `9fb629c4189551a2d022fa330f9573f3` |
| `t10k-labels-idx1-ubyte.gz` | 4 542 | `ec29112dd5afa0611ce80d1b7f02629c` |

Ces archives sont des copies exactes des originaux, sans modification : les octets,
les tailles et les empreintes MD5 correspondent à la source.

## Téléchargement

`npm install` lance le crochet `postinstall`, qui essaie de télécharger les archives
(`node scripts/fetch-mnist-data.mjs --optional`). En cas d'échec réseau, cette étape
affiche un avertissement et n'empêche pas l'installation.

Pour les télécharger explicitement, ou pour réessayer après un échec :

```bash
npm run mnist:data
```

Le script vérifie la taille et le MD5 de chaque fichier après téléchargement, et
conserve un fichier déjà présent dont la taille et le MD5 sont corrects. Il échoue
avec un code non nul si une vérification échoue. La liste des fichiers attendus et
leurs empreintes vivent dans `scripts/mnist-data-manifest.mjs`.

## Sources

Les quatre fichiers proviennent du miroir S3 maintenu par PyTorch (dépôt
`ossci-datasets`) :

```
https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
```

## Attribution

Le jeu de données MNIST est l'œuvre de :

- **Yann LeCun** ;
- **Corinna Cortes** ;
- **Christopher J.C. Burges**.

Référence d'origine : *The MNIST database of handwritten digits*, page officielle
<http://yann.lecun.com/exdb/mnist/>. La page officielle ne comporte **aucune**
mention de licence : elle ne déclare donc pas la licence CC BY-SA 3.0 et ne
l'accorde pas explicitement. Cette redistribution s'appuie sur la licence décrite
ci-dessous, telle que documentée par Keras, et non sur la page officielle.

## Licence (CC BY-SA 3.0)

La documentation Keras de `MNIST` indique que le jeu est distribué sous licence
**Creative Commons Attribution-ShareAlike 3.0** (CC BY-SA 3.0). Texte de la
licence : <https://creativecommons.org/licenses/by-sa/3.0/>.

En résumé, toute réutilisation doit créditer les auteurs ci-dessus et partager
les œuvres dérivées sous la même licence. Les fichiers IDX téléchargés dans ce
dossier sont redistribués sous cette licence ; ils ne sont pas modifiés.

## Poids du modèle

`model-v1.json` est le manifeste de poids du CNN exporté depuis le point de
contrôle `tf-playground/mnist/model.pth`. Ce point de contrôle appartient au
propriétaire du dépôt, qui a autorisé sa redistribution. Le manifeste est produit
par `demos/scripts/export_mnist_cnn_checkpoint.py` (voir
`src/core/mnist-cnn-assets.ts` pour le format et la validation).
