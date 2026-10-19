# MNIST CNN — Note de faisabilité WebGL / WebGPU

## Objet

Cette note évalue deux pistes pour la démo interactive **« Classifieur de chiffre
MNIST »** de la présentation :

1. le prototype GLSL/WebGL existant (Shadertoy, inférence seulement) ;
2. l'entraînement GPU dans le navigateur (WebGL2 fragment, WebGPU compute), en le
   comparant à TensorFlow.js.

La note ne contient ni shader ni code d'application. **Aucun benchmark n'a été mesuré**
: les chiffres de mémoire sont des estimations. Public : l'équipe de la présentation.
Décision attendue : quelle pile portera la démo produit.

## Le modèle de référence

Source : `tf-playground/mnist/train.py` (PyTorch, code vérifié). Entrée 28×28 en
niveaux degris, **un canal**.

| # | Couche | Sortie par échantillon |
|---|---|---|
| 1 | conv 1→3, 3×3, pas 1, aucun padding, sigmoid | 3×26×26 |
| 2 | conv 3→3, 3×3, pas 2, padding symétrique 1, sigmoid | 3×13×13 |
| 3 | conv 3→3, 3×3, pas 1, aucun padding, sigmoid | 3×11×11 |
| 4 | conv 3→3, 3×3, pas 2, aucun padding, sigmoid | 3×5×5 |
| — | flatten | 75 |
| 5 | dense 75→8, sigmoid | 8 |
| 6 | dense 8→10 (logits) | 10 |

Le dropout (p = 0.2) se situe après la convolution 4 et n'est actif qu'à
l'entraînement (`model.train()`). Perte : entropie croisée. Optimiseur : Adam, taux
`3e-3`. Lot d'entraînement : 64. Le modèle retourne des logits — le softmax est
commenté dans `train.py`, `CrossEntropyLoss` travaille directement dessus.

Le réseau compte **980 paramètres** (≈ 4 ko en float32) : il est minuscule. Ce sont la
mémoire d'activation et la logique de rétropropagation qui dominent, pas les poids.

## Le prototype Shadertoy existant (inférence seulement)

`tf-playground/mnist/shadertoy.json` est un prototype Shadertoy publié (« Neural network
prototype - MNIST », utilisateur `pcouy`). Faits vérifiés :

- zone de dessin à la souris (carré noir à gauche) ; `[ESPACE]` efface, dessiner avec
  `[SHIFT]` efface ;
- les autres carrés visualisent les sorties intermédiaires des convolutions ;
- bandeau au-dessus des chiffres : sortie de la couche linéaire avant softmax ;
- bandeau à séparateurs jaunes : sortie finale après softmax ;
- bandeau du bas : Buffer A à l'échelle d'origine ;
- poids exportés depuis un script Python/PyTorch ;
- les trois entrées déclarent des samplers `internal: byte` — cela ne dit rien du
  format des buffers internes.

**Non vérifié.** Le corps du shader tient sur une seule ligne JSON ; les permissions
des outils ont tronqué le fichier à 2 000 caractères et bloqué toute ré-extraction
(coquille refusée). Restent donc non vérifiés :

- la disposition exacte des buffers internes ;
- la précision exacte des samplers ;
- la correspondance exacte des coordonnées (retournement y, espacement des régions) ;
- le format interne des activations — **ne pas affirmer qu'il s'agit de RGBA8** ;
- l'indexation conv/dense et la disposition des tenseurs.

D'après l'analyse du projet, le chemin actif `ALTERNATE` utiliserait quatre couches de
convolution et dix sorties directes ; le chemin optionnel à deux couches denses serait
désactivé. À confirmer par relecture du shader.

Le prototype suppose **trois canaux en dur** et des régions de couches fixes, alors
que le modèle de référence a une entrée un canal. Aucun test de parité n'existe. Les
points à tester lors de toute reprise : bascule des axes du noyau, bascule y de
l'image, padding, ordre d'aplatissement (*flatten*), transpose dense.

## Pourquoi l'entraînement shader est un problème distinct

Le prototype Shadertoy démontre **l'inférence** : une passe avant, pixel par pixel,
sans réduction. **L'entraînement** exige, en plus, des rétropropagation et des
réductions (sommes sur le lot et les positions spatiales) pour les gradients du noyau
et du biais. Les deux sujets ne doivent pas être confondus : la faisabilité de
l'inférence ne dit rien de celle de l'entraînement.

## Comparaison des trois piles GPU

| Critère | WebGL2 (fragment) | WebGPU (compute) | TensorFlow.js |
|---|---|---|---|
| Étape de calcul dédiée | Non — rasterisation fragment uniquement | Oui — compute shaders WGSL | Oui (backend WebGL/WebGPU) |
| Réductions pour les gradients | Multipass sur textures float, complexe | Réductions partielles par workgroup + seconde passe | Gérées par la bibliothèque |
| Atomics shader | Aucun en GLSL ES 3.00 | Types entiers uniquement (`atomic<i32>`, `atomic<u32>`), pas de float natif | Sans objet |
| Cibles couleur float | `EXT_color_buffer_float` requise | Selon l'appareil (ex. `rgba32float`) | Géré par la bibliothèque |
| Effort d'implémentation | Élevé | Élevé, mais proche de l'entraînement GPU classique | Faible |
| Performance navigateur | Inconnue | Inconnue | Inconnue |

Précisions :

- WebGL2 ne fournit **ni étape de calcul ni atomics shader** (GLSL ES 3.00). Les
  gradients d'entraînement exigent des réductions sur le lot et les positions
  spatiales ; des réductions multipass sur textures float sont possibles mais
  complexes. Les cibles float exigent `EXT_color_buffer_float`. Vérifier `highp` et
  les limites de texture à l'exécution.
- WebGPU compute est **plus adapté à l'expérimentation d'entraînement GPU** que le
  fragment seul, mais les atomics WGSL sont entiers : utiliser des réductions
  partielles par workgroup suivies d'une seconde passe de réduction.
- Aucune performance navigateur n'est promise pour les trois piles : aucun benchmark
  n'a été exécuté. Les lancements de shaders, la synchronisation et les réductions
  sont des risques probablement **plus grands** que la mémoire tenseur.

## Mémoire d'activation (estimation)

Tampons par échantillon, d'après les dimensions vérifiées de `train.py` :

| Tampon | Flottants | Octets (float32) |
|---|---|---|
| entrée 1×28×28 | 784 | 3 136 |
| sortie conv1 3×26×26 | 2 028 | 8 112 |
| sortie conv2 3×13×13 | 507 | 2 028 |
| sortie conv3 3×11×11 | 363 | 1 452 |
| sortie conv4 3×5×5 | 75 | 300 |
| sortie dense1 | 8 | 32 |
| logits | 10 | 40 |
| **Total activations** | **3 775** | **15 100** |

Pour l'entraînement, activations + deltas ≈ 2 × 15 100 = **30 200 octets/échantillon** :

- lot 32 : ≈ 0,92 Mio (966 400 octets) ;
- lot 64 : ≈ 1,84 Mio (1 932 800 octets).

L'estimation antérieure du projet (≈ 0,83 Mio à lot 32, ≈ 1,65 Mio à lot 64) correspond
à ces totaux **hors tampon d'entrée** (866 ko et 1 732 ko respectivement). Il s'agit
d'une estimation, pas d'un benchmark ; elle exclut les surcoûts d'implémentation
(*ping-pong* de textures, alignement, mise en scène des shaders).

## Spike de parité minimal : une convolution

Avant toute mise à l'échelle, valider **une** convolution complète, la plus simple :
**convolution 1** (1→3, 3×3, pas 1, aucun padding, 30 paramètres).

1. Fixer un lot (ex. 4 échantillons 28×28, graine fixe) et un delta amont ∂L/∂a fixe.
2. Calculer dans le shader les quatre tenseurs :
   - passe avant : `a = sigmoid(conv(x))` ;
   - gradient d'entrée ∂L/∂x ;
   - gradient du noyau ∂L/∂W ;
   - gradient du biais ∂L/∂b.
3. Calculer la même chose dans une référence CPU (modèle PyTorch ou référence pure
   NumPy/TypeScript) sur les mêmes entrées.

**Critère GO/NO-GO :**

- **GO** — les quatre tenseurs correspondent à la référence avec une erreur absolue
  maximale ≤ 1e-4 (float32), formes et dispositions identiques. Alors on met à l'échelle
  couche par couche (conv2 avec padding, conv4 avec pas 2, puis denses), en refaisant
  le test de parité à chaque étape.
- **NO-GO** — un gradient dépasse la tolérance. On abandonne l'entraînement shader ;
  on garde TensorFlow.js pour l'entraînement et une inférence TypeScript pure pour la
  visualisation.

## Recommandation

1. **Démo produit : TensorFlow.js** pour l'entraînement et l'inférence.
2. **Séparer l'adaptateur d'entraînement** (TensorFlow.js) du passage d'inférence et de
   visualisation en TypeScript pur — conforme à la contrainte `demos/README.md` :
   core pur testable par Vitest, sans import DOM, Lit, D3 ou Three.
3. **Ne promettre aucune vitesse navigateur** avant benchmark réel.
4. Si l'entraînement shader reste un objectif : exécuter d'abord le spike de parité de
   la section précédente ; ne monter en puissance qu'en cas de GO.

## Références

- [WebGL 2.0 Specification — Khronos](https://registry.khronos.org/webgl/specs/latest/2.0/)
- [EXT_color_buffer_float — Khronos](https://registry.khronos.org/webgl/extensions/EXT_color_buffer_float/)
- [GLSL ES Specification 3.00 — Khronos (PDF)](https://registry.khronos.org/OpenGL/specs/es/3.0/GLSL_ES_Specification_3.00.pdf)
- [WebGPU — W3C](https://www.w3.org/TR/webgpu/)
- [WGSL — W3C](https://www.w3.org/TR/WGSL/)

## Limites de cette note

- Corps du shader Shadertoy non relu en entier (troncature à 2 000 caractères,
  coquille refusée) : voir la liste de la section dédiée.
- Mémoire d'activation : estimation analytique, pas un benchmark.
- Performances navigateurs des trois piles : inconnues, aucun benchmark exécuté.
- `model.glsl` (export plus ancien : 3 convolutions, linéaire direct 48→10) n'est pas
  le prototype actif ; il illustre un format d'export différent de `shadertoy.json`.
