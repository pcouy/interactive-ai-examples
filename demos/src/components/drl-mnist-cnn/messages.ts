/**
 * Catalogues de messages du groupe de démonstration MNIST CNN.
 *
 * Chaque groupe exporte un catalogue typé `fr`/`en`, l'union de ses clés et un
 * helper qui appelle `translate`. Les textes français reproduisent mot pour mot
 * les libellés d'origine ; les textes anglais suivent la terminologie de la
 * démonstration.
 */

import type { Locale } from '../../i18n/locale.js';
import { type Catalog, type TranslateParams, translate } from '../../i18n/translate.js';

const mnistCnnFr = {
  drawing: 'Dessin',
  inputPanelAria: "Zone de dessin et choix de l'entrée",
  drawingTitle: 'Dessin 28×28',
  drawingCanvasAria: 'Surface de dessin 28 par 28 pixels',
  dataset: 'Jeu de données',
  loadingDataset: 'Chargement du jeu…',
  example: 'Exemple {name}',
  chooseExample: 'Choisir un exemple de {name}',
  loadingGallery: 'Chargement de la galerie…',
  loadGallery: 'Charger la galerie…',
  browseDigitExamplesAria: 'Parcourir les exemples du chiffre choisi',
  previous: '← Précédent',
  next: 'Suivant →',
  clear: 'Effacer',
  trainingPanelAria: 'Entraînement du modèle',
  trainingTitle: 'Entraînement',
  trainingParagraph:
    "TensorFlow.js est chargé au premier clic sur un bouton d'entraînement, ainsi que les exemples MNIST locaux. L'entraînement repart toujours des poids affichés : pré-entraînés, aléatoires ou modifiés à la main. « Entraîner 1 époque » fait une passe ; « Entraîner en boucle » enchaîne les époques jusqu'à Pause. La courbe en pointillés mesure la précision d'entraînement sur l'échantillon borné ; la courbe en trait plein mesure la précision de validation sur le sous-ensemble tenu à l'écart.",
  trainingControlsAria: "Lancer et contrôler l'entraînement",
  trainOneEpoch: 'Entraîner 1 époque',
  trainInLoop: 'Entraîner en boucle',
  pause: 'Pause',
  resetWeights: 'Réinitialiser les poids',
  randomWeights: 'Poids aléatoires',
  trainingSlidersAria:
    "Fraction des jeux d'entraînement et de validation, taux d'apprentissage et taille des lots",
  trainingData: "Données d'entraînement",
  validation: 'Validation',
  learningRate: "Taux d'apprentissage",
  batchSize: 'Taille des lots',
  trainingError: "Erreur d'entraînement : {message}",
  slidersNote:
    "Les curseurs de pourcentage, de taux et de taille des lots ne prennent effet qu'au prochain entraînement, sans interrompre une run.",
  resultsPanelAria: "Statistiques, probabilités et courbes d'entraînement",
  resultsTitle: 'Résultats',
  parameters: 'Paramètres',
  predictedClass: 'Classe prédite',
  input: 'Entrée',
  weights: 'Poids',
  probabilitiesPerClass: 'Probabilités par classe',
  probabilitiesAria: 'Probabilités par classe pour {name}',
  waitingFirstForward: 'En attente du premier passage avant.',
  filtersPerLayer: 'Filtres par couche',
  filtersParagraph:
    "Chaque curseur arrête l'entraînement et reconstruit le modèle : poids pré-entraînés (manifeste) si l'architecture correspond, poids déterministes (graine {seed}) sinon. Les curseurs de profondeur changent le nombre de couches de convolution (0 à 4) et de couches denses (1 à 3) ; le choix de la fonction d'activation arrête aussi la run et reconstruit le modèle ; le manifeste pré-entraîné ne s'applique qu'à la profondeur de référence (quatre convolutions, deux couches denses) et avec la sigmoïde, toute autre configuration repart des poids déterministes. Les curseurs de largeur règlent les unités de chaque couche dense cachée (1 à 128) et arrêtent la run ; le manifeste exige la largeur de référence, soit une couche cachée de 8 unités.",
  networkDepthAria: 'Profondeur du réseau',
  convLayers: 'Couches de convolution',
  denseLayers: 'Couches denses',
  hiddenDenseSizesAria: 'Taille des couches denses cachées',
  activationFunction: "Fonction d'activation",
  activationFunctionAria: "Fonction d'activation des convolutions et de la couche cachée",
  sigmoid: 'Sigmoïde',
  filterCountsAria: 'Nombre de filtres de chaque couche de convolution',
  filtersLayer: 'Filtres {layer}',
  filtersAndExamplesAria: 'Filtres et exemples',
  filtersTabsAria: 'Filtres de convolution et exemples',
  convFiltersTitle: 'Filtres de convolution 3×3',
  noConv: "Aucune couche de convolution : l'entrée 28×28 est aplatie vers les couches denses.",
  convLayerChannels: '{layer} — {count} canal(aux)',
  filterCanvasAria: 'Filtre {layer}, canal de sortie {channel}, biais {bias}',
  biasCaption: 'sortie {channel} · b = {value}',
  examplesCanvasAria: 'Exemple de la galerie, surface de dessin 28 par 28 pixels',
  sourceOfExamples: 'Source des exemples',
  exampleTab: 'Exemple {name} (onglet Exemples)',
  chooseExampleTab: "Choisir un exemple de {name} dans l'onglet Exemples",
  galleryUnavailable: 'Galerie indisponible : {message}',
  chooseLoadGallery: 'Choisissez « Charger la galerie… » dans la liste pour afficher des exemples.',
  browseGalleryAria: 'Parcourir les exemples de la galerie',
  featureMapsAria: 'Cartes de caractéristiques',
  featureMapsTitle: 'Cartes de caractéristiques',
  activationsAfter: 'Activations après {activation} ; rouge = {low}, gris = {mid}, bleu = {high}.',
  featureMapSizes: 'Tailles : {sizes}.',
  noConvMap: 'Aucune carte de convolution.',
  featureMapAria: 'Carte de caractéristiques {name}, canal {channel}',
  channel: 'canal {channel}',
  apply: 'Appliquer',
  cancel: 'Annuler',
  denseLayersTitle: 'Couches denses',
  activationsOf: 'Activations de {title}',
  probabilitiesMaxAria: 'Probabilités par classe, maximum surligné',
  classesLegend: 'Classes ({count} sur {total})',
  minClassesHint: 'Deux classes minimum. Sortie : {count} neurones.',
  classLabel: '{index} — {name}',
  galleryInputSource: 'Galerie {source} · Exemple {name} ({className})',
  galleryInputHintDrawing:
    "Dessinez une entrée à la souris ou au doigt, ou choisissez un exemple de {name} ; l'inférence se met à jour pendant le tracé.",
  galleryInputHintExample:
    'Exemple de la galerie {source} {name} : la classe attendue est « {className} ».',
  linkLabel: 'conv{layer} — entrée {input}, sortie {output}',
  linkWeightCount: '{name} · 1 poids sur {count}',
  filterSubtitleMean:
    "Moyenne des canaux d'entrée pour chaque canal de sortie ; bleu = positif, rouge = négatif.",
  filterSubtitleKernel: 'La toile du couple sélectionné montre le noyau 3×3 exact.',
  denseLayerTitle: 'Dense {index} — {units} unités',
  outputLayerTitle: 'Sortie — {classes} classes',
  denseSizeLabel: 'Unités de la couche dense {layer}',
  denseCellLabel: 'unité {unit} : {value}',
  outputCellLabel: 'classe {digit}{className} : {percent} %',
  outputCellMax: '{base}, maximum',
  activationScaleAria: 'Échelle des cartes : rouge à {low}, gris à {mid}, bleu à {high}',
  chartWaiting: 'En attente de deux époques',
  chartTrainingLoss: "Perte d'entraînement",
  chartTraining: 'Entraînement',
  chartValidation: 'Validation',
  chartAccuracy: 'Précision (entraînement et validation)',
} as const;

export type MnistCnnMessageKey = keyof typeof mnistCnnFr;

export const MNIST_CNN_MESSAGES: Catalog<MnistCnnMessageKey> = {
  fr: mnistCnnFr,
  en: {
    drawing: 'Drawing',
    inputPanelAria: 'Drawing area and input selection',
    drawingTitle: 'Drawing 28×28',
    drawingCanvasAria: 'Drawing area, 28 by 28 pixels',
    dataset: 'Dataset',
    loadingDataset: 'Loading the dataset…',
    example: 'Example {name}',
    chooseExample: 'Choose an example from {name}',
    loadingGallery: 'Loading the gallery…',
    loadGallery: 'Load the gallery…',
    browseDigitExamplesAria: 'Browse the examples of the selected digit',
    previous: '← Previous',
    next: 'Next →',
    clear: 'Clear',
    trainingPanelAria: 'Model training',
    trainingTitle: 'Training',
    trainingParagraph:
      'TensorFlow.js loads on the first click on a training button, along with the local MNIST examples. Training always starts from the displayed weights: pre-trained, random or hand-edited. "Train 1 epoch" runs one pass; "Train in a loop" chains epochs until Pause. The dashed curve measures training accuracy on the bounded sample; the solid curve measures validation accuracy on the held-out subset.',
    trainingControlsAria: 'Start and control training',
    trainOneEpoch: 'Train 1 epoch',
    trainInLoop: 'Train in a loop',
    pause: 'Pause',
    resetWeights: 'Reset weights',
    randomWeights: 'Random weights',
    trainingSlidersAria: 'Training and validation set fraction, learning rate and batch size',
    trainingData: 'Training data',
    validation: 'Validation',
    learningRate: 'Learning rate',
    batchSize: 'Batch size',
    trainingError: 'Training error: {message}',
    slidersNote:
      'The percentage, rate and batch size sliders take effect at the next training only, without interrupting a run.',
    resultsPanelAria: 'Statistics, probabilities and training curves',
    resultsTitle: 'Results',
    parameters: 'Parameters',
    predictedClass: 'Predicted class',
    input: 'Input',
    weights: 'Weights',
    probabilitiesPerClass: 'Probabilities per class',
    probabilitiesAria: 'Probabilities per class for {name}',
    waitingFirstForward: 'Waiting for the first forward pass.',
    filtersPerLayer: 'Filters per layer',
    filtersParagraph:
      'Each slider stops training and rebuilds the model: pre-trained weights (manifest) if the architecture matches, deterministic weights (seed {seed}) otherwise. The depth sliders change the number of convolution layers (0 to 4) and dense layers (1 to 3); the activation function choice also stops the run and rebuilds the model; the pre-trained manifest applies only to the reference depth (four convolutions, two dense layers) with the sigmoid, and every other configuration starts from deterministic weights. The width sliders set the units of each hidden dense layer (1 to 128) and stop the run; the manifest requires the reference width, a hidden layer of 8 units.',
    networkDepthAria: 'Network depth',
    convLayers: 'Convolution layers',
    denseLayers: 'Dense layers',
    hiddenDenseSizesAria: 'Hidden dense layer size',
    activationFunction: 'Activation function',
    activationFunctionAria: 'Activation function of the convolutions and the hidden layer',
    sigmoid: 'Sigmoid',
    filterCountsAria: 'Number of filters in each convolution layer',
    filtersLayer: 'Filters {layer}',
    filtersAndExamplesAria: 'Filters and examples',
    filtersTabsAria: 'Convolution filters and examples',
    convFiltersTitle: '3×3 convolution filters',
    noConv: 'No convolution layer: the 28×28 input is flattened into the dense layers.',
    convLayerChannels: '{layer} — {count} channel(s)',
    filterCanvasAria: 'Filter {layer}, output channel {channel}, bias {bias}',
    biasCaption: 'output {channel} · b = {value}',
    examplesCanvasAria: 'Gallery example, drawing area, 28 by 28 pixels',
    sourceOfExamples: 'Source of examples',
    exampleTab: 'Example {name} (Examples tab)',
    chooseExampleTab: 'Choose an example from {name} in the Examples tab',
    galleryUnavailable: 'Gallery unavailable: {message}',
    chooseLoadGallery: 'Choose "Load the gallery…" in the list to show examples.',
    browseGalleryAria: 'Browse the gallery examples',
    featureMapsAria: 'Feature maps',
    featureMapsTitle: 'Feature maps',
    activationsAfter: 'Activations after {activation}; red = {low}, grey = {mid}, blue = {high}.',
    featureMapSizes: 'Sizes: {sizes}.',
    noConvMap: 'No convolution map.',
    featureMapAria: 'Feature map {name}, channel {channel}',
    channel: 'channel {channel}',
    apply: 'Apply',
    cancel: 'Cancel',
    denseLayersTitle: 'Dense layers',
    activationsOf: 'Activations of {title}',
    probabilitiesMaxAria: 'Probabilities per class, maximum highlighted',
    classesLegend: 'Classes ({count} of {total})',
    minClassesHint: 'At least two classes. Output: {count} neurons.',
    classLabel: '{index} — {name}',
    galleryInputSource: 'Gallery {source} · Example {name} ({className})',
    galleryInputHintDrawing:
      'Draw an input with the mouse or a finger, or choose an example from {name}. Inference updates while you draw.',
    galleryInputHintExample:
      'Gallery {source} example {name}: the expected class is "{className}".',
    linkLabel: 'conv{layer} — input {input}, output {output}',
    linkWeightCount: '{name} · 1 weight out of {count}',
    filterSubtitleMean:
      'Mean of the input channels for each output channel; blue = positive, red = negative.',
    filterSubtitleKernel: 'The canvas of the selected pair shows the exact 3×3 kernel.',
    denseLayerTitle: 'Dense {index} — {units} units',
    outputLayerTitle: 'Output — {classes} classes',
    denseSizeLabel: 'Units of dense layer {layer}',
    denseCellLabel: 'unit {unit}: {value}',
    outputCellLabel: 'class {digit}{className}: {percent}%',
    outputCellMax: '{base}, maximum',
    activationScaleAria: 'Map scale: red at {low}, grey at {mid}, blue at {high}',
    chartWaiting: 'Waiting for two epochs',
    chartTrainingLoss: 'Training loss',
    chartTraining: 'Training',
    chartValidation: 'Validation',
    chartAccuracy: 'Accuracy (training and validation)',
  },
};

export function mnistCnnMessage(
  locale: Locale,
  key: MnistCnnMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, MNIST_CNN_MESSAGES, key, params);
}

const demoStatusFr = {
  modified: 'Modifié',
  random: 'Aléatoires',
  trained: 'Entraîné',
  pretrained: 'Pré-entraîné',
  deterministic: 'Déterministes',
  input: 'Entrée : {source}',
  predictedClass: 'classe prédite : {className}',
  parameters: '{count} paramètres',
  batches: 'lots : {count}',
  loss: 'perte : {value}',
  validationAccuracy: 'précision validation (sous-ensemble) : {value}',
  time: 'temps : {value}',
  epoch: 'époque {count}',
  round: 'round {index}/{total}',
  window: 'fenêtre {count} fragments',
  trainingCount: 'entraînement : {count} exemples',
  validationCount: 'validation : {count} exemples',
  weightsModified: 'poids modifiés à la main',
  manifestUnavailable: 'manifeste inaccessible : {message}',
} as const;

export type DemoStatusMessageKey = keyof typeof demoStatusFr;

export const DEMO_STATUS_MESSAGES: Catalog<DemoStatusMessageKey> = {
  fr: demoStatusFr,
  en: {
    modified: 'Modified',
    random: 'Random',
    trained: 'Trained',
    pretrained: 'Pre-trained',
    deterministic: 'Deterministic',
    input: 'input: {source}',
    predictedClass: 'predicted class: {className}',
    parameters: '{count} parameters',
    batches: 'batches: {count}',
    loss: 'loss: {value}',
    validationAccuracy: 'validation accuracy (subset): {value}',
    time: 'time: {value}',
    epoch: 'epoch {count}',
    round: 'round {index}/{total}',
    window: 'window {count} shards',
    trainingCount: 'training: {count} examples',
    validationCount: 'validation: {count} examples',
    weightsModified: 'weights modified manually',
    manifestUnavailable: 'manifest unavailable: {message}',
  },
};

export function demoStatusMessage(
  locale: Locale,
  key: DemoStatusMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, DEMO_STATUS_MESSAGES, key, params);
}

const trainingStateFr = {
  training: 'Entraînement',
  paused: 'En pause',
  trained: 'Entraîné',
  pretrained: 'Pré-entraîné',
  random: 'Poids aléatoires',
  deterministic: 'Poids déterministes',
} as const;

export type TrainingStateMessageKey = keyof typeof trainingStateFr;

export const TRAINING_STATE_MESSAGES: Catalog<TrainingStateMessageKey> = {
  fr: trainingStateFr,
  en: {
    training: 'Training',
    paused: 'Paused',
    trained: 'Trained',
    pretrained: 'Pre-trained',
    random: 'Random weights',
    deterministic: 'Deterministic weights',
  },
};

export function trainingStateMessage(
  locale: Locale,
  key: TrainingStateMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, TRAINING_STATE_MESSAGES, key, params);
}

const trainingChartFr = {
  noData: '{name} : aucune donnée.',
  singlePoint: '{name} : {count} point, dernière valeur {value}.',
  points: '{name} : {count} points, dernière valeur {value}.',
  seriesSummary: '{name} : {segments}',
} as const;

export type TrainingChartMessageKey = keyof typeof trainingChartFr;

export const TRAINING_CHART_MESSAGES: Catalog<TrainingChartMessageKey> = {
  fr: trainingChartFr,
  en: {
    noData: '{name}: no data.',
    singlePoint: '{name}: {count} point, last value {value}.',
    points: '{name}: {count} points, last value {value}.',
    seriesSummary: '{name}: {segments}',
  },
};

export function trainingChartMessage(
  locale: Locale,
  key: TrainingChartMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, TRAINING_CHART_MESSAGES, key, params);
}

const filterTabsFr = {
  filters: 'Filtres de convolution',
  examples: 'Exemples',
} as const;

export type FilterTabsMessageKey = keyof typeof filterTabsFr;

export const FILTER_TABS_MESSAGES: Catalog<FilterTabsMessageKey> = {
  fr: filterTabsFr,
  en: {
    filters: 'Convolution filters',
    examples: 'Examples',
  },
};

export function filterTabsMessage(
  locale: Locale,
  key: FilterTabsMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, FILTER_TABS_MESSAGES, key, params);
}

const galleryFr = {
  train: 'Entraînement',
  test: 'Validation',
  trainSource: "d'entraînement",
  testSource: 'de validation',
} as const;

export type GalleryMessageKey = keyof typeof galleryFr;

export const GALLERY_MESSAGES: Catalog<GalleryMessageKey> = {
  fr: galleryFr,
  en: {
    train: 'Training',
    test: 'Validation',
    trainSource: 'training',
    testSource: 'validation',
  },
};

export function galleryMessage(
  locale: Locale,
  key: GalleryMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, GALLERY_MESSAGES, key, params);
}

const weightEditorFr = {
  notWeight: "La sélection courante n'est pas un poids.",
  missingWeight: "Ce poids n'existe pas dans l'architecture courante.",
  empty: 'Saisissez une valeur.',
  nonFinite: 'Valeur non finie : le nombre dépasse les limites.',
  unreadable: 'Nombre illisible : utilisez un décimal, par exemple 0,5.',
  weightsMismatch: "Les poids affichés ne correspondent pas à l'architecture courante.",
  hint: 'Cliquez une liaison, puis une case de la toile de filtre en aval, puis saisissez la valeur.',
} as const;

export type WeightEditorMessageKey = keyof typeof weightEditorFr;

export const WEIGHT_EDITOR_MESSAGES: Catalog<WeightEditorMessageKey> = {
  fr: weightEditorFr,
  en: {
    notWeight: 'The current selection is not a weight.',
    missingWeight: 'This weight does not exist in the current architecture.',
    empty: 'Enter a value.',
    nonFinite: 'Non-finite value: the number exceeds the limits.',
    unreadable: 'Unreadable number: use a decimal, for example 0.5.',
    weightsMismatch: 'The displayed weights do not match the current architecture.',
    hint: 'Click a connection, then a cell in the downstream filter canvas, then enter the value.',
  },
};

export function weightEditorMessage(
  locale: Locale,
  key: WeightEditorMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, WEIGHT_EDITOR_MESSAGES, key, params);
}

const weightSelectionFr = {
  linkLabel: 'conv{layer} — entrée {input}, sortie {output}',
  position: '{link}, position ({ky}, {kx})',
} as const;

export type WeightSelectionMessageKey = keyof typeof weightSelectionFr;

export const WEIGHT_SELECTION_MESSAGES: Catalog<WeightSelectionMessageKey> = {
  fr: weightSelectionFr,
  en: {
    linkLabel: 'conv{layer} — input {input}, output {output}',
    position: '{link}, position ({ky}, {kx})',
  },
};

export function weightSelectionMessage(
  locale: Locale,
  key: WeightSelectionMessageKey,
  params?: TranslateParams,
): string {
  return translate(locale, WEIGHT_SELECTION_MESSAGES, key, params);
}
