/**
 * Courbes d'entraînement du composant `drl-mnist-cnn` : historique des époques,
 * extraction des points, domaine et graduations, tracé SVG et libellés.
 *
 * Module pur : ni Lit, ni DOM, ni dépendance. Toutes les fonctions sont
 * testables seules et ne mutent jamais leurs entrées.
 *
 * Un point est ajouté par époque achevée. L'historique est borné à
 * `TRAINING_HISTORY_CAP` points : les points les plus anciens sont évincés, mais
 * le compteur `epochs` continue de croître, de sorte que l'abscisse d'une courbe
 * qui déborde reste monotone et sans trou.
 */

/** Un point de l'historique : une époque achevée et ses deux mesures. */
export interface TrainingHistoryPoint {
  /** Numéro d'époque, à partir de 1. */
  readonly epoch: number;
  /** Perte d'entraînement moyenne de l'époque, ou `null` si non finie. */
  readonly loss: number | null;
  /** Précision de validation de l'époque, ou `null` si non finie. */
  readonly validationAccuracy: number | null;
}

/** Historique des époques, borné à `TRAINING_HISTORY_CAP` points. */
export interface TrainingHistory {
  /** Points conservés, du plus ancien au plus récent. */
  readonly points: readonly TrainingHistoryPoint[];
  /** Nombre d'époques achevées depuis le début, y compris les points évincés. */
  readonly epochs: number;
}

/** Nombre maximal de points conservés par courbe. */
export const TRAINING_HISTORY_CAP = 50;

/** Mesures d'une époque achevées, telles que rapportées par l'entraîneur. */
export interface EpochProgress {
  /** Nombre d'époques achevées depuis le début de l'entraînement. */
  readonly completedEpochs: number;
  /** Perte d'entraînement moyenne de la dernière époque achevée, ou `null`. */
  readonly epochLoss: number | null;
  /** Précision de validation de la dernière époque achevée, ou `null`. */
  readonly validationAccuracy: number | null;
}

/** Point de courbe : abscisse (époque) et ordonnée (mesure). */
export interface ChartPoint {
  readonly x: number;
  readonly y: number;
}

/** Boîte d'un graphique SVG : dimensions totales et remplissages. */
export interface ChartBox {
  readonly width: number;
  readonly height: number;
  readonly padLeft: number;
  readonly padRight: number;
  readonly padTop: number;
  readonly padBottom: number;
}

/** Boîte par défaut des deux courbes du panneau d'entraînement. */
export const CHART_BOX: ChartBox = {
  width: 260,
  height: 90,
  padLeft: 28,
  padRight: 8,
  padTop: 8,
  padBottom: 16,
};

/** Historique vide : aucun point, aucune époque. */
export function emptyHistory(): TrainingHistory {
  return { points: [], epochs: 0 };
}

/**
 * Ajoute un point à l'historique sans modifier l'historique reçu. Quand la
 * longueur dépasse `cap`, le point le plus ancien est évincé ; `epochs` augmente
 * d'une unité à chaque ajout et reste donc monotone après éviction.
 */
export function appendHistory(
  history: TrainingHistory,
  point: TrainingHistoryPoint,
  cap: number = TRAINING_HISTORY_CAP,
): TrainingHistory {
  const kept =
    history.points.length + 1 > cap
      ? history.points.slice(history.points.length + 1 - cap)
      : [...history.points];
  kept.push(point);
  return { points: kept, epochs: history.epochs + 1 };
}

/**
 * Point de courbe pour l'événement d'époque `progress`, ou `null` quand
 * l'événement ne rapporte aucune époque nouvelle (`completedEpochs` inférieur ou
 * égal à `lastEpoch`). `nextEpoch` est l'abscisse du point. Une mesure non
 * finie devient `null` : la courbe ignore alors le point.
 */
export function epochPoint(
  progress: EpochProgress,
  lastEpoch: number,
  nextEpoch: number,
): TrainingHistoryPoint | null {
  if (progress.completedEpochs <= lastEpoch) {
    return null;
  }
  return {
    epoch: nextEpoch,
    loss: Number.isFinite(progress.epochLoss) ? progress.epochLoss : null,
    validationAccuracy: Number.isFinite(progress.validationAccuracy)
      ? progress.validationAccuracy
      : null,
  };
}

/** Points de la courbe de perte : les points sans perte finie sont écartés. */
export function lossPoints(history: TrainingHistory): ChartPoint[] {
  return history.points.flatMap((point) =>
    point.loss !== null && Number.isFinite(point.loss) ? [{ x: point.epoch, y: point.loss }] : [],
  );
}

/** Points de la courbe de précision : les points sans précision finie sont écartés. */
export function accuracyPoints(history: TrainingHistory): ChartPoint[] {
  return history.points.flatMap((point) =>
    point.validationAccuracy !== null && Number.isFinite(point.validationAccuracy)
      ? [{ x: point.epoch, y: point.validationAccuracy }]
      : [],
  );
}

/**
 * Domaine vertical d'une courbe : le minimum et le maximum des ordonnées finies.
 * Sans point, le domaine par défaut va de 0 à 1. Une valeur constante — ou un
 * seul point — reçoit un remplissage de 10 % pour que le tracé reste visible et
 * que le domaine reste strictement croissant.
 */
export function chartDomain(points: readonly ChartPoint[]): {
  readonly min: number;
  readonly max: number;
} {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    if (!Number.isFinite(point.y)) {
      continue;
    }
    if (point.y < min) {
      min = point.y;
    }
    if (point.y > max) {
      max = point.y;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }
  if (min === max) {
    const pad = Math.max(Math.abs(min), 1) * 0.1;
    return { min: min - pad, max: max + pad };
  }
  return { min, max };
}

/** Graduations régulières de `min` à `max`, en `count` valeurs ; `[]` si `count < 2`. */
export function chartTicks(min: number, max: number, count: number): number[] {
  if (!Number.isInteger(count) || count < 2 || !Number.isFinite(min) || !Number.isFinite(max)) {
    return [];
  }
  const ticks: number[] = [];
  for (let index = 0; index < count; index += 1) {
    ticks.push(min + ((max - min) * index) / (count - 1));
  }
  return ticks;
}

/**
 * Graduations ramenées en pixels : `min` se place au bas de la zone de tracé et
 * `max` en haut. `count` vaut 4 par défaut.
 */
export function chartTickPixels(
  min: number,
  max: number,
  box: ChartBox = CHART_BOX,
  count: number = 4,
): ReadonlyArray<{ readonly value: number; readonly y: number }> {
  const plotTop = box.padTop;
  const plotBottom = box.height - box.padBottom;
  const span = max - min;
  return chartTicks(min, max, count).map((value) => ({
    value,
    y: span === 0 ? plotBottom : plotBottom - ((value - min) / span) * (plotBottom - plotTop),
  }));
}

/** Arrondi à deux décimales, avec `-0` normalisé en `0`. */
function roundPixel(value: number): number {
  const rounded = Math.round(value * 100) / 100;
  return Object.is(rounded, -0) ? 0 : rounded;
}

/**
 * Tracé SVG d'une courbe : `M x y` puis un ` L ` par point. Retourne `''` sans
 * point ou avec un domaine invalide. Une valeur non finie interrompt le tracé :
 * le point suivant repart sur un nouveau `M`, ce qui dessine un trou plutôt
 * qu'une ligne absurde. Aucune coordonnée n'est écrite sans être finie.
 */
export function chartPath(
  points: readonly ChartPoint[],
  box: ChartBox,
  yMin: number,
  yMax: number,
): string {
  if (points.length === 0) {
    return '';
  }
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMax <= yMin) {
    return '';
  }
  const plotLeft = box.padLeft;
  const plotRight = box.width - box.padRight;
  const plotTop = box.padTop;
  const plotBottom = box.height - box.padBottom;
  const span = yMax - yMin;
  const xStep = points.length > 1 ? (plotRight - plotLeft) / (points.length - 1) : 0;
  const segments: string[] = [];
  let open = false;
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      open = false;
      continue;
    }
    const x = roundPixel(plotLeft + index * xStep);
    const y = roundPixel(plotBottom - ((point.y - yMin) / span) * (plotBottom - plotTop));
    segments.push(`${open ? 'L' : 'M'} ${x} ${y}`);
    open = true;
  }
  return segments.join(' ');
}

/** Valeur formatée à `digits` décimales, virgule française, `-0` normalisé en `0`. */
export function formatValue(value: number, digits: number): string {
  const rounded = Number(value.toFixed(digits));
  const normalised = Object.is(rounded, -0) ? 0 : rounded;
  return normalised.toFixed(digits).replace('.', ',');
}

/** Libellé accessible d'une courbe : la dernière valeur et le nombre de points. */
export function chartAriaLabel(
  name: string,
  points: readonly ChartPoint[],
  format: (value: number) => string,
): string {
  if (points.length === 0) {
    return `${name} : aucune donnée.`;
  }
  const latest = points[points.length - 1];
  return `${name} : ${points.length} point${points.length > 1 ? 's' : ''}, dernière valeur ${format(latest.y)}.`;
}

/** Courbe complète : points, graduations, tracé, axe et libellé accessible. */
export interface ChartLayout {
  /** Nom de la courbe, repris par le titre et le libellé accessible. */
  readonly name: string;
  /** Boîte du graphique. */
  readonly box: ChartBox;
  /** Points de la courbe. */
  readonly points: readonly ChartPoint[];
  /** Graduations verticales, en valeur et en pixels. */
  readonly ticks: ReadonlyArray<{ readonly value: number; readonly y: number }>;
  /** Tracé SVG, ou `''` sans point. */
  readonly path: string;
  /** Ordonnée de l'axe horizontal, au bas de la zone de tracé. */
  readonly axisY: number;
  /** Vrai avec moins de deux points : le tracé n'est pas encore lisible. */
  readonly empty: boolean;
  /** Libellé accessible du graphique. */
  readonly ariaLabel: string;
}

/**
 * Assemble une courbe pour le rendu : domaine, graduations, tracé et libellé.
 * `empty` vaut vrai avec moins de deux points — une courbe d'un seul point ne
 * montre qu'une position, pas une tendance. `axisY` se place au bas de la zone
 * de tracé.
 */
export function layoutChart(
  name: string,
  points: readonly ChartPoint[],
  box: ChartBox = CHART_BOX,
  format: (value: number) => string = (value) => formatValue(value, 3),
): ChartLayout {
  const { min, max } = chartDomain(points);
  return {
    name,
    box,
    points,
    ticks: chartTickPixels(min, max, box),
    path: chartPath(points, box, min, max),
    axisY: box.height - box.padBottom,
    empty: points.length < 2,
    ariaLabel: chartAriaLabel(name, points, format),
  };
}
