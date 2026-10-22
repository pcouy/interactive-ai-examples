import { describe, expect, it } from 'vitest';
import {
  accuracyPoints,
  appendHistory,
  CHART_BOX,
  type ChartBox,
  type ChartPoint,
  type ChartSeries,
  chartAriaLabel,
  chartDomain,
  chartPath,
  chartTickPixels,
  chartTicks,
  type EpochProgress,
  emptyHistory,
  epochPoint,
  formatValue,
  layoutChart,
  layoutMultiChart,
  lossPoints,
  multiChartAriaLabel,
  TRAINING_HISTORY_CAP,
  type TrainingHistory,
  type TrainingHistoryPoint,
  trainAccuracyPoints,
} from './training-chart.js';

/** Boîte de test : zone de tracé de 85 × 50 pixels. */
const BOX: ChartBox = {
  width: 100,
  height: 60,
  padLeft: 10,
  padRight: 5,
  padTop: 4,
  padBottom: 6,
};

/** Point d'historique complet, avec les trois mesures. */
function point(
  epoch: number,
  loss: number | null,
  accuracy: number | null,
  trainAccuracy: number | null = null,
): TrainingHistoryPoint {
  return { epoch, loss, trainAccuracy, validationAccuracy: accuracy };
}

/** Historique de `count` points, un par époque, tous mesurés. */
function measuredHistory(
  count: number,
  loss: (epoch: number) => number | null = () => 0.5,
): TrainingHistory {
  let history = emptyHistory();
  for (let epoch = 1; epoch <= count; epoch += 1) {
    history = appendHistory(history, point(epoch, loss(epoch), 0.9));
  }
  return history;
}

/** Progression de fin d'époque : les trois mesures finies. */
function epochProgress(
  completedEpochs: number,
  epochLoss = 0.4,
  accuracy = 0.8,
  trainAccuracy: number | null = null,
): EpochProgress {
  return {
    completedEpochs,
    epochLoss,
    epochTrainAccuracy: trainAccuracy,
    validationAccuracy: accuracy,
  };
}

/** Étiquette de formatage simple, sans locale. */
function plainFormat(value: number): string {
  return String(value);
}

describe('emptyHistory', () => {
  it('has no points and zero epochs', () => {
    const history = emptyHistory();
    expect(history.points).toEqual([]);
    expect(history.epochs).toBe(0);
  });
});

describe('appendHistory', () => {
  it('appends the point and increments the epoch counter', () => {
    const first = appendHistory(emptyHistory(), point(1, 0.4, 0.8));
    expect(first.points).toHaveLength(1);
    expect(first.epochs).toBe(1);
    const second = appendHistory(first, point(2, 0.3, 0.85));
    expect(second.points).toHaveLength(2);
    expect(second.epochs).toBe(2);
    expect(second.points[0].epoch).toBe(1);
    expect(second.points[1].epoch).toBe(2);
    // L'historique reçu est remplacé, jamais modifié.
    expect(first.points).toHaveLength(1);
  });

  it('evicts the oldest point at the cap and keeps epochs monotonic', () => {
    const total = TRAINING_HISTORY_CAP + 5;
    let history = emptyHistory();
    for (let epoch = 1; epoch <= total; epoch += 1) {
      history = appendHistory(history, point(epoch, 0.5, 0.5));
    }
    expect(history.points).toHaveLength(TRAINING_HISTORY_CAP);
    // Le compteur continue de croître : les abscisses restent sans trou.
    expect(history.epochs).toBe(total);
    expect(history.points[0].epoch).toBe(total - TRAINING_HISTORY_CAP + 1);
    expect(history.points[history.points.length - 1].epoch).toBe(total);
    expect(history.points.map((item) => item.epoch)).toEqual(
      Array.from(
        { length: TRAINING_HISTORY_CAP },
        (_unused, index) => index + total - TRAINING_HISTORY_CAP + 1,
      ),
    );
  });

  it('honours a custom cap', () => {
    let history = emptyHistory();
    for (let epoch = 1; epoch <= 5; epoch += 1) {
      history = appendHistory(history, point(epoch, 0.5, 0.5), 3);
    }
    expect(history.points.map((item) => item.epoch)).toEqual([3, 4, 5]);
    expect(history.epochs).toBe(5);
  });

  it('does not mutate the given history nor the given point', () => {
    const history = measuredHistory(1);
    const before = history.points[0];
    const added = point(2, 0.3, 0.9);
    const result = appendHistory(history, added);
    expect(history.points).toHaveLength(1);
    expect(history.epochs).toBe(1);
    expect(history.points[0]).toBe(before);
    expect(added).toEqual({
      epoch: 2,
      loss: 0.3,
      trainAccuracy: null,
      validationAccuracy: 0.9,
    });
    expect(result).not.toBe(history);
  });
});

describe('epochPoint', () => {
  it('returns null when the progress reports no new epoch', () => {
    expect(epochPoint(epochProgress(0), 0, 1)).toBeNull();
    expect(epochPoint(epochProgress(1), 1, 2)).toBeNull();
    expect(epochPoint(epochProgress(1), 2, 3)).toBeNull();
  });

  it('uses nextEpoch as the abscissa and keeps finite measures', () => {
    const result = epochPoint(epochProgress(3, 0.25, 0.75, 0.9), 2, 12);
    expect(result).toEqual({
      epoch: 12,
      loss: 0.25,
      trainAccuracy: 0.9,
      validationAccuracy: 0.75,
    });
  });

  it('maps null and non-finite measures to null', () => {
    const notFinite = epochPoint(
      {
        completedEpochs: 1,
        epochLoss: Number.NaN,
        epochTrainAccuracy: Number.POSITIVE_INFINITY,
        validationAccuracy: Number.POSITIVE_INFINITY,
      },
      0,
      1,
    );
    expect(notFinite).toEqual({
      epoch: 1,
      loss: null,
      trainAccuracy: null,
      validationAccuracy: null,
    });
    const missing = epochPoint(
      { completedEpochs: 1, epochLoss: null, epochTrainAccuracy: null, validationAccuracy: null },
      0,
      2,
    );
    expect(missing).toEqual({
      epoch: 2,
      loss: null,
      trainAccuracy: null,
      validationAccuracy: null,
    });
  });
});

describe('lossPoints', () => {
  it('drops points without a finite loss and maps x to the epoch', () => {
    const history: TrainingHistory = {
      points: [
        point(1, 0.4, null),
        point(2, null, null),
        point(3, Number.NaN, null),
        point(4, Number.POSITIVE_INFINITY, null),
        point(5, 0.1, null),
      ],
      epochs: 5,
    };
    expect(lossPoints(history)).toEqual([
      { x: 1, y: 0.4 },
      { x: 5, y: 0.1 },
    ]);
  });
});

describe('accuracyPoints', () => {
  it('returns an empty array when no accuracy is available', () => {
    const history: TrainingHistory = {
      points: [point(1, 0.4, null), point(2, 0.3, Number.NaN)],
      epochs: 2,
    };
    expect(accuracyPoints(history)).toEqual([]);
  });

  it('keeps only the finite accuracies', () => {
    const history: TrainingHistory = {
      points: [point(1, 0.4, null), point(2, 0.3, 0.7), point(3, 0.2, Number.POSITIVE_INFINITY)],
      epochs: 3,
    };
    expect(accuracyPoints(history)).toEqual([{ x: 2, y: 0.7 }]);
  });
});

describe('trainAccuracyPoints', () => {
  it('returns an empty array when no train accuracy is available', () => {
    const history: TrainingHistory = {
      points: [point(1, 0.4, 0.9, null), point(2, 0.3, 0.9, Number.NaN)],
      epochs: 2,
    };
    expect(trainAccuracyPoints(history)).toEqual([]);
  });

  it('keeps only the finite train accuracies and maps x to the epoch', () => {
    const history: TrainingHistory = {
      points: [
        point(1, 0.4, 0.9, null),
        point(2, 0.3, 0.9, 0.7),
        point(3, 0.2, 0.9, Number.POSITIVE_INFINITY),
        point(4, 0.1, 0.9, 0.85),
      ],
      epochs: 4,
    };
    expect(trainAccuracyPoints(history)).toEqual([
      { x: 2, y: 0.7 },
      { x: 4, y: 0.85 },
    ]);
  });
});

describe('chartDomain', () => {
  it('returns the default domain without points', () => {
    expect(chartDomain([])).toEqual({ min: 0, max: 1 });
  });

  it('pads a single point so the domain stays strictly increasing', () => {
    const { min, max } = chartDomain([{ x: 1, y: 0.4 }]);
    expect(min).toBeLessThan(0.4);
    expect(max).toBeGreaterThan(0.4);
    expect(min).toBeLessThan(max);
  });

  it('pads constant values so the domain stays strictly increasing', () => {
    const { min, max } = chartDomain([
      { x: 1, y: 0.5 },
      { x: 2, y: 0.5 },
    ]);
    expect(min).toBeLessThan(0.5);
    expect(max).toBeGreaterThan(0.5);
    expect(min).toBeLessThan(max);
  });

  it('ignores non-finite values', () => {
    const { min, max } = chartDomain([
      { x: 0, y: Number.NaN },
      { x: 1, y: 0.2 },
      { x: 2, y: Number.POSITIVE_INFINITY },
      { x: 3, y: Number.NEGATIVE_INFINITY },
      { x: 4, y: 0.6 },
    ]);
    expect(min).toBe(0.2);
    expect(max).toBe(0.6);
  });

  it('falls back to the default domain when every value is non-finite', () => {
    expect(chartDomain([{ x: 1, y: Number.NaN }])).toEqual({ min: 0, max: 1 });
  });
});

describe('chartTicks', () => {
  it('returns the requested count of ticks from min to max', () => {
    expect(chartTicks(0, 1, 5)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(chartTicks(2, 4, 3)).toEqual([2, 3, 4]);
  });

  it('returns no tick with fewer than two values or non-finite bounds', () => {
    expect(chartTicks(0, 1, 1)).toEqual([]);
    expect(chartTicks(0, 1, 0)).toEqual([]);
    expect(chartTicks(0, 1, -3)).toEqual([]);
    expect(chartTicks(0, 1, 1.5)).toEqual([]);
    expect(chartTicks(0, 1, Number.NaN)).toEqual([]);
    expect(chartTicks(Number.NaN, 1, 4)).toEqual([]);
    expect(chartTicks(0, Number.POSITIVE_INFINITY, 4)).toEqual([]);
  });
});

describe('chartTickPixels', () => {
  it('places min at the bottom of the plot area and max at the top', () => {
    const plotTop = BOX.padTop;
    const plotBottom = BOX.height - BOX.padBottom;
    const ticks = chartTickPixels(0, 1, BOX, 2);
    expect(ticks.map((tick) => tick.value)).toEqual([0, 1]);
    expect(ticks[0].y).toBe(plotBottom);
    expect(ticks[1].y).toBe(plotTop);
    for (const tick of ticks) {
      expect(tick.y).toBeGreaterThanOrEqual(plotTop);
      expect(tick.y).toBeLessThanOrEqual(plotBottom);
    }
  });

  it('collapses every tick to the bottom on a zero span', () => {
    const ticks = chartTickPixels(0.5, 0.5, CHART_BOX, 3);
    expect(ticks).toHaveLength(3);
    for (const tick of ticks) {
      expect(tick.y).toBe(CHART_BOX.height - CHART_BOX.padBottom);
    }
  });
});

describe('chartPath', () => {
  /** Coordonnées d'un tracé : les valeurs qui suivent un `M` ou un `L`. */
  function coordinates(path: string): number[][] {
    return path
      .split(' ')
      .reduce<number[][]>(
        (parts, part) => {
          if (part === 'M' || part === 'L') {
            parts.push([]);
          } else if (part !== '') {
            parts[parts.length - 1].push(Number(part));
          }
          return parts;
        },
        [[]],
      )
      .filter((part) => part.length > 0);
  }

  it('returns an empty path without points or with an invalid domain', () => {
    expect(chartPath([], BOX, 0, 1)).toBe('');
    expect(chartPath([{ x: 1, y: 0.5 }], BOX, 1, 1)).toBe('');
    expect(chartPath([{ x: 1, y: 0.5 }], BOX, 0.8, 0.2)).toBe('');
    expect(chartPath([{ x: 1, y: 0.5 }], BOX, Number.NaN, 1)).toBe('');
    expect(chartPath([{ x: 1, y: 0.5 }], BOX, 0, Number.POSITIVE_INFINITY)).toBe('');
  });

  it('draws a single M for one point', () => {
    const path = chartPath([{ x: 1, y: 0.5 }], BOX, 0, 1);
    expect(path.startsWith('M ')).toBe(true);
    expect(path).not.toContain('L');
    expect(coordinates(path)).toHaveLength(1);
  });

  it('draws an M then an L for two points', () => {
    const path = chartPath(
      [
        { x: 1, y: 0 },
        { x: 2, y: 1 },
      ],
      BOX,
      0,
      1,
    );
    const parts = path.split(' ');
    expect(parts[0]).toBe('M');
    expect(parts[3]).toBe('L');
    expect(parts).toHaveLength(6);
    // Le point le plus haut se place en haut de la zone de tracé.
    expect(Number(parts[5])).toBe(BOX.padTop);
    // Le point le plus bas se place en bas de la zone de tracé.
    expect(Number(parts[2])).toBe(BOX.height - BOX.padBottom);
  });

  it('splits the path on a non-finite value without writing NaN', () => {
    const path = chartPath(
      [
        { x: 1, y: 0 },
        { x: 2, y: 0.2 },
        { x: 3, y: Number.NaN },
        { x: 4, y: 0.8 },
        { x: 5, y: 1 },
      ],
      BOX,
      0,
      1,
    );
    expect(path).not.toContain('NaN');
    expect(path).not.toContain('Infinity');
    const marks = path.split(' ').filter((part) => part === 'M' || part === 'L');
    expect(marks).toEqual(['M', 'L', 'M', 'L']);
    for (const [x, y] of coordinates(path)) {
      expect(Number.isFinite(x)).toBe(true);
      expect(Number.isFinite(y)).toBe(true);
    }
  });

  it('keeps constant values inside the box', () => {
    const path = chartPath(
      [
        { x: 1, y: 0.5 },
        { x: 2, y: 0.5 },
      ],
      BOX,
      0.4,
      0.6,
    );
    expect(path).not.toBe('');
    for (const [, y] of coordinates(path)) {
      expect(y).toBeGreaterThanOrEqual(BOX.padTop);
      expect(y).toBeLessThanOrEqual(BOX.height - BOX.padBottom);
    }
  });

  it('keeps negative values inside the box', () => {
    const path = chartPath(
      [
        { x: 1, y: -2 },
        { x: 2, y: -1 },
        { x: 3, y: -3 },
      ],
      BOX,
      -3,
      -1,
    );
    expect(path).not.toBe('');
    for (const [, y] of coordinates(path)) {
      expect(y).toBeGreaterThanOrEqual(BOX.padTop);
      expect(y).toBeLessThanOrEqual(BOX.height - BOX.padBottom);
    }
  });
});

describe('formatValue', () => {
  it('uses a comma as the decimal separator', () => {
    expect(formatValue(0.5, 2)).toBe('0,50');
    expect(formatValue(1, 3)).toBe('1,000');
    expect(formatValue(123.456, 1)).toBe('123,5');
  });

  it('normalises -0 to 0', () => {
    expect(formatValue(-0, 2)).toBe('0,00');
    expect(formatValue(-0.0001, 2)).toBe('0,00');
  });
});

describe('chartAriaLabel', () => {
  it('states the latest value and the point count', () => {
    const label = chartAriaLabel(
      'Perte',
      [
        { x: 1, y: 0.4 },
        { x: 2, y: 0.3 },
      ],
      plainFormat,
    );
    expect(label).toContain('Perte');
    expect(label).toContain('2 points');
    expect(label).toContain('dernière valeur 0.3');
  });

  it('uses the singular for one point', () => {
    const label = chartAriaLabel('Précision', [{ x: 1, y: 0.9 }], plainFormat);
    expect(label).toContain('1 point,');
    expect(label).not.toContain('1 points');
  });

  it('states that there is no data when empty', () => {
    const label = chartAriaLabel('Perte', [], plainFormat);
    expect(label).toBe('Perte : aucune donnée.');
    expect(label).toContain('aucune donnée');
  });
});

describe('layoutChart', () => {
  it('marks a chart with fewer than two points as empty', () => {
    expect(layoutChart('Perte', []).empty).toBe(true);
    expect(layoutChart('Perte', [{ x: 1, y: 0.4 }]).empty).toBe(true);
    const full = layoutChart('Perte', [
      { x: 1, y: 0.4 },
      { x: 2, y: 0.3 },
    ]);
    expect(full.empty).toBe(false);
  });

  it('places the horizontal axis at the bottom of the plot area', () => {
    const layout = layoutChart('Perte', [
      { x: 1, y: 0.4 },
      { x: 2, y: 0.3 },
    ]);
    expect(layout.axisY).toBe(CHART_BOX.height - CHART_BOX.padBottom);
    expect(layout.name).toBe('Perte');
    expect(layout.box).toBe(CHART_BOX);
    expect(layout.points).toHaveLength(2);
    expect(layout.ticks.length).toBeGreaterThan(0);
    expect(layout.path).not.toBe('');
    expect(layout.ariaLabel).toContain('Perte');
  });

  it('draws nothing for an empty chart', () => {
    const layout = layoutChart('Précision', []);
    expect(layout.path).toBe('');
    expect(layout.ariaLabel).toBe('Précision : aucune donnée.');
  });
});

describe('multiChartAriaLabel', () => {
  /** Courbe de test, avec un formatage simple. */
  function series(name: string, points: ChartPoint[], lineClass = 'line'): ChartSeries {
    return { name, points, lineClass, format: plainFormat };
  }

  it('names every present series and its last value', () => {
    const label = multiChartAriaLabel('Précision', [
      series('Entraînement', [
        { x: 1, y: 0.5 },
        { x: 2, y: 0.9 },
      ]),
      series('Validation', [
        { x: 1, y: 0.4 },
        { x: 2, y: 0.8 },
      ]),
    ]);
    expect(label).toContain('Entraînement');
    expect(label).toContain('Validation');
    expect(label).toContain('2 points');
    expect(label).toContain('dernière valeur 0.9');
    expect(label).toContain('dernière valeur 0.8');
  });

  it('uses the singular for a one-point series', () => {
    const label = multiChartAriaLabel('Précision', [series('Entraînement', [{ x: 1, y: 0.9 }])]);
    expect(label).toContain('1 point,');
    expect(label).not.toContain('1 points');
  });

  it('omits a series without points', () => {
    const label = multiChartAriaLabel('Précision', [
      series('Entraînement', [
        { x: 1, y: 0.5 },
        { x: 2, y: 0.9 },
      ]),
      series('Validation', []),
    ]);
    expect(label).toContain('Entraînement');
    expect(label).not.toContain('Validation');
  });

  it('states that there is no data when every series is empty', () => {
    const label = multiChartAriaLabel('Précision', [
      series('Entraînement', []),
      series('Validation', []),
    ]);
    expect(label).toBe('Précision : aucune donnée.');
  });
});

describe('layoutMultiChart', () => {
  /** Courbe de test, avec un formatage simple. */
  function series(name: string, points: ChartPoint[], lineClass = 'line'): ChartSeries {
    return { name, points, lineClass, format: plainFormat };
  }

  /** Courbe d'entraînement de deux points. */
  const train = series('Entraînement', [
    { x: 1, y: 0.5 },
    { x: 2, y: 0.9 },
  ]);

  it('shares one domain and one tick array across the series', () => {
    const validation = series('Validation', [
      { x: 1, y: 0.4 },
      { x: 2, y: 0.8 },
    ]);
    const layout = layoutMultiChart('Précision', [train, validation], BOX);
    const { min, max } = chartDomain([...train.points, ...validation.points]);
    expect(layout.ticks).toEqual(chartTickPixels(min, max, BOX));
    // Chaque tracé utilise le domaine partagé, donc la même échelle verticale.
    expect(layout.series[0].path).toBe(chartPath(train.points, BOX, min, max));
    expect(layout.series[1].path).toBe(chartPath(validation.points, BOX, min, max));
  });

  it('draws one non-empty path per series', () => {
    const layout = layoutMultiChart(
      'Précision',
      [
        train,
        series('Validation', [
          { x: 1, y: 0.4 },
          { x: 2, y: 0.8 },
        ]),
      ],
      BOX,
    );
    expect(layout.series).toHaveLength(2);
    for (const item of layout.series) {
      expect(item.path.startsWith('M ')).toBe(true);
      expect(item.path).toContain('L');
    }
    expect(layout.name).toBe('Précision');
    expect(layout.box).toBe(BOX);
    expect(layout.axisY).toBe(BOX.height - BOX.padBottom);
  });

  it('still draws a single-point series but marks the chart empty', () => {
    const layout = layoutMultiChart(
      'Précision',
      [series('Entraînement', [{ x: 1, y: 0.5 }]), series('Validation', [{ x: 1, y: 0.4 }])],
      BOX,
    );
    expect(layout.empty).toBe(true);
    for (const item of layout.series) {
      expect(item.path.startsWith('M ')).toBe(true);
      expect(item.path).not.toContain('L');
    }
  });

  it('marks the chart non-empty when one series has two points', () => {
    const layout = layoutMultiChart('Précision', [train, series('Validation', [])], BOX);
    expect(layout.empty).toBe(false);
    // Une série de validation vide laisse tout de même la courbe d'entraînement.
    expect(layout.series[0].path).toContain('L');
    expect(layout.series[1].path).toBe('');
  });
});

describe('chart layout from a training history', () => {
  /** Points de perte d'un historique de trois époques. */
  function threeLossPoints(): ChartPoint[] {
    return lossPoints(measuredHistory(3, (epoch) => 0.4 / epoch));
  }

  it('maps the loss history to a readable curve', () => {
    const layout = layoutChart("Perte d'entraînement", threeLossPoints(), CHART_BOX, plainFormat);
    expect(layout.empty).toBe(false);
    expect(layout.points.map((item) => item.x)).toEqual([1, 2, 3]);
    expect(layout.ariaLabel).toContain('3 points');
    // Chaque point est tracé dans la boîte.
    const path = layout.path;
    expect(path.startsWith('M ')).toBe(true);
    expect(path.split(' ')).toHaveLength(9);
  });
});

describe('English output', () => {
  it('formats values with a decimal point', () => {
    expect(formatValue(0.5, 2, 'en')).toBe('0.50');
    expect(formatValue(1, 3, 'en')).toBe('1.000');
    expect(formatValue(123.456, 1, 'en')).toBe('123.5');
    expect(formatValue(-0, 2, 'en')).toBe('0.00');
  });

  it('localizes single and multi chart aria labels', () => {
    expect(
      chartAriaLabel(
        'Loss',
        [
          { x: 1, y: 0.4 },
          { x: 2, y: 0.3 },
        ],
        plainFormat,
        'en',
      ),
    ).toBe('Loss: 2 points, last value 0.3.');
    expect(chartAriaLabel('Loss', [{ x: 1, y: 0.9 }], plainFormat, 'en')).toBe(
      'Loss: 1 point, last value 0.9.',
    );
    expect(chartAriaLabel('Loss', [], plainFormat, 'en')).toBe('Loss: no data.');
    expect(
      multiChartAriaLabel(
        'Accuracy',
        [
          {
            name: 'Training',
            points: [
              { x: 1, y: 0.5 },
              { x: 2, y: 0.9 },
            ],
            lineClass: 'line',
            format: plainFormat,
          },
          {
            name: 'Validation',
            points: [
              { x: 1, y: 0.4 },
              { x: 2, y: 0.8 },
            ],
            lineClass: 'line',
            format: plainFormat,
          },
        ],
        'en',
      ),
    ).toBe('Accuracy: Training: 2 points, last value 0.9. Validation: 2 points, last value 0.8.');
    expect(multiChartAriaLabel('Accuracy', [], 'en')).toBe('Accuracy: no data.');
  });

  it('localizes the chart layouts', () => {
    expect(layoutChart('Loss', [], CHART_BOX, plainFormat, 'en').ariaLabel).toBe('Loss: no data.');
    const layout = layoutMultiChart(
      'Accuracy',
      [
        {
          name: 'Training',
          points: [{ x: 1, y: 0.5 }],
          lineClass: 'line',
          format: plainFormat,
        },
      ],
      CHART_BOX,
      'en',
    );
    expect(layout.ariaLabel).toBe('Accuracy: Training: 1 point, last value 0.5.');
  });
});
