import { describe, expect, it } from 'vitest';
import { GradientDescent } from './drl-gradient-descent.js';
import { gradientDescentMessage } from './messages.js';

/** Creates a component without the Lit update loop: the node bench has no DOM. */
function createGradientDescent(): GradientDescent {
  const element = new GradientDescent();
  (element as unknown as { performUpdate(): void }).performUpdate = () => {};
  return element;
}

/** Flattens a Lit `TemplateResult` into plain text for assertions. */
function renderTemplateTextFlat(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }
  if (Array.isArray(value)) {
    return value.map(renderTemplateTextFlat).join('');
  }
  if (typeof value === 'object' && 'strings' in value && 'values' in value) {
    const template = value as { strings: readonly string[]; values: readonly unknown[] };
    let out = '';
    for (let index = 0; index < template.strings.length; index += 1) {
      out += template.strings[index];
      if (index < template.values.length) {
        out += renderTemplateTextFlat(template.values[index]);
      }
    }
    return out.replace(/\s+/g, ' ').trim();
  }
  return String(value).replace(/\s+/g, ' ').trim();
}

function headerText(element: GradientDescent): string {
  return renderTemplateTextFlat((element as unknown as { renderHeader(): unknown }).renderHeader());
}

function contentText(element: GradientDescent): string {
  return renderTemplateTextFlat(
    (element as unknown as { renderContent(): unknown }).renderContent(),
  );
}

function statusLine(element: GradientDescent): string {
  return (element as unknown as { statusLine(): string }).statusLine();
}

describe('GradientDescent locale', () => {
  it('renders the French labels by default', () => {
    const element = createGradientDescent();
    const header = headerText(element);
    const content = contentText(element);

    expect(header).toContain('Prêt — « Un pas » ou « Une batch »');
    expect(header).toContain('Époque');
    expect(header).toContain('Position (a, b)');
    expect(header).toContain('Contrôles de la descente de gradient');
    expect(content).toContain(
      'Glissez pour tourner, clic molette + glisser pour déplacer, molette pour zoomer.',
    );
  });

  it('renders the English labels when the locale is en', () => {
    const element = createGradientDescent();
    element.locale = 'en';
    const header = headerText(element);
    const content = contentText(element);

    expect(header).toContain('Ready — “One step” or “One batch”');
    expect(header).toContain('One step');
    expect(header).toContain('One batch');
    expect(header).toContain('Epoch');
    expect(header).toContain('Gradient descent controls');
    expect(content).toContain('3D loss surface and batch gradient descent');
    expect(content).toContain('Drag to rotate, middle-click and drag to pan, scroll to zoom.');
  });

  it('uses a decimal comma in French and a decimal point in English', () => {
    const element = createGradientDescent();
    const internals = element as unknown as {
      phase: string;
      exampleNumber: number;
      currentExample: { x1: number; x2: number; t: number } | null;
    };
    internals.phase = 'present';
    internals.exampleNumber = 2;
    internals.currentExample = { x1: 1.5, x2: -0.25, t: 0.75 };

    expect(statusLine(element)).toContain('(1,50 ; -0,25) → t = 0,75');
    expect(statusLine(element)).toContain('Exemple 2/');

    element.locale = 'en';
    expect(statusLine(element)).toContain('(1.50, -0.25) → t = 0.75');
    expect(statusLine(element)).toContain('Example 2/');
  });

  it('keeps the French catalog strings stable', () => {
    expect(gradientDescentMessage('fr', 'phaseFade')).toBe('Archivage');
    expect(gradientDescentMessage('fr', 'statusMove')).toBe(
      'La balle descend : position -= learning rate × gradient de batch',
    );
    expect(gradientDescentMessage('fr', 'hint')).toBe(
      'Glissez pour tourner, clic molette + glisser pour déplacer, molette pour zoomer.',
    );
    expect(gradientDescentMessage('en', 'phaseFade')).toBe('Archive');
  });
});
