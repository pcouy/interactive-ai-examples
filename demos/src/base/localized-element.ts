import type { ComplexAttributeConverter, PropertyValues } from 'lit';
import { LitElement } from 'lit';
import { property } from 'lit/decorators.js';
import { DEFAULT_LOCALE, type Locale, normalizeLocale } from '../i18n/locale.js';

export const localeConverter: ComplexAttributeConverter<Locale> = {
  fromAttribute: (value) => normalizeLocale(value) ?? DEFAULT_LOCALE,
  toAttribute: (value) => value,
};

/** Base class for elements that render localized text. */
export abstract class LocalizedElement extends LitElement {
  @property({ attribute: 'lang', reflect: true, converter: localeConverter })
  locale: Locale = DEFAULT_LOCALE;

  /** Runs after a locale change, once the element has updated. */
  protected onLocaleChange(): void {}

  protected override updated(changed: PropertyValues<this>): void {
    super.updated(changed);
    if (changed.has('locale')) {
      this.onLocaleChange();
    }
  }
}
