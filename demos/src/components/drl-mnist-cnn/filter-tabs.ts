/**
 * Helpers purs des onglets « Filtres de convolution » et « Exemples » du
 * composant `drl-mnist-cnn`.
 *
 * Le module ne dépend ni du DOM ni de Lit : il fournit les identifiants ARIA,
 * les libellés et la navigation clavier des onglets. Le composant pose les
 * attributs `role="tab"`, `aria-selected` et `aria-controls` à partir de ces
 * valeurs.
 */

import type { Locale } from '../../i18n/locale.js';
import { filterTabsMessage } from './messages.js';

/** Onglets du panneau latéral. */
export type FilterTab = 'filters' | 'examples';

/** Onglets dans l'ordre du clavier. */
export const FILTER_TABS: readonly FilterTab[] = ['filters', 'examples'];

/** Identifiant DOM de l'onglet. */
export function filterTabId(tab: FilterTab): string {
  return `drl-mnist-tab-${tab}`;
}

/** Identifiant DOM du panneau contrôlé par l'onglet. */
export function filterTabPanelId(tab: FilterTab): string {
  return `drl-mnist-tabpanel-${tab}`;
}

/** Libellé visible de l'onglet. */
export function filterTabLabel(tab: FilterTab, locale: Locale = 'fr'): string {
  return filterTabsMessage(locale, tab);
}

/**
 * Onglet atteint par une touche de navigation, ou `null` quand la touche ne
 * déplace pas le focus. `ArrowRight` et `ArrowLeft` bouclent, `Home` va au
 * premier onglet et `End` au dernier.
 */
export function nextFilterTab(current: FilterTab, key: string): FilterTab | null {
  const index = FILTER_TABS.indexOf(current);
  switch (key) {
    case 'ArrowRight':
      return FILTER_TABS[(index + 1) % FILTER_TABS.length];
    case 'ArrowLeft':
      return FILTER_TABS[(index - 1 + FILTER_TABS.length) % FILTER_TABS.length];
    case 'Home':
      return FILTER_TABS[0];
    case 'End':
      return FILTER_TABS[FILTER_TABS.length - 1];
    default:
      return null;
  }
}
