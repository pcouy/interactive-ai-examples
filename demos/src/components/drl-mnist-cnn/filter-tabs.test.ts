import { describe, expect, it } from 'vitest';
import {
  FILTER_TABS,
  filterTabId,
  filterTabLabel,
  filterTabPanelId,
  nextFilterTab,
} from './filter-tabs.js';

describe('FILTER_TABS', () => {
  it('lists filters then examples', () => {
    expect(FILTER_TABS).toEqual(['filters', 'examples']);
  });
});

describe('filterTabId', () => {
  it('returns a stable DOM id per tab', () => {
    expect(filterTabId('filters')).toBe('drl-mnist-tab-filters');
    expect(filterTabId('examples')).toBe('drl-mnist-tab-examples');
  });
});

describe('filterTabPanelId', () => {
  it('returns a stable panel DOM id per tab', () => {
    expect(filterTabPanelId('filters')).toBe('drl-mnist-tabpanel-filters');
    expect(filterTabPanelId('examples')).toBe('drl-mnist-tabpanel-examples');
  });
});

describe('filterTabLabel', () => {
  it('returns the visible label per tab', () => {
    expect(filterTabLabel('filters')).toBe('Filtres de convolution');
    expect(filterTabLabel('examples')).toBe('Exemples');
  });

  it('returns the English label per tab', () => {
    expect(filterTabLabel('filters', 'en')).toBe('Convolution filters');
    expect(filterTabLabel('examples', 'en')).toBe('Examples');
  });
});

describe('nextFilterTab', () => {
  it('moves right and wraps', () => {
    expect(nextFilterTab('filters', 'ArrowRight')).toBe('examples');
    expect(nextFilterTab('examples', 'ArrowRight')).toBe('filters');
  });

  it('moves left and wraps', () => {
    expect(nextFilterTab('examples', 'ArrowLeft')).toBe('filters');
    expect(nextFilterTab('filters', 'ArrowLeft')).toBe('examples');
  });

  it('jumps to the first or last tab with Home and End', () => {
    expect(nextFilterTab('examples', 'Home')).toBe('filters');
    expect(nextFilterTab('filters', 'End')).toBe('examples');
  });

  it('returns null for any other key', () => {
    for (const key of ['Enter', ' ', 'Tab', 'ArrowUp', 'ArrowDown', 'a']) {
      expect(nextFilterTab('filters', key)).toBeNull();
      expect(nextFilterTab('examples', key)).toBeNull();
    }
  });
});
