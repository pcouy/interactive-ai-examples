(() => {
  const KEY = 'drl-lang';
  const root = document.documentElement;

  let stored = null;
  try {
    stored = localStorage.getItem(KEY);
  } catch {
    stored = null;
  }

  const detect = () => {
    const languages = navigator.languages;
    const first =
      Array.isArray(languages) && languages.length > 0 ? languages[0] : navigator.language;
    return typeof first === 'string' && first.toLowerCase().startsWith('en') ? 'en' : 'fr';
  };

  const preferred = stored === 'fr' || stored === 'en' ? stored : detect();
  const current = root.lang === 'en' ? 'en' : 'fr';

  if (preferred !== current) {
    const target = preferred === 'en' ? root.dataset.langEn : root.dataset.langFr;
    if (target) {
      const next = new URL(target, location.href);
      if (next.pathname !== location.pathname) {
        location.replace(target + location.search + location.hash);
        return;
      }
    }
  }

  const remember = (event) => {
    if (!(event.target instanceof Element)) return;

    const link = event.target.closest('[data-set-lang]');
    if (!link) return;

    const value = link.getAttribute('data-set-lang');
    if (value !== 'fr' && value !== 'en') return;

    try {
      localStorage.setItem(KEY, value);
    } catch {
      /* Storage can be unavailable in private or sandboxed contexts. */
    }
  };

  document.addEventListener('click', remember);

  // A middle click opens the link in a new tab without firing `click` in some browsers.
  document.addEventListener('auxclick', (event) => {
    if (event.button === 1) {
      remember(event);
    }
  });
})();
