import path from 'node:path';

export type Lang = 'fr' | 'en';

export const DEFAULT_LANG: Lang = 'fr';

export type PageId = 'home' | 'deep-blue' | 'frozen-lake';

const fr = {
  siteName: "Jouer pour comprendre l'IA",
  brand: "Comprendre l'IA par l'exemple",
  description:
    "Démonstrations interactives pour comprendre l'apprentissage par renforcement et la recherche adversariale.",
  skipLink: 'Aller au contenu',
  nav: {
    home: 'Accueil',
    deepBlue: 'Deep Blue',
    frozenLake: 'Frozen Lake',
  },
  navLabel: 'Navigation principale',
  languagePickerLabel: 'Choisir la langue',
  footerTagline: "Démonstrations pédagogiques. Tout s'exécute dans le navigateur.",
  footerNavLabel: 'Navigation de pied de page',
  footerLinksLabel: 'Liens externes',
  footerHeading: 'Liens',
  externalSite: 'Mon site',
  externalGithub: 'Dépôt GitHub',
  newTabSuffix: ' (nouvel onglet)',
  themeToLight: 'Passer au thème clair',
  themeToDark: 'Passer au thème sombre',
  backToBlog: '← Retour au blog',
  allDemos: '← Toutes les démos',
  eyebrowDeepBlue: 'Recherche adversariale',
  eyebrowFrozenLake: 'Apprentissage par renforcement',
  home: {
    eyebrow: 'Apprentissage par renforcement et recherche adversariale',
    title: "Comprendre les bases de l'IA, un pas à la fois.",
    titleSuffix: 'démos interactives',
    lede: "Deux démonstrations interactives : la recherche adversariale arborescente de Deep Blue et l'apprentissage par renforcement sur l'environnement Frozen Lake. Tout s'exécute dans le navigateur.",
    deepBlueCta: 'Explorer Deep Blue',
    frozenLakeCta: 'Ouvrir Frozen Lake',
    meta: 'Deux démos · Aucune installation',
    demosTitle: 'Les démos',
    deepBlueBody:
      "Un moteur de recherche qui évalue les coups au tic-tac-toe et au Puissance 4. Réglez la profondeur et l'évaluateur, puis jouez contre l'agent.",
    frozenLakeBody:
      "Un agent Q-learning apprend à traverser un lac gelé. Suivez la table Q, les récompenses et l'exploration, épisode après épisode.",
    cardLink: 'Voir la démo',
    cardLinkDeepBlue: 'Voir la démo : Deep Blue',
    cardLinkFrozenLake: 'Voir la démo : Frozen Lake',
    aboutTitle: 'À propos',
  },
  sliders: {
    learningRate: "Taux d'apprentissage",
    epsilon: 'Epsilon (exploration)',
    discountFactor: "Facteur d'actualisation",
    slipProbability: 'Glissade (probabilité)',
    initialValue: 'Valeur initiale Q',
    candyReward: 'Récompense bonbon',
    replayBatchSize: "Taille de l'échantillon",
  },
};

const en: typeof fr = {
  siteName: 'Play to understand AI',
  brand: 'Understand AI by example',
  description:
    'Interactive demonstrations to understand reinforcement learning and adversarial search.',
  skipLink: 'Skip to content',
  nav: {
    home: 'Home',
    deepBlue: 'Deep Blue',
    frozenLake: 'Frozen Lake',
  },
  navLabel: 'Main navigation',
  languagePickerLabel: 'Choose language',
  footerTagline: 'Educational demonstrations. Everything runs in your browser.',
  footerNavLabel: 'Footer navigation',
  footerLinksLabel: 'External links',
  footerHeading: 'Links',
  externalSite: 'My site',
  externalGithub: 'GitHub repository',
  newTabSuffix: ' (new tab)',
  themeToLight: 'Switch to light theme',
  themeToDark: 'Switch to dark theme',
  backToBlog: '← Back to blog',
  allDemos: '← All demos',
  eyebrowDeepBlue: 'Adversarial search',
  eyebrowFrozenLake: 'Reinforcement learning',
  home: {
    eyebrow: 'Reinforcement learning and adversarial search',
    title: 'Understand the basics of AI, one step at a time.',
    titleSuffix: 'interactive demos',
    lede: "Two interactive demonstrations: Deep Blue's adversarial tree search and reinforcement learning on the Frozen Lake environment. Everything runs in your browser.",
    deepBlueCta: 'Explore Deep Blue',
    frozenLakeCta: 'Open Frozen Lake',
    meta: 'Two demos · No installation',
    demosTitle: 'The demos',
    deepBlueBody:
      'A search engine that evaluates moves in tic-tac-toe and Connect Four. Set the depth and the evaluator, then play against the agent.',
    frozenLakeBody:
      'A Q-learning agent learns to cross a frozen lake. Follow the Q-table, the rewards, and the exploration, episode after episode.',
    cardLink: 'View the demo',
    cardLinkDeepBlue: 'View the demo: Deep Blue',
    cardLinkFrozenLake: 'View the demo: Frozen Lake',
    aboutTitle: 'About',
  },
  sliders: {
    learningRate: 'Learning rate',
    epsilon: 'Epsilon (exploration)',
    discountFactor: 'Discount factor',
    slipProbability: 'Slip (probability)',
    initialValue: 'Initial Q value',
    candyReward: 'Candy reward',
    replayBatchSize: 'Sample size',
  },
};

export const ui = { fr, en };

const ROUTES: Record<Lang, Record<PageId, string>> = {
  fr: {
    home: '/',
    'deep-blue': '/deep-blue',
    'frozen-lake': '/frozen-lake',
  },
  en: {
    home: '/en/',
    'deep-blue': '/en/deep-blue',
    'frozen-lake': '/en/frozen-lake',
  },
};

export function routeFor(lang: Lang, page: PageId): string {
  return ROUTES[lang][page];
}

export function pageFromPath(pathname: string): PageId {
  let page = pathname.split(/[?#]/, 1)[0] ?? '';
  page = page.replace(/\/index\.html$/, '/').replace(/\.html$/, '');
  page = page.replace(/^\/+|\/+$/g, '');

  if (page === 'en') page = '';
  else if (page.startsWith('en/')) page = page.slice(3);

  if (page === 'deep-blue') return 'deep-blue';
  if (page === 'frozen-lake') return 'frozen-lake';
  return 'home';
}

export function outputFileForRoute(route: string): string {
  if (route === '' || route === '/') return 'index.html';
  if (route.endsWith('/')) return `${route.replace(/^\/+/, '')}index.html`;
  return `${route.replace(/^\/+/, '')}.html`;
}

function outputFileForPathname(pathname: string): string {
  const clean = pathname.split(/[?#]/, 1)[0] ?? '';

  if (clean === '' || clean === '/' || clean === '/index' || clean === '/index.html') {
    return 'index.html';
  }
  if (clean === '/en' || clean === '/en/' || clean === '/en/index' || clean === '/en/index.html') {
    return 'en/index.html';
  }

  const withoutLeadingSlash = clean.replace(/^\/+/, '');
  if (withoutLeadingSlash.endsWith('/')) return `${withoutLeadingSlash}index.html`;
  if (withoutLeadingSlash.endsWith('.html')) return withoutLeadingSlash;
  return `${withoutLeadingSlash}.html`;
}

export function relativeHref(currentPathname: string, targetRoute: string): string {
  const currentFile = outputFileForPathname(currentPathname);
  const fromDir = path.posix.dirname(currentFile);
  const targetFile = outputFileForRoute(targetRoute);
  const relative = path.posix.relative(fromDir, targetFile);

  if (relative === '') return './';
  return relative.startsWith('.') ? relative : `./${relative}`;
}

export function absoluteUrl(site: URL, targetRoute: string): string {
  const basePath = site.pathname.replace(/\/+$/, '');
  return new URL(`${basePath}${targetRoute}`, site.origin).href;
}
