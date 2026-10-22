const page = document.getElementById('deep-blue-demo');
const log = document.querySelector('[data-demo-log]');
const isEnglish = document.documentElement.lang === 'en';

const PLAYER_NAMES = {
  'tic-tac-toe': isEnglish ? ['Crosses', 'Circles'] : ['Croix', 'Cercles'],
  connect4: isEnglish ? ['Blues', 'Reds'] : ['Bleus', 'Rouges'],
};

const TEXT = isEnglish
  ? {
      player: (id) => `Player ${id}`,
      square: (index) => `square ${index}`,
      column: (index) => `column ${index}`,
      draw: 'Draw',
      win: (name) => `Win: ${name}`,
      move: (move, name) => `Move ${move} (${name})`,
      search: (name) => `Search started for ${name}`,
      displayRoot: 'Display root: search root',
      displayedNode: (key) => `Displayed node: ${key}`,
      undone: (count) => `${count} move(s) undone`,
      reset: 'Game reset',
    }
  : {
      player: (id) => `Joueur ${id}`,
      square: (index) => `case ${index}`,
      column: (index) => `colonne ${index}`,
      draw: 'Partie nulle',
      win: (name) => `Victoire : ${name}`,
      move: (move, name) => `Coup ${move} (${name})`,
      search: (name) => `Recherche lancée pour ${name}`,
      displayRoot: 'Racine d\u2019affichage : racine de recherche',
      displayedNode: (key) => `Nœud affiché : ${key}`,
      undone: (count) => `${count} coup(s) annulé(s)`,
      reset: 'Partie réinitialisée',
    };

const entries = [];

function push(message) {
  entries.unshift(message);
  if (entries.length > 6) {
    entries.length = 6;
  }
  if (log) {
    log.textContent = entries.join(' · ');
  }
}

function toNumber(value) {
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? value : parsed;
  }
  return value;
}

function playerName(player) {
  const index = toNumber(player);
  const names = page ? PLAYER_NAMES[page.gameKey] : undefined;
  if (names && Number.isInteger(index) && index >= 0 && index < names.length) {
    return names[index];
  }
  if (Number.isInteger(index) && index >= 0) {
    return TEXT.player(index + 1);
  }
  return TEXT.player(player);
}

function moveLabel(move) {
  if (move && typeof move.index === 'number') {
    return TEXT.square(move.index + 1);
  }
  if (move && typeof move.col === 'number') {
    return TEXT.column(move.col + 1);
  }
  return JSON.stringify(move);
}

function outcomeLabel(outcome) {
  if (outcome === 'draw') {
    return TEXT.draw;
  }
  return TEXT.win(playerName(toNumber(outcome)));
}

if (page) {
  page.addEventListener('drl-move-played', (event) => {
    push(TEXT.move(moveLabel(event.detail.move), playerName(event.detail.player)));
  });
  page.addEventListener('drl-search-started', (event) => {
    push(TEXT.search(playerName(event.detail.player)));
  });
  page.addEventListener('drl-root-changed', (event) => {
    push(
      event.detail.nodeKey === null
        ? TEXT.displayRoot
        : TEXT.displayedNode(event.detail.nodeKey),
    );
  });
  page.addEventListener('drl-game-over', (event) => {
    push(outcomeLabel(event.detail.outcome));
  });
  page.addEventListener('drl-move-undone', (event) => {
    push(TEXT.undone(event.detail.plies));
  });
  page.addEventListener('drl-game-reset', () => {
    push(TEXT.reset);
  });
}
