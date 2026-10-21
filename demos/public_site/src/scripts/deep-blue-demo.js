const page = document.getElementById('deep-blue-demo');
const log = document.querySelector('[data-demo-log]');

const PLAYER_NAMES = {
  'tic-tac-toe': ['Croix', 'Cercles'],
  connect4: ['Bleus', 'Rouges'],
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
    return `Joueur ${index + 1}`;
  }
  return `Joueur ${player}`;
}

function moveLabel(move) {
  if (move && typeof move.index === 'number') {
    return `case ${move.index + 1}`;
  }
  if (move && typeof move.col === 'number') {
    return `colonne ${move.col + 1}`;
  }
  return JSON.stringify(move);
}

function outcomeLabel(outcome) {
  if (outcome === 'draw') {
    return 'Partie nulle';
  }
  return `Victoire : ${playerName(toNumber(outcome))}`;
}

if (page) {
  page.addEventListener('drl-move-played', (event) => {
    push(`Coup ${moveLabel(event.detail.move)} (${playerName(event.detail.player)})`);
  });
  page.addEventListener('drl-search-started', (event) => {
    push(`Recherche lancée pour ${playerName(event.detail.player)}`);
  });
  page.addEventListener('drl-root-changed', (event) => {
    push(
      event.detail.nodeKey === null
        ? 'Racine d\u2019affichage : racine de recherche'
        : `Nœud affiché : ${event.detail.nodeKey}`,
    );
  });
  page.addEventListener('drl-game-over', (event) => {
    push(outcomeLabel(event.detail.outcome));
  });
  page.addEventListener('drl-move-undone', (event) => {
    push(`${event.detail.plies} coup(s) annulé(s)`);
  });
  page.addEventListener('drl-game-reset', () => {
    push('Partie réinitialisée');
  });
}
