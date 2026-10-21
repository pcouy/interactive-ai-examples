const page = document.getElementById('deep-blue-demo');
const log = document.querySelector('[data-demo-log]');

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

function outcomeLabel(outcome) {
  if (outcome === 'draw') {
    return 'Partie nulle';
  }
  return `Victoire du joueur ${outcome}`;
}

if (page) {
  page.addEventListener('drl-move-played', (event) => {
    push(`Coup ${JSON.stringify(event.detail.move)} (joueur ${event.detail.player})`);
  });
  page.addEventListener('drl-search-started', (event) => {
    push(`Recherche lancée pour le joueur ${event.detail.player}`);
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
