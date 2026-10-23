const page = document.getElementById('deep-blue-demo');
const log = document.querySelector('[data-demo-log]');

// The page embeds the log catalog in the data-labels attribute.
let labels = {};
try {
  labels = log?.dataset.labels ? JSON.parse(log.dataset.labels) : {};
} catch {
  labels = {};
}

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

function formatLabel(key, params = {}) {
  const label = labels[key];
  if (typeof label !== 'string') {
    return key;
  }
  return label.replace(/\{([^{}]+)\}/g, (match, name) =>
    Object.hasOwn(params, name) ? String(params[name]) : match,
  );
}

function playerName(player) {
  const index = toNumber(player);
  const names =
    page?.gameKey === 'connect4'
      ? [labels.blues, labels.reds]
      : page?.gameKey === 'tic-tac-toe'
        ? [labels.crosses, labels.circles]
        : undefined;
  if (names && Number.isInteger(index) && index >= 0 && index < names.length) {
    const name = names[index];
    if (typeof name === 'string') {
      return name;
    }
  }
  if (Number.isInteger(index) && index >= 0) {
    return formatLabel('player', { number: index + 1 });
  }
  return formatLabel('player', { number: player });
}

function moveLabel(move) {
  if (move && typeof move.index === 'number') {
    return formatLabel('square', { number: move.index + 1 });
  }
  if (move && typeof move.col === 'number') {
    return formatLabel('column', { number: move.col + 1 });
  }
  return JSON.stringify(move);
}

function outcomeLabel(outcome) {
  if (outcome === 'draw') {
    return formatLabel('draw');
  }
  return formatLabel('win', { player: playerName(toNumber(outcome)) });
}

if (page) {
  page.addEventListener('drl-move-played', (event) => {
    push(
      formatLabel('logMove', {
        move: moveLabel(event.detail.move),
        player: playerName(event.detail.player),
      }),
    );
  });
  page.addEventListener('drl-search-started', (event) => {
    push(formatLabel('logSearchStarted', { player: playerName(event.detail.player) }));
  });
  page.addEventListener('drl-root-changed', (event) => {
    push(
      event.detail.nodeKey === null
        ? formatLabel('logDisplayRoot')
        : formatLabel('logDisplayedNode', { key: event.detail.nodeKey }),
    );
  });
  page.addEventListener('drl-game-over', (event) => {
    push(outcomeLabel(event.detail.outcome));
  });
  page.addEventListener('drl-move-undone', (event) => {
    push(formatLabel('logUndone', { count: event.detail.plies }));
  });
  page.addEventListener('drl-game-reset', () => {
    push(formatLabel('logGameReset'));
  });
}
