import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const script = readFileSync(new URL('../src/scripts/deep-blue-demo.js', import.meta.url), 'utf8');

const EN_LABELS = {
  logMove: 'Move {move} ({player})',
  logSearchStarted: 'Search started for {player}',
  logDisplayRoot: 'Display root: search root',
  logDisplayedNode: 'Displayed node: {key}',
  logUndone: '{count} move(s) undone',
  logGameReset: 'Game reset',
  player: 'Player {number}',
  square: 'square {number}',
  column: 'column {number}',
  draw: 'Draw',
  win: 'Win: {player}',
  crosses: 'Crosses',
  circles: 'Circles',
  blues: 'Blues',
  reds: 'Reds',
};

const FR_LABELS = {
  logMove: 'Coup {move} ({player})',
  logSearchStarted: 'Recherche lancée pour {player}',
  logDisplayRoot: 'Racine d’affichage : racine de recherche',
  logDisplayedNode: 'Nœud affiché : {key}',
  logUndone: '{count} coup(s) annulé(s)',
  logGameReset: 'Partie réinitialisée',
  player: 'Joueur {number}',
  square: 'case {number}',
  column: 'colonne {number}',
  draw: 'Partie nulle',
  win: 'Victoire : {player}',
  crosses: 'Croix',
  circles: 'Cercles',
  blues: 'Bleus',
  reds: 'Rouges',
};

/** Runs the log glue script with a fake page and log element. */
function createPage({ labels = EN_LABELS, gameKey = 'connect4' } = {}) {
  const listeners = new Map();
  const log = {
    dataset: { labels: JSON.stringify(labels) },
    textContent: '',
  };
  const page = {
    gameKey,
    addEventListener(type, listener) {
      listeners.set(type, listener);
    },
  };

  const context = vm.createContext({
    document: {
      getElementById() {
        return page;
      },
      querySelector() {
        return log;
      },
    },
  });

  vm.runInContext(script, context);

  const dispatch = (type, detail) => {
    const listener = listeners.get(type);
    assert.ok(listener, `the script must listen for ${type}`);
    listener({ detail });
  };

  return { dispatch, log };
}

test('an English Connect Four move formats the column and the player', () => {
  const page = createPage();
  page.dispatch('drl-move-played', { move: { col: 2 }, player: 1 });
  assert.equal(page.log.textContent, 'Move column 3 (Reds)');
});

test('an English search names the player', () => {
  const page = createPage();
  page.dispatch('drl-search-started', { player: 0 });
  assert.equal(page.log.textContent, 'Search started for Blues');
});

test('the display root and a displayed node use distinct labels', () => {
  const root = createPage();
  root.dispatch('drl-root-changed', { nodeKey: null });
  assert.equal(root.log.textContent, 'Display root: search root');

  const node = createPage();
  node.dispatch('drl-root-changed', { nodeKey: 'abc' });
  assert.equal(node.log.textContent, 'Displayed node: abc');
});

test('draw and win outcomes use the shared game labels', () => {
  const draw = createPage();
  draw.dispatch('drl-game-over', { outcome: 'draw' });
  assert.equal(draw.log.textContent, 'Draw');

  const win = createPage();
  win.dispatch('drl-game-over', { outcome: 1 });
  assert.equal(win.log.textContent, 'Win: Reds');
});

test('undo and reset use the log labels', () => {
  const undo = createPage();
  undo.dispatch('drl-move-undone', { plies: 2 });
  assert.equal(undo.log.textContent, '2 move(s) undone');

  const reset = createPage();
  reset.dispatch('drl-game-reset', {});
  assert.equal(reset.log.textContent, 'Game reset');
});

test('a French tic-tac-toe move formats the square and the player', () => {
  const page = createPage({ labels: FR_LABELS, gameKey: 'tic-tac-toe' });
  page.dispatch('drl-move-played', { move: { index: 4 }, player: 0 });
  assert.equal(page.log.textContent, 'Coup case 5 (Croix)');
});

test('missing labels fall back to the key without crashing', () => {
  const page = createPage({ labels: {} });
  page.dispatch('drl-move-played', { move: { index: 4 }, player: 0 });
  assert.equal(page.log.textContent, 'logMove');
});
