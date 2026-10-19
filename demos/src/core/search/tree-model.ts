import type {
  CutInfo,
  SearchWindow,
  TreeRecorder,
  ValueEvent,
  VisitEvent,
  VisitKind,
} from '../contracts.js';

export interface TreeNode<S> {
  readonly nodeKey: string;
  readonly parentKey: string | null;
  move: unknown;
  state: S;
  depth: number;
  kind: VisitKind;
  score: number | null;
  exact: boolean;
  order: number;
  children: string[];
  /** Fenêtre alpha-bêta effective au dernier évènement du nœud. */
  window?: SearchWindow;
  /** Coupure alpha-bêta, présente uniquement sur un nœud coupé. */
  cut?: CutInfo;
}

const RICHER_KIND: Record<VisitKind, number> = {
  expand: 2,
  pruned: 0,
  'tt-hit': 1,
  leaf: 1,
  terminal: 2,
};

export class TreeModel<S> {
  private nodes = new Map<string, TreeNode<S>>();

  private rootKey: string | null = null;

  private counter = 0;

  private lastKey: string | null = null;

  insert(event: VisitEvent<S>): void {
    const existing = this.nodes.get(event.nodeKey);
    if (existing) {
      const sticky = existing.kind === 'pruned' || existing.kind === 'tt-hit';
      if (!sticky && RICHER_KIND[event.kind] > RICHER_KIND[existing.kind]) {
        existing.kind = event.kind;
      }
      // Une re-visite rafraîchit la fenêtre observée sans effacer une
      // information absente de l'évènement.
      if (event.window) {
        existing.window = event.window;
      }
      if (event.cut) {
        existing.cut = event.cut;
      }
      this.lastKey = event.nodeKey;
      return;
    }
    if (event.parentKey !== null) {
      const parent = this.nodes.get(event.parentKey);
      if (!parent) {
        return;
      }
      if (parent.kind === 'pruned' || parent.kind === 'tt-hit') {
        return;
      }
    }
    const node: TreeNode<S> = {
      nodeKey: event.nodeKey,
      parentKey: event.parentKey,
      move: event.move,
      state: event.state,
      depth: event.depth,
      kind: event.kind,
      score: null,
      exact: false,
      order: this.counter,
      children: [],
      window: event.window,
      cut: event.cut,
    };
    this.counter += 1;
    this.nodes.set(event.nodeKey, node);
    if (event.parentKey !== null) {
      const parent = this.nodes.get(event.parentKey);
      if (parent) {
        parent.children.push(event.nodeKey);
      }
    } else if (this.rootKey === null) {
      this.rootKey = event.nodeKey;
    }
    this.lastKey = event.nodeKey;
  }

  applyValue(event: ValueEvent): void {
    let node = this.nodes.get(event.nodeKey);
    if (!node) {
      node = {
        nodeKey: event.nodeKey,
        parentKey: null,
        move: null,
        state: undefined as S,
        depth: event.depth,
        kind: 'leaf',
        score: null,
        exact: false,
        order: this.counter,
        children: [],
        window: event.window,
        cut: event.cut,
      };
      this.counter += 1;
      this.nodes.set(event.nodeKey, node);
    } else {
      if (event.window) {
        node.window = event.window;
      }
      if (event.cut) {
        node.cut = event.cut;
      }
    }
    // Une valeur exacte prouve qu'aucune coupure ne biaise ce nœud.
    if (event.exact) {
      node.cut = undefined;
    }
    node.score = event.score;
    node.exact = event.exact;
  }

  root(): TreeNode<S> | null {
    if (this.rootKey === null) {
      return null;
    }
    return this.nodes.get(this.rootKey) ?? null;
  }

  size(): number {
    return this.nodes.size;
  }

  node(key: string): TreeNode<S> | null {
    return this.nodes.get(key) ?? null;
  }

  children(key: string): TreeNode<S>[] {
    const node = this.nodes.get(key);
    if (!node) {
      return [];
    }
    return node.children
      .map((childKey) => this.nodes.get(childKey))
      .filter((child): child is TreeNode<S> => child !== undefined);
  }

  has(key: string): boolean {
    return this.nodes.has(key);
  }

  resolved(key: string): boolean {
    const node = this.nodes.get(key);
    return node?.exact ?? false;
  }

  lastVisited(): TreeNode<S> | null {
    if (this.lastKey === null) {
      return null;
    }
    return this.nodes.get(this.lastKey) ?? null;
  }

  clear(): void {
    this.nodes.clear();
    this.rootKey = null;
    this.counter = 0;
    this.lastKey = null;
  }
}

export function createModelRecorder<S>(model: TreeModel<S>): TreeRecorder<S> {
  return {
    onVisit: (event: VisitEvent<S>) => model.insert(event),
    onValue: (event: ValueEvent) => model.applyValue(event),
  };
}
