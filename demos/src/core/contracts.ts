export type PlayerId = number;

export type GameOutcome = PlayerId | 'draw' | null;

export type Cell = PlayerId | null;

export interface CellCoord {
  readonly row: number;
  readonly col: number;
}

export interface BoardView {
  readonly rows: number;
  readonly cols: number;
  readonly cells: ReadonlyArray<readonly Cell[]>;
  readonly highlight?: ReadonlyArray<CellCoord>;
  readonly clickable?: ReadonlyArray<CellCoord>;
}

export interface Game<S, M> {
  readonly winScore: number;
  initialState(): S;
  legalMoves(state: S): readonly M[];
  apply(state: S, move: M): S;
  undo(state: S, move: M): S;
  currentPlayer(state: S): PlayerId;
  winner(state: S): GameOutcome;
  stateKey(state: S): string;
  boardView(state: S): BoardView;
  /** Nombre de pièces posées sur le plateau, c'est-à-dire le tour intra-partie
   * (une pièce par coup dans les deux jeux). Optionnel : les jeux qui ne
   * l'exposent pas laissent le champ absent des entrées de transposition. */
  pieceCount?(state: S): number;
}

export const WIN_SCORE = 1;
export const LOSS_SCORE = -1;
export const DRAW_SCORE = 0;
export const WIN_SCORE_MARGIN = 0.95;

export interface EvalContext<S, M> {
  readonly rootPlayer: PlayerId;
  readonly game: Game<S, M>;
}

export interface Evaluator<S, M> {
  readonly id: string;
  /** Présent uniquement pour les évaluateurs stochastic-échantillonnés :
   * permet au moteur de savoir combien de tirages produisent une évaluation
   * (pour merger les moyennes dans la table de transposition). Non
   * obligatoire pour les autres évaluateurs. */
  readonly samplesPerEvaluation?: number;
  evaluate(state: S, ctx: EvalContext<S, M>): number;
}

export type SearchStatus = 'running' | 'done';

export interface SearchStats {
  readonly nodesVisited: number;
  readonly nodesPruned: number;
  readonly ttHits: number;
  readonly ttMisses: number;
  readonly leavesEvaluated: number;
  readonly elapsedSteps: number;
  /** Profondeur effective certifiée de la valeur retournée à la racine
   * (`SOLVED_DEPTH` si la position racine est résolue indépendamment de
   * l'horizon). */
  readonly effectiveDepth: number;
}

export interface SearchAgent<S, M> {
  start(state: S): void;
  step(): SearchStatus;
  bestMove(): M | null;
  stats(): SearchStats;
}

export type VisitKind = 'expand' | 'pruned' | 'tt-hit' | 'leaf' | 'terminal';

/** Fenêtre alpha-bêta effective d'un nœud au moment de l'évènement. */
export interface SearchWindow {
  readonly alpha: number;
  readonly beta: number;
}

export interface CutInfo {
  /** Borne franchie : beta au niveau maximisant, alpha au niveau minimisant. */
  readonly bound: 'alpha' | 'beta';
  /** true quand strictPruning est actif : comparaison > / < au lieu de >= / <=. */
  readonly strict: boolean;
  /** Meilleur score du nœud au moment de la coupure (espace actualisé). */
  readonly bestScore: number;
  /** Valeur de la borne franchie. */
  readonly limit: number;
  /** Clé du fils qui a établi le meilleur score ayant déclenché la coupure. */
  readonly triggerKey: string | null;
}

export interface VisitEvent<S> {
  readonly nodeKey: string;
  readonly parentKey: string | null;
  readonly move: unknown;
  readonly state: S;
  readonly depth: number;
  readonly kind: VisitKind;
  readonly window?: SearchWindow;
  readonly cut?: CutInfo;
  /** Profondeur certifiée de l'entrée de table consommée (tt-hit seulement). */
  readonly entryDepth?: number;
}

export interface ValueEvent {
  readonly nodeKey: string;
  readonly score: number;
  readonly depth: number;
  readonly exact: boolean;
  readonly window?: SearchWindow;
  readonly cut?: CutInfo;
  /** Profondeur certifiée de l'entrée de table consommée (tt-hit seulement). */
  readonly entryDepth?: number;
}

export interface TreeRecorder<S> {
  onVisit(event: VisitEvent<S>): void;
  onValue(event: ValueEvent): void;
}

export const noopRecorder: TreeRecorder<unknown> = {
  onVisit: () => undefined,
  onValue: () => undefined,
};

export type PlayerKind = 'human' | 'agent';

export interface PlayerConfig<S, M> {
  readonly kind: PlayerKind;
  readonly agent?: SearchAgent<S, M>;
  readonly stepBudget?: number;
}
