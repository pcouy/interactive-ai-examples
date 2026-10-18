import {
  DRAW_SCORE,
  type EvalContext,
  type Evaluator,
  type Game,
  LOSS_SCORE,
  type SearchAgent,
  type SearchStats,
  type SearchStatus,
  type TreeRecorder,
  type ValueEvent,
  type VisitEvent,
  type VisitKind,
  WIN_SCORE,
} from '../contracts.js';
import { isEntryUsable, type ProbedEntry, type TranspositionTable } from './transposition.js';

export interface MinimaxOptions<S, M> {
  readonly depth: number;
  readonly evaluator: Evaluator<S, M>;
  readonly prune: boolean;
  readonly recorder: TreeRecorder<S>;
  readonly transposition?: TranspositionTable;
  /** Facteur d'actualisation γ ∈ ]0, 1] appliqué à chaque remontée vers le
   * parent : un coup gagnant plus rapide obtient une valeur plus proche de 1.
   * 1 par défaut (aucun actualisation). */
  readonly discount?: number;
  /** Source aléatoire pour départager les coups à valeur égale : fournie,
   * chaque ex æquo exact est départagé par échantillonnage réservoir (choix
   * final uniforme). Absente, aucun tirage — le premier coup visité à valeur
   * égale reste retenu, pour des recherches reproductibles à l'identique. */
  readonly rng?: () => number;
  /** Politique de consommation de la table : 'previous' est configuré pour
   * l'agent rejoué dans l'arbre — il ignore les entrées du tour courant, déjà
   * déployées à l'écran, et ne consomme que les entrées des tours antérieurs. */
  readonly probeTurns?: 'all' | 'previous';
}

interface Frame<S, M> {
  key: string;
  state: S;
  move: M | null;
  parentKey: string | null;
  depth: number;
  remaining: number;
  maximizing: boolean;
  alpha: number;
  beta: number;
  moves: readonly M[];
  nextChild: number;
  cut: boolean;
  bestScore: number;
  bestMove: M | null;
  /** Nombre de coups ex æquo vus jusqu'ici au meilleur score. */
  ties: number;
}

type NodeResolution =
  | { readonly expanded: true }
  | { readonly expanded: false; readonly score: number; readonly exact: boolean };

export function createMinimaxAgent<S, M>(
  game: Game<S, M>,
  options: MinimaxOptions<S, M>,
): SearchAgent<S, M> {
  const maxDepth = options.depth;
  const evaluator = options.evaluator;
  const prune = options.prune;
  const recorder = options.recorder;
  const transposition = options.transposition ?? null;

  // Facteur d'actualisation par remontée : hors bornes ou non fini, on
  // retombe sur 1 (aucune actualisation) plutôt que de propager un absurde.
  const rawDiscount = options.discount ?? 1;
  const discount =
    Number.isFinite(rawDiscount) && rawDiscount > 0 && rawDiscount <= 1 ? rawDiscount : 1;
  // Source aléatoire du départage des ex æquo (échantillonnage réservoir).
  // Sans rng fourni, aucun tirage : premier coup visité gagne (déterministe,
  // donc reproductible d'une exécution à l'autre).
  const tieBreak = options.rng ?? null;
  // Politique de consommation des entrées : 'previous' refuse les entrées
  // estampillées du tour courant, pour que le rejoué les ré-explore.
  const probeTurns = options.probeTurns ?? 'all';
  // Nombre de tirages par évaluation d'un évaluateur stochastic-échantillonné
  // (indéfini pour un évaluateur déterministe : le cache-feuille reste inerte).
  const rolloutSamples = evaluator.samplesPerEvaluation;

  let iterator: Generator<void, void, void> | null = null;
  let finished = false;
  let rootMove: M | null = null;

  let nodesVisited = 0;
  let nodesPruned = 0;
  let ttHits = 0;
  let ttMisses = 0;
  let leavesEvaluated = 0;
  let steps = 0;
  let keyCounter = 0;

  const emitVisit = (event: VisitEvent<S>): void => {
    nodesVisited += 1;
    recorder.onVisit(event);
  };

  const emitValue = (event: ValueEvent): void => {
    recorder.onValue(event);
  };

  function makeIterator(rootState: S): Generator<void, void, void> {
    const rootPlayer = game.currentPlayer(rootState);
    const ctx: EvalContext<S, M> = { rootPlayer, game };
    const frames: Frame<S, M>[] = [];

    const ttKey = (state: S): string => `${rootPlayer}:${game.stateKey(state)}`;

    // Tour courant de la table au moment du démarrage de la recherche : les
    // entrées à profondeur égale de ce tour sont consommables, pas celles
    // des tours précédents.
    const currentTurn = transposition !== null ? transposition.currentTurn() : 0;

    // Un même état peut être atteint par plusieurs ordres de coups : une clé
    // par chemin (plutôt que par état) évite que les transpositions écrasent
    // les scores des autres occurrences dans l'arbre enregistré.
    const nodeKey = (state: S): string => {
      keyCounter += 1;
      return `${game.stateKey(state)}#${keyCounter}`;
    };

    const createFrame = (
      state: S,
      move: M | null,
      parentKey: string | null,
      depth: number,
      alpha: number,
      beta: number,
    ): Frame<S, M> => ({
      key: nodeKey(state),
      state,
      move,
      parentKey,
      depth,
      remaining: maxDepth - depth,
      maximizing: game.currentPlayer(state) === rootPlayer,
      alpha,
      beta,
      moves: [],
      nextChild: 0,
      cut: false,
      bestScore: 0,
      bestMove: null,
      ties: 0,
    });

    const visitEvent = (frame: Frame<S, M>, kind: VisitKind): VisitEvent<S> => ({
      nodeKey: frame.key,
      parentKey: frame.parentKey,
      move: frame.move,
      state: frame.state,
      depth: frame.depth,
      kind,
    });

    const visitNode = (frame: Frame<S, M>): NodeResolution => {
      const outcome = game.winner(frame.state);
      if (outcome !== null) {
        const score =
          outcome === 'draw' ? DRAW_SCORE : outcome === rootPlayer ? WIN_SCORE : LOSS_SCORE;
        emitVisit(visitEvent(frame, 'terminal'));
        emitValue({ nodeKey: frame.key, score, depth: frame.remaining, exact: true });
        return { expanded: false, score, exact: true };
      }
      frame.moves = game.legalMoves(frame.state);
      // Feuille d'horizon : plus de coups, ou budget de profondeur épuisé.
      // Décidé une seule fois ici, réutilisé par les deux branches ci-dessous.
      const isHorizonLeaf = frame.moves.length === 0 || frame.remaining <= 0;
      // La racine n'est jamais résolue depuis la table : elle explore toujours
      // ses enfants pour choisir le coup. Seules les frames internes sondent.
      if (transposition !== null && frame.parentKey !== null) {
        const entry: ProbedEntry | null = transposition.probe(ttKey(frame.state));
        const turnOk = probeTurns === 'all' || entry === null || entry.turn < currentTurn;
        // Cas spécial cache-feuille : une entrée à profondeur 0 est une
        // moyenne de tirages rollout pour cet état. On refait le même nombre
        // de tirages, on élargit la moyenne et on restocke (tour
        // re-estampillé). La recherche joue avec cette moyenne élargie : le
        // résultat consolidé est meilleur qu'un tirage isolé.
        if (
          entry !== null &&
          turnOk &&
          isHorizonLeaf &&
          rolloutSamples !== undefined &&
          entry.depth === 0
        ) {
          const fresh = evaluator.evaluate(frame.state, ctx);
          leavesEvaluated += 1;
          ttHits += 1;
          // Une entrée profondeur 0 sans samples ne devrait jamais exister ;
          // le garde-fou ?? 0 replongerait alors dans une moyenne brute.
          const priorSamples = entry.samples ?? 0;
          const total = priorSamples + rolloutSamples;
          const merged =
            total > 0 ? (priorSamples * entry.score + rolloutSamples * fresh) / total : fresh;
          transposition.store(ttKey(frame.state), {
            score: merged,
            depth: 0,
            exact: true,
            samples: total,
          });
          // Feuille (pas 'tt-hit') : l'arbre enregistré doit montrer une vraie
          // feuille évaluée, portant ici la moyenne élargie consommée.
          emitVisit(visitEvent(frame, 'leaf'));
          emitValue({ nodeKey: frame.key, score: merged, depth: frame.remaining, exact: true });
          return { expanded: false, score: merged, exact: true };
        }
        if (
          entry !== null &&
          turnOk &&
          isEntryUsable(
            entry,
            frame.remaining,
            frame.alpha,
            frame.beta,
            frame.maximizing,
            currentTurn,
          )
        ) {
          ttHits += 1;
          emitVisit(visitEvent(frame, 'tt-hit'));
          emitValue({
            nodeKey: frame.key,
            score: entry.score,
            depth: frame.remaining,
            exact: entry.exact,
          });
          return { expanded: false, score: entry.score, exact: entry.exact };
        }
        ttMisses += 1;
      }
      if (isHorizonLeaf) {
        const score = evaluator.evaluate(frame.state, ctx);
        leavesEvaluated += 1;
        emitVisit(visitEvent(frame, 'leaf'));
        emitValue({ nodeKey: frame.key, score, depth: frame.remaining, exact: true });
        // Cache-feuille rollout : une feuille d'horizon (budget épuisé) est
        // stockée à profondeur 0 avec son nombre de tirages, pour que les
        // passages suivants sur cet état élargissent la moyenne.
        // Sous-cas moves.length === 0 avec remaining > 0 : PAS de stockage —
        // son score est indépendant du budget et une entrée profondeur 0
        // induirait en erreur des recherches plus profondes ensuite.
        if (transposition !== null && rolloutSamples !== undefined && frame.remaining <= 0) {
          transposition.store(ttKey(frame.state), {
            score,
            depth: 0,
            exact: true,
            samples: rolloutSamples,
          });
        }
        return { expanded: false, score, exact: true };
      }
      frame.bestScore = frame.maximizing ? -Infinity : Infinity;
      emitVisit(visitEvent(frame, 'expand'));
      return { expanded: true };
    };

    const emitPruned = (parent: Frame<S, M>, move: M, childState: S, childDepth: number): void => {
      emitVisit({
        nodeKey: nodeKey(childState),
        parentKey: parent.key,
        move,
        state: childState,
        depth: childDepth,
        kind: 'pruned',
      });
      nodesPruned += 1;
    };

    const accumulate = (frame: Frame<S, M>, move: M, childScore: number, exact: boolean): void => {
      // Actualisation unique par ply de remontée : la valeur vue par le parent
      // est γ × score enfant. Le score BRUT `childScore` reste celui émis dans
      // l'arbre enregistré — seule la comptabilité du frame (bestScore, bornes
      // alpha/beta, condition de coupure) travaille sur la valeur actualisée,
      // uniformément à chaque frame, ce qui préserve la cohérence des fenêtres.
      const value = discount * childScore;
      const improves = frame.maximizing ? value > frame.bestScore : value < frame.bestScore;
      if (improves) {
        frame.bestScore = value;
        frame.bestMove = move;
        frame.ties = 1;
        if (frame.maximizing) {
          if (value > frame.alpha) {
            frame.alpha = value;
          }
        } else if (value < frame.beta) {
          frame.beta = value;
        }
        emitValue({ nodeKey: frame.key, score: childScore, depth: frame.remaining, exact: false });
        if (prune) {
          if (frame.maximizing ? frame.bestScore >= frame.beta : frame.bestScore <= frame.alpha) {
            frame.cut = true;
          }
        }
        return;
      }
      // Ex æquo : échantillonnage réservoir. Le i-ème candidat à valeur égale
      // remplace le coup retenu avec probabilité 1/i, si bien que le choix
      // final est uniforme parmi TOUS les ex æquo, indépendant de l'ordre de
      // visite. La valeur ne change pas : ni bornes, ni coupure à revisiter,
      // et aucun évènement de valeur supplémentaire n'est émis.
      //
      // Deux garde-fous : un enfant coupé (fail-soft) peut renvoyer par
      // artefact d'élagage une valeur tout juste égale au meilleur sans
      // l'atteindre réellement — seul un score exact peut donc départager.
      // Et sans rng fourni, aucun tirage : le premier coup visité gagne.
      if (value !== frame.bestScore || !exact || tieBreak === null) {
        return;
      }
      frame.ties += 1;
      if (tieBreak() < 1 / frame.ties) {
        frame.bestMove = move;
      }
    };

    const completeFrame = (frame: Frame<S, M>): void => {
      frames.pop();
      const exact = !frame.cut;
      emitValue({ nodeKey: frame.key, score: frame.bestScore, depth: frame.remaining, exact });
      if (transposition !== null) {
        transposition.store(ttKey(frame.state), {
          score: frame.bestScore,
          depth: frame.remaining,
          exact,
        });
      }
      if (frame.parentKey === null) {
        rootMove = frame.bestMove;
        return;
      }
      const parent = frames.at(-1);
      if (parent) {
        accumulate(parent, frame.move as M, frame.bestScore, exact);
      }
    };

    const settleUp = (): void => {
      for (;;) {
        const frame = frames.at(-1);
        if (!frame || frame.nextChild < frame.moves.length) {
          return;
        }
        completeFrame(frame);
      }
    };

    function* run(): Generator<void, void, void> {
      const root = createFrame(rootState, null, null, 0, -Infinity, Infinity);
      const rootResolution = visitNode(root);
      if (rootResolution.expanded) {
        frames.push(root);
        yield;
      } else {
        return;
      }
      for (;;) {
        const frame = frames.at(-1);
        if (!frame) {
          break;
        }
        if (frame.cut) {
          const move = frame.moves[frame.nextChild];
          const childState = game.apply(frame.state, move);
          emitPruned(frame, move, childState, frame.depth + 1);
          frame.nextChild += 1;
          settleUp();
          if (frames.length === 0) {
            break;
          }
          yield;
          continue;
        }
        const move = frame.moves[frame.nextChild];
        frame.nextChild += 1;
        const childState = game.apply(frame.state, move);
        const child = createFrame(
          childState,
          move,
          frame.key,
          frame.depth + 1,
          frame.alpha,
          frame.beta,
        );
        const resolution = visitNode(child);
        if (resolution.expanded) {
          frames.push(child);
        } else {
          accumulate(frame, move, resolution.score, resolution.exact);
          settleUp();
        }
        if (frames.length === 0) {
          break;
        }
        yield;
      }
    }

    return run();
  }

  return {
    start(state: S): void {
      iterator?.return();
      iterator = makeIterator(state);
      finished = false;
      rootMove = null;
      keyCounter = 0;
      nodesVisited = 0;
      nodesPruned = 0;
      ttHits = 0;
      ttMisses = 0;
      leavesEvaluated = 0;
      steps = 0;
    },
    step(): SearchStatus {
      if (iterator === null || finished) {
        return 'done';
      }
      steps += 1;
      const result = iterator.next();
      if (result.done) {
        finished = true;
        return 'done';
      }
      return 'running';
    },
    bestMove(): M | null {
      return finished ? rootMove : null;
    },
    stats(): SearchStats {
      return {
        nodesVisited,
        nodesPruned,
        ttHits,
        ttMisses,
        leavesEvaluated,
        elapsedSteps: steps,
      };
    },
  };
}
