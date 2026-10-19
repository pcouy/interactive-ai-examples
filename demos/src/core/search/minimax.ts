import {
  type CutInfo,
  DRAW_SCORE,
  type EvalContext,
  type Evaluator,
  type Game,
  LOSS_SCORE,
  type SearchAgent,
  type SearchStats,
  type SearchStatus,
  type SearchWindow,
  type TreeRecorder,
  type ValueEvent,
  type VisitEvent,
  type VisitKind,
  WIN_SCORE,
} from '../contracts.js';
import {
  isEntryUsable,
  type ProbedEntry,
  SOLVED_DEPTH,
  type TranspositionTable,
} from './transposition.js';

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
   * final uniforme). Absente, le tirage par défaut dépend du mode : minimax
   * pur (`prune: false`) et élagage strict (`strictPruning: true`) tirent sur
   * Math.random, car leurs ex æquo sont authentiques ; l'alpha-bêta classique
   * ne tire pas (premier coup visité retenu), pour rester reproductible à
   * l'identique quand on compare deux arbres. */
  readonly rng?: () => number;
  /** Politique de consommation de la table : 'previous' est configuré pour
   * l'agent rejoué dans l'arbre — il ignore les entrées du tour courant, déjà
   * déployées à l'écran, et ne consomme que les entrées des tours antérieurs. */
  readonly probeTurns?: 'all' | 'previous';
  /** Élagage strict : la coupure n'a lieu que sur un dépassement STRICT de la
   * borne (max : > β, min : < α) et les coups à valeur exactement égale sont
   * départagés au hasard. Sans `rng` fourni, le départage utilise Math.random.
   * false (défaut) : coupure sur >= / <=. */
  readonly strictPruning?: boolean;
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
  /** Profondeur certifiée minimale parmi les enfants résolus : chaque enfant
   * résolu (borné ou non) contribue sa propre profondeur certifiée. */
  minChildDepth: number;
  /** Clé du fils qui a établi le meilleur score courant : la coupure la
   * rapporte pour que l'arbre enregistré sache quelle branche l'a déclenchée. */
  triggerKey: string | null;
  /** Détail de la coupure, figé au moment où elle est décidée. */
  cutInfo: CutInfo | null;
}

type NodeResolution =
  | { readonly expanded: true }
  | {
      readonly expanded: false;
      readonly score: number;
      readonly exact: boolean;
      readonly depth: number;
    };

export function createMinimaxAgent<S, M>(
  game: Game<S, M>,
  options: MinimaxOptions<S, M>,
): SearchAgent<S, M> {
  const maxDepth = options.depth;
  const evaluator = options.evaluator;
  const prune = options.prune;
  const strictPruning = options.strictPruning ?? false;
  const recorder = options.recorder;
  const transposition = options.transposition ?? null;

  // Facteur d'actualisation par remontée : hors bornes ou non fini, on
  // retombe sur 1 (aucune actualisation) plutôt que de propager un absurde.
  const rawDiscount = options.discount ?? 1;
  const discount =
    Number.isFinite(rawDiscount) && rawDiscount > 0 && rawDiscount <= 1 ? rawDiscount : 1;
  // Source aléatoire du départage des ex æquo (échantillonnage réservoir).
  // Sans rng fourni, le tirage par défaut dépend du mode : minimax pur
  // (`prune: false`) et élagage strict n'ont que des ex æquo authentiques
  // (aucun enfant borné), donc tirer uniformément est le comportement
  // honnête. L'alpha-bêta classique, lui, ne tire pas : ses ex æquo peuvent
  // être un artefact de fail-soft, et l'absence de tirage rend deux arbres
  // comparables à l'identique.
  const tieBreak = options.rng ?? (prune === false || strictPruning ? () => Math.random() : null);
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
  let effectiveDepth = 0;

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

    // Clé de table partagée entre tous les joueurs racine : `stateKey` encode
    // déjà le joueur au trait, les valeurs stockées étant intrinsèques à l'état
    // (point de vue du joueur qui doit jouer), une entrée reste donc valable
    // quel que soit le joueur racine de la recherche qui la sonde.
    const ttKey = (state: S): string => game.stateKey(state);

    // Tour courant de la table au moment du démarrage de la recherche : les
    // entrées à profondeur ÉGALE de ce tour sont consommables, pas celles des
    // tours précédents (qui ne peuvent fournir que du strictement plus profond).
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
      minChildDepth: SOLVED_DEPTH,
      triggerKey: null,
      cutInfo: null,
    });

    const windowOf = (frame: Frame<S, M>): SearchWindow => ({
      alpha: frame.alpha,
      beta: frame.beta,
    });

    const visitEvent = (frame: Frame<S, M>, kind: VisitKind): VisitEvent<S> => ({
      nodeKey: frame.key,
      parentKey: frame.parentKey,
      move: frame.move,
      state: frame.state,
      depth: frame.depth,
      kind,
      window: windowOf(frame),
    });

    const visitNode = (frame: Frame<S, M>): NodeResolution => {
      const outcome = game.winner(frame.state);
      if (outcome !== null) {
        const score =
          outcome === 'draw' ? DRAW_SCORE : outcome === rootPlayer ? WIN_SCORE : LOSS_SCORE;
        emitVisit(visitEvent(frame, 'terminal'));
        emitValue({
          nodeKey: frame.key,
          score,
          depth: frame.remaining,
          exact: true,
          window: windowOf(frame),
        });
        return { expanded: false, score, exact: true, depth: SOLVED_DEPTH };
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
          // La moyenne stockée vit du point de vue du joueur au trait : le
          // tirage frais (relatif à la racine) y est ramené par négation quand
          // le nœud n'est pas maximisant. Une entrée profondeur 0 sans samples
          // ne devrait jamais exister ; le garde-fou ?? 0 replongerait alors
          // dans une moyenne brute.
          const freshStored = frame.maximizing ? fresh : -fresh;
          const priorSamples = entry.samples ?? 0;
          const total = priorSamples + rolloutSamples;
          const mergedStored =
            total > 0
              ? (priorSamples * entry.score + rolloutSamples * freshStored) / total
              : freshStored;
          transposition.store(ttKey(frame.state), {
            score: mergedStored,
            depth: 0,
            exact: true,
            samples: total,
            // Le tour intra-partie n'est renseigné que si le jeu l'expose : un
            // spread garde le champ absent chez les jeux qui l'ignorent.
            ...(game.pieceCount === undefined ? {} : { pieces: game.pieceCount(frame.state) }),
          });
          // Feuille (pas 'tt-hit') : l'arbre enregistré doit montrer une vraie
          // feuille évaluée, portant ici la moyenne élargie consommée.
          const merged = frame.maximizing ? mergedStored : -mergedStored;
          emitVisit(visitEvent(frame, 'leaf'));
          emitValue({
            nodeKey: frame.key,
            score: merged,
            depth: frame.remaining,
            exact: true,
            window: windowOf(frame),
          });
          return { expanded: false, score: merged, exact: true, depth: 0 };
        }
        if (entry !== null && turnOk) {
          // Valeur ramenée du point de vue du joueur au trait vers celui de la
          // racine courante (négation exacte des évaluateurs antisymétriques).
          const visibleScore = frame.maximizing ? entry.score : -entry.score;
          const visibleEntry: ProbedEntry = { ...entry, score: visibleScore };
          if (
            isEntryUsable(
              visibleEntry,
              frame.remaining,
              frame.alpha,
              frame.beta,
              frame.maximizing,
              currentTurn,
              strictPruning,
            )
          ) {
            ttHits += 1;
            emitVisit(visitEvent(frame, 'tt-hit'));
            emitValue({
              nodeKey: frame.key,
              score: visibleScore,
              depth: frame.remaining,
              exact: entry.exact,
              window: windowOf(frame),
            });
            return { expanded: false, score: visibleScore, exact: entry.exact, depth: entry.depth };
          }
        }
        ttMisses += 1;
      }
      if (isHorizonLeaf) {
        const score = evaluator.evaluate(frame.state, ctx);
        leavesEvaluated += 1;
        emitVisit(visitEvent(frame, 'leaf'));
        emitValue({
          nodeKey: frame.key,
          score,
          depth: frame.remaining,
          exact: true,
          window: windowOf(frame),
        });
        // Une position sans coup légal atteinte AVANT l'horizon est résolue
        // indépendamment du budget : sa profondeur certifiée est maximale.
        const solved = frame.moves.length === 0 && frame.remaining > 0;
        // Cache-feuille rollout : une feuille d'horizon (budget épuisé) est
        // stockée à profondeur 0 avec son nombre de tirages, pour que les
        // passages suivants sur cet état élargissent la moyenne. Le score est
        // ramené du point de vue de la racine à celui du joueur au trait.
        // Sous-cas moves.length === 0 avec remaining > 0 : PAS de stockage —
        // son score est indépendant du budget et une entrée profondeur 0
        // induirait en erreur des recherches plus profondes ensuite.
        if (transposition !== null && rolloutSamples !== undefined && frame.remaining <= 0) {
          transposition.store(ttKey(frame.state), {
            score: frame.maximizing ? score : -score,
            depth: 0,
            exact: true,
            samples: rolloutSamples,
            ...(game.pieceCount === undefined ? {} : { pieces: game.pieceCount(frame.state) }),
          });
        }
        return { expanded: false, score, exact: true, depth: solved ? SOLVED_DEPTH : 0 };
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
        window: windowOf(parent),
        ...(parent.cutInfo === null ? {} : { cut: parent.cutInfo }),
      });
      nodesPruned += 1;
    };

    const accumulate = (
      frame: Frame<S, M>,
      move: M,
      childKey: string,
      childScore: number,
      exact: boolean,
      childDepth: number,
    ): void => {
      // Profondeur certifiée du frame : le minimum sur TOUS les enfants résolus
      // (même ceux qui n'améliorent pas le meilleur score). Un enfant borné
      // (coupé) certifie lui aussi sa profondeur : sa valeur reste valable
      // jusqu'à ce point, donc il contribue comme n'importe quel enfant résolu.
      frame.minChildDepth = Math.min(frame.minChildDepth, childDepth);
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
        frame.triggerKey = childKey;
        frame.ties = 1;
        if (frame.maximizing) {
          if (value > frame.alpha) {
            frame.alpha = value;
          }
        } else if (value < frame.beta) {
          frame.beta = value;
        }
        emitValue({
          nodeKey: frame.key,
          score: childScore,
          depth: frame.remaining,
          exact: false,
          window: windowOf(frame),
        });
        if (prune) {
          // La borne opposée n'a pas bougé dans cette branche : `limit` capture
          // bien la fenêtre héritée du parent. En mode strict, l'égalité ne
          // coupe pas — elle sera départagée par échantillonnage réservoir.
          const cut = frame.maximizing
            ? strictPruning
              ? frame.bestScore > frame.beta
              : frame.bestScore >= frame.beta
            : strictPruning
              ? frame.bestScore < frame.alpha
              : frame.bestScore <= frame.alpha;
          if (cut) {
            frame.cut = true;
            frame.cutInfo = {
              bound: frame.maximizing ? 'beta' : 'alpha',
              strict: strictPruning,
              bestScore: frame.bestScore,
              limit: frame.maximizing ? frame.beta : frame.alpha,
              triggerKey: frame.triggerKey,
            };
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
      // Profondeur effective certifiée : un frame coupé ne certifie que son
      // budget restant ; sinon un ply de plus que son enfant le moins profond
      // (1 + SOLVED_DEPTH === SOLVED_DEPTH).
      const certifiedDepth = frame.cut ? frame.remaining : 1 + frame.minChildDepth;
      emitValue({
        nodeKey: frame.key,
        score: frame.bestScore,
        depth: frame.remaining,
        exact,
        window: windowOf(frame),
        ...(frame.cutInfo === null ? {} : { cut: frame.cutInfo }),
      });
      if (transposition !== null) {
        // Score intrinsèque : du point de vue du joueur au trait dans cet état.
        transposition.store(ttKey(frame.state), {
          score: frame.maximizing ? frame.bestScore : -frame.bestScore,
          depth: certifiedDepth,
          exact,
          ...(game.pieceCount === undefined ? {} : { pieces: game.pieceCount(frame.state) }),
        });
      }
      if (frame.parentKey === null) {
        effectiveDepth = certifiedDepth;
        rootMove = frame.bestMove;
        return;
      }
      const parent = frames.at(-1);
      if (parent) {
        accumulate(parent, frame.move as M, frame.key, frame.bestScore, exact, certifiedDepth);
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
        // Racine terminale : jamais « complétée » via un frame, mais sa
        // profondeur certifiée (SOLVED_DEPTH) reste la profondeur effective.
        effectiveDepth = rootResolution.depth;
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
          accumulate(frame, move, child.key, resolution.score, resolution.exact, resolution.depth);
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
      effectiveDepth = 0;
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
        effectiveDepth,
      };
    },
  };
}
