Two interactive demonstrations to see, manipulate, and compare two families of algorithms that matter in the history of AI. Everything happens in your browser. No server receives your clicks, and no data leaves your machine.

### Deep Blue: search before playing

The first demonstration replays a game of tic-tac-toe or Connect Four. The agent explores the tree of possible moves, then propagates the values back up to the root. You can compare minimax and alpha-beta pruning, change the search depth or the evaluation function. The displayed tree shows the visited nodes, the cutoffs, and the transposition table. The score of a position depends on the remaining depth and the discount factor $\gamma$: an immediate gain is worth more than a distant one.

### Frozen Lake: learn by trying

The second demonstration places an agent on a frozen lake. It must reach a candy without falling into a hole. This time, nobody gives it the map: it learns the value $Q(s,a)$ of each state-action pair by exploring, making mistakes, and starting over. The sliders set the exploration, the learning rate, and the replay memory.

### A common thread

Search and learning answer the same question: "how do you choose the next move?" Search computes a value from the rules of the game. Learning estimates that value from experience. Both pages show this difference with the same controls, without sending anything over the network.
