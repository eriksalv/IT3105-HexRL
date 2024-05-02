from games.nim import Nim
from mcts import MCTS, Node

if __name__ == "__main__":
    root_state = Nim(15, 3, True)
    root = Node(root_state)
    mcts = MCTS(root, expansion_threshold=400)
    best_child = mcts.search(10000)
    print(f"Best move: {root_state.n - best_child.state.n} stones")
