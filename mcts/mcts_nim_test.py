from mcts import MCTS, Node
from nim import Nim

if __name__ == "__main__":
    root_state = Nim(7, 3, True) 
    root = Node(root_state)
    mcts = MCTS(root)
    best_child = mcts.search(10000) 
    print(f"Best move: {root_state.n - best_child.state.n} stones")