from hex import HexStateManager
from mcts import MCTS, Node
import numpy as np
from anet import ActorNetwork
import json


def find_last_move(prev_state, current_state):
    diff = current_state - prev_state
    indices = np.argwhere(diff != 0)

    if len(indices) == 1:
        return tuple(indices[0])
    else:
        raise ValueError("Invalid states: More than one difference found.")


if __name__ == "__main__":
    k = 3
    hsm = HexStateManager(k)
    hsm.new_game()
    hsm.make_move((0, 0))
    hsm.make_move((2, 0))
    hsm.make_move((0, 1))
    hsm.make_move((1, 0))
    hsm.make_move((1, 1))
    hsm.make_move((1, 2))
    # hsm.show_board()
    # hsm.make_move((2,1)) winning move

    root_state = hsm
    root = Node(root_state)

    net = ActorNetwork(3, 'anet')
    mcts = MCTS(root, actor_net=net)
    best_child = mcts.search(100)
    best_move = find_last_move(root_state.board, best_child.state.board)
    print(f"Best move: {best_move} ")

    hsm = HexStateManager(k)
    hsm.new_game()
    hsm.make_move((1, 0))
    hsm.make_move((2, 0))
    hsm.make_move((0, 1))
    hsm.make_move((1, 1))
    hsm.make_move((1, 2))
    # hsm.make_move((0, 2)) #winning move for blue
    hsm.show_board()

    root_state = hsm
    root = Node(root_state)

    net = ActorNetwork(3, 'anet')
    mcts = MCTS(root, actor_net=net)
    best_child = mcts.search(100, original_player_value=2)
    best_move = find_last_move(root_state.board, best_child.state.board)
    print(f"Best move: {best_move} ")
