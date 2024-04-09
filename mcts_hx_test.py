import numpy as np

from anet import ActorNetwork
from games.hex import HexStateManager, Player
from mcts import MCTS, Node


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
    hsm.new_game(starting_player=Player.BLUE)
    hsm.make_move((0, 0))
    hsm.make_move((1, 0))
    hsm.make_move((0, 1))
    hsm.make_move((1, 1))
    # hsm.make_move((0,2)) winning move

    root = Node(state=hsm.get_state())

    net = ActorNetwork(3, 'anet')
    mcts = MCTS(state_manager=hsm, root=root, actor_net=net)
    best_move, distribution = mcts.search(100)
    print(f"Best move: {best_move} ")
    print(distribution)
    print(root.value)
    print(root.actions)

    hsm = HexStateManager(k)
    hsm.new_game()
    hsm.make_move((1, 0))
    hsm.make_move((2, 0))
    hsm.make_move((0, 1))
    hsm.make_move((1, 1))
    hsm.make_move((1, 2))
    hsm.show_board()
    # hsm.make_move((0,2)) winning move

    root = Node(state=hsm.get_state())

    net = ActorNetwork(3, 'anet')
    mcts = MCTS(state_manager=hsm, root=root, actor_net=net)
    best_move, distribution = mcts.search(100)
    print(f"Best move: {best_move} ")
    print(distribution)
    print(root.value)
    print(root.actions)
