import numpy as np

from games.hex import HexStateManager, Player
from networks.anet import ActorNetwork


def generate_more_cases(case: tuple[np.ndarray, np.ndarray], k: int, contains_bridges=False):
    """
    Generates 3 additional cases by transposing board state, and rotating both by
    180 degrees.
    """
    cases = [case]
    board_state = case[0][:-1].reshape(k, k)
    current_player = case[0][-1]
    distribution = case[1]

    # generate state for opponent
    opponent_state = generate_opponent_state(board_state, contains_bridges=contains_bridges)
    opponent_distribution = generate_opponent_distribution(k, distribution)
    opponent_value = 1 if current_player == 2 else 2
    cases.append((np.append(opponent_state, opponent_value), opponent_distribution))

    # generate mirrored cases
    for i in range(2):
        mirrored_state = mirror_state(k, cases[i][0][:-1])
        mirrored_distribution = mirror_distribution(k, cases[i][1])
        cases.append((np.append(mirrored_state, cases[i][0][-1]), mirrored_distribution))

    return cases


def generate_opponent_state(board_state: np.ndarray, contains_bridges=False):
    # Use transpose to preserve neighbourhood
    rotated_board = np.transpose(board_state)

    conditions = [rotated_board == 1, rotated_board == 2]
    choices = [2, 1]  # The choices to swap
    swapped_board = np.select(conditions, choices, default=rotated_board)

    if contains_bridges:
        conditions = [rotated_board == 3, rotated_board == 4]
        choices = [4, 3]
        swapped_board = np.select(conditions, choices, default=swapped_board)

    return swapped_board.flatten()


def generate_opponent_distribution(k: int, distribution: np.ndarray) -> np.ndarray:
    distribution_2d = distribution.reshape((k, k))

    rotated_distribution = np.transpose(distribution_2d)
    return rotated_distribution.flatten()


def mirror_state(k: int, state: np.ndarray):
    state = state.reshape((k, k))
    state = np.rot90(state, 2)
    state = state.flatten()
    return state


def mirror_distribution(k: int, distribution: np.ndarray) -> np.ndarray:
    distribution = distribution.reshape((k, k))
    distribution = np.rot90(distribution, 2)
    distribution = distribution.flatten()
    return distribution


def play_game(k: int, starting_player: Player, current_best_net: ActorNetwork, new_net: ActorNetwork, show_board=False,
              random_move=True):
    hsm = HexStateManager(k)
    hsm.new_game(starting_player=starting_player)

    # To make games not identical, start by playing a random move in the middle horizontal line of the hex board
    if random_move:
        hsm.make_random_starting_move()

    while not hsm.is_final():
        if hsm.current_player == Player.RED:
            action = current_best_net.get_action(
                hsm.board, hsm.current_player.value)
        else:
            action = new_net.get_action(
                hsm.board, hsm.current_player.value)

        hsm.make_move(action)

    if show_board:
        hsm.show_board()

    return hsm.winner.value


def simulate_games(k: int, current_best_net: ActorNetwork, new_net: ActorNetwork, n_games=400, show_board=False,
                   random_move=True):
    win_dict = {1: 0, 2: 0}
    starting_player = Player.RED
    for _ in range(n_games):
        winner = play_game(
            k, starting_player, current_best_net, new_net, show_board=show_board, random_move=random_move)
        win_dict[winner] += 1

        # alternate if player 1 (red) or player 2 (blue) starts
        starting_player = Player.BLUE if starting_player == Player.RED else Player.RED

    return win_dict


def evaluate_network(current_best_net, new_net, k, n_games=400, threshold=0.55, random_move=True):
    res = simulate_games(k=k, current_best_net=current_best_net, new_net=new_net, n_games=n_games,
                         random_move=random_move)
    print(res)
    new_net_wr = res[2] / n_games
    print('Winrate of new net: ')
    print(new_net_wr)
    return new_net_wr > threshold


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('Example 1 - shows that the move to be made as blue will translate to the move that should be made by red')
    k = 3
    hsm = HexStateManager(k)
    hsm.new_game()
    hsm.make_move((0, 0))  # red
    hsm.make_move((2, 0))  # blue
    hsm.make_move((0, 1))  # red
    hsm.make_move((1, 0))  # blue
    hsm.make_move((1, 1))  # red

    # hsm.show_board()

    final_cases = generate_more_cases(
        (ActorNetwork.vectorize_state(hsm.board, hsm.current_player.value), np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])), k)

    # hsm.board = final_cases[1][0][:-1].reshape(k, k)

    # hsm.show_board()

    # print(np.flip(hsm.board.flatten(), axis=0)) # faster? seems like this performs the same operation

    print('Example 2 - shows that the move to be made as red will translate to the move that should be made by blue')

    k = 3
    hsm = HexStateManager(k)
    hsm.new_game()
    hsm.make_move((0, 0))  # red
    hsm.make_move((2, 0))  # blue
    hsm.make_move((0, 1))  # red
    hsm.make_move((1, 0))  # blue
    hsm.make_move((1, 1))  # red
    hsm.make_move((2, 2))  # blue

    # hsm.show_board()

    final_cases = generate_more_cases(
        (ActorNetwork.vectorize_state(hsm.board, hsm.current_player.value), np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])), k)

    details = ['Blue can win on the next turn with the move: (2,1), so red needs to block',
               'Swapped players, red can win on the next move (1,2), so blue needs to block',
               'Mirrored case of case 1, this time blue can win with the move (0,1), so red needs to block',
               'Mirrored case of case 2, this time red can win with the move (1,0), so blue needs to block']
    i = 1
    for case in final_cases:
        print(f'Case {i}: ', details[i - 1])
        print(f'Player {case[0][-1]} to play')
        i += 1
        print(case[0][:-1].reshape(k, k))
        print()
        print(case[1].reshape(k, k))

    plt.show()

    # state_manager = HexStateManager(4)
    # state_manager.new_game()
    # state_manager.make_move((1, 1))
    # state_manager.make_move((3, 0))
    # state_manager.make_move((0, 0))
    # state_manager.make_move((2, 1))
    # state_manager.make_move((2, 2))
    # padded_board = state_manager.board
    #
    # bridge_board = mark_bridges(padded_board)
    #
    # generate_more_cases(bridge_board, np.array(np.zeros(16)), hsm.current_player.value, 4, contains_bridges=True,
    #                     padding=True)
