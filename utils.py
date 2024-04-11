import numpy as np

from games.hex import HexStateManager, Player, mark_bridges


def generate_more_cases(state: np.ndarray, distribution: np.ndarray, player_value, k, contains_bridges = False, padding = False):
    if padding:
        state = add_padding(state)
        k+=2
    cases = [(np.append(state.flatten(), player_value), distribution)]

    #generate state for opponent
    opponent_state = generate_opponent_state(state, contains_bridges = contains_bridges)
    opponent_distribution = generate_opponent_distribution(k, distribution, padding = padding)
    if player_value == 1: opponent_value = 2 
    else: opponent_value = 1
    cases.append((np.append(opponent_state, opponent_value), opponent_distribution))

    #generate mirrored cases
    final_cases = []
    final_cases.extend(cases)
 
    for case in cases:
        mirrored_state= rotate_state(k, case[0][:-1])
        mirrored_distribution = rotate_distribution(k, case[1], padding = padding)
        
        final_cases.append((np.append(mirrored_state, case[0][-1]), mirrored_distribution))
    #print(final_cases)
    return final_cases

def rotate_state(k, state: np.ndarray):
    
    state = state.reshape((k, k))
    state = np.flip(np.flip(state, axis=0), axis=1)
    state = state.flatten()
    return state

def rotate_distribution(k: int,distribution: np.ndarray, padding = False) -> np.ndarray:
    if padding:
        k -=2
    distribution_matrix = distribution.reshape((k, k))
    rotated_distribution_matrix = np.flip(np.flip(distribution_matrix, axis=0), axis=1)
    rotated_distribution = rotated_distribution_matrix.flatten()
    return rotated_distribution

def generate_opponent_state(board, contains_bridges = False):
    #print(board)
    rotated_board = np.transpose(board)[:, ::-1]
    conditions = [rotated_board == 1, rotated_board == 2]
    choices = [2, 1]  # The choices to swap
    swapped_board = np.select(conditions, choices, default=rotated_board)
    if contains_bridges:
        conditions = [rotated_board == 3, rotated_board == 4]
        choices = [4, 3]
        swapped_board = np.select(conditions, choices, default=swapped_board)
    
    #print(swapped_board)
    
    return swapped_board.flatten()

def generate_opponent_distribution(k: int, distribution: np.ndarray, padding = False) -> np.ndarray:
    if padding:
        k -=2
    distribution_2d = distribution.reshape((k, k))
    #print(distribution_2d)

    rotated_distribution = np.transpose(distribution_2d)[:, ::-1]
    #print(rotated_distribution)
    transformed_distribution = rotated_distribution.flatten()

    return transformed_distribution
def add_padding(board):
    # Assuming 'board' is your 2D NumPy array for the game state

    # Add a row of ones at the top and bottom
    top_bottom_padding = np.ones((1, board.shape[1]))
    board_with_vertical_padding = np.vstack([top_bottom_padding, board, top_bottom_padding])

    # Add a column of twos on the left and right sides
    left_right_padding = np.full((board_with_vertical_padding.shape[0], 1), 2)
    padded_board = np.hstack([left_right_padding, board_with_vertical_padding, left_right_padding])

    return padded_board

def play_game(k, starting_player, current_best_net, new_net, show_board=False):
        hsm = HexStateManager(k)
        hsm.new_game(starting_player=starting_player)

        # To make games not identical, start by playing a random move in the middle horizontal line of the hex board
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

def simulate_games(k, current_best_net, new_net, n_games=400, show_board=False):
        win_dict = {1: 0, 2: 0}
        starting_player = Player.RED
        for _ in range(n_games):
            winner = play_game(
                k, starting_player, current_best_net, new_net, show_board=show_board)
            win_dict[winner] += 1

            # alternate if player 1 (red) or player 2 (blue) starts
            starting_player = Player.BLUE if starting_player == Player.RED else Player.RED

        return win_dict

def evaluate_network(current_best_net, new_net, k, n_games = 400, threshold = 0.55):
    res = simulate_games(k=k, current_best_net=current_best_net, new_net=new_net, n_games=n_games)
    new_net_wr = res[2]/n_games
    print('Winrate of new net: ')
    print(new_net_wr)
    return new_net_wr>threshold

if __name__ == '__main__':
    print('Example 1 - shows that the move to be made as blue will translate to the move that should be made by red')
    k = 3
    hsm = HexStateManager(k)
    hsm.new_game()
    hsm.make_move((0, 0)) #red
    hsm.make_move((2, 0)) #blue
    hsm.make_move((0, 1)) #red
    hsm.make_move((1, 0)) #blue
    hsm.make_move((1, 1)) #red
    
    

    #hsm.show_board()


    generate_more_cases(hsm.board, np.array([0,0,0,0,0,0,0,1,0]), hsm.current_player.value, k)

    #print(np.flip(hsm.board.flatten(), axis=0)) # faster? seems like this performs the same operation

    print('Example 2 - shows that the move to be made as red will translate to the move that should be made by blue')

    k = 3
    hsm = HexStateManager(k)
    hsm.new_game()
    hsm.make_move((0, 0)) #red
    hsm.make_move((2, 0)) #blue
    hsm.make_move((0, 1)) #red
    hsm.make_move((1, 0)) #blue
    hsm.make_move((1, 1)) #red
    hsm.make_move((2, 1)) #blue

    #hsm.show_board()

 
    generate_more_cases(hsm.board, np.array([0,0,0,0,0,0,0,0,1]), hsm.current_player.value, k)
  


    final_cases = generate_more_cases(hsm.board, np.array([0,0,0,0,0,0,0,0,1]), hsm.current_player.value, k)
    details = ['Blue can win on the next turn with the move: (2,2), so red needs to block',
               'Swapped players, red can win on the next move (2,0), so blue needs to block',
               'Mirrored case of case 1, this time blue can win with the move (0,0), so red needs to block',
               'Mirrored case of case 2, this time red can win with the move (0,2), so blue needs to block']
    i=1
    for case in final_cases:
        print(f'Case {i}: ', details[i-1])
        print(f'Player {case[0][-1]} to play')
        i+=1
        print(case[0][:-1].reshape(k, k))
        print()
        print(case[1].reshape(k, k))
    

    state_manager = HexStateManager(4)
    state_manager.new_game()
    state_manager.make_move((1,1))
    state_manager.make_move((3,0))
    state_manager.make_move((0,0))
    state_manager.make_move((2,1))
    state_manager.make_move((2,2))
    padded_board = add_padding(state_manager.board)

    bridge_board = mark_bridges(padded_board)
 
    generate_more_cases(bridge_board, np.array(np.zeros(36)), hsm.current_player.value, 4, contains_bridges=True, padding=True)

    