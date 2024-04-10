import numpy as np

from games.hex import HexStateManager, Player


def generate_more_cases(state: np.ndarray, distribution: np.ndarray, player_value, k):

    cases = [(np.append(state.flatten(), player_value), distribution)]

    #generate state for opponent
    opponent_state = generate_opponent_state(state)
    opponent_distribution = generate_opponent_distribution(k, distribution)
    if player_value == 1: opponent_value = 2 
    else: opponent_value = 1
    cases.append((np.append(opponent_state, opponent_value), opponent_distribution))

    #generate mirrored cases
    final_cases = []
    final_cases.extend(cases)
 
    for case in cases:
        mirrored_state= rotate_state(k, case[0][:-1])
        mirrored_distribution = rotate_distribution(k, distribution)
        
        final_cases.append((np.append(mirrored_state, case[0][-1]), mirrored_distribution))
    #print(final_cases)
    return cases

def rotate_state(k, state: np.ndarray):

    state = state.reshape((k, k))
    state = np.flip(np.flip(state, axis=0), axis=1)
    state = state.flatten()
    return state

def rotate_distribution(k: int,distribution: np.ndarray) -> np.ndarray:

    distribution_matrix = distribution.reshape((k, k))
    rotated_distribution_matrix = np.flip(np.flip(distribution_matrix, axis=0), axis=1)
    rotated_distribution = rotated_distribution_matrix.flatten()
    return rotated_distribution

def generate_opponent_state(board):
    print(board)
    rotated_board = np.transpose(board)[:, ::-1]
    conditions = [rotated_board == 1, rotated_board == 2]
    choices = [2, 1]  # The choices to swap
    swapped_board = np.select(conditions, choices, default=rotated_board)
    print(swapped_board)
    
    return swapped_board.flatten()

def generate_opponent_distribution(k: int, distribution: np.ndarray) -> np.ndarray:
    distribution_2d = distribution.reshape((k, k))
    print(distribution_2d)

    rotated_distribution = np.transpose(distribution_2d)[:, ::-1]
    print(rotated_distribution)
    transformed_distribution = rotated_distribution.flatten()

    return transformed_distribution

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