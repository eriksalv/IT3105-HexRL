import numpy as np

from games.hex import HexStateManager, Player


def generate_more_cases(state: np.ndarray, distribution: np.ndarray, player_value, k):
    #print(state)
    #print(distribution)
    #print(player_value)
    cases = [(np.append(state.flatten(), player_value), distribution)]

    #generate state for opponent
    opponent_state = generate_blue_state(state)
    opponent_distribution = generate_blue_distribution(k, distribution)
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

def generate_blue_state(board):
    rotated_board = np.transpose(board)[:, ::-1]
    conditions = [rotated_board == 1, rotated_board == 2]
    choices = [2, 1]  # The choices to swap
    swapped_board = np.select(conditions, choices, default=rotated_board)
    
    return swapped_board.flatten()

def generate_blue_distribution(k: int, distribution: np.ndarray) -> np.ndarray:
    distribution_2d = distribution.reshape((k, k))
    rotated_distribution = np.transpose(distribution_2d)[:, ::-1]
    
    transformed_distribution = rotated_distribution.flatten()

    return transformed_distribution

if __name__ == '__main__':
    k = 3
    hsm = HexStateManager(k)
    hsm.new_game()
    hsm.make_move((0, 0))
    hsm.make_move((2, 0))
    hsm.make_move((0, 1))
    hsm.make_move((1, 0))
    hsm.make_move((1, 1))
    hsm.make_move((1, 2))
    hsm.make_move((2 ,1)) 

    hsm.show_board()


    generate_more_cases(hsm.board, np.array([1,0,0,0,0,0,0,0,0]), k)
    #print(np.flip(hsm.board.flatten(), axis=0)) # faster? seems like this performs the same operation
