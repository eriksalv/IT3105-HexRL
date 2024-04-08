import json
from rbuf import ReplayBuffer
from anet import ActorNetwork
from mcts import MCTS, Node
from hex import HexStateManager
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')


def initialize_distribution(k):
    return {(i, j): 0.0 for i in range(k) for j in range(k)}


def create_distribution(true_distribution, k):
    distribution = initialize_distribution(k)
    for move, probability in true_distribution.items():
        distribution[move] = probability
    return distribution


def vectorize_distribution(distribution, k):

    # initialize the vectorized distribution array
    vectorized_distribution = np.zeros(len(distribution), dtype=float)

    # iterate over the dictionary keys and set the probabilities in the vectorized distribution array
    for move, probability in distribution.items():
        # Calculate the index based on the tuple (i, j)
        index = move[0] * k + move[1]
        vectorized_distribution[index] = probability

    return vectorized_distribution


class GameManager:
    def __init__(self, k):
        self.k = k
        self.hex_state_manager = HexStateManager(self.k)
        self.replay_buffer = ReplayBuffer()

        with open('config/anet.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.anet = ActorNetwork(self.k, config)

    def train(self, num_actual_games, num_search_games, save_interval):
        for ga in range(1, num_actual_games + 1):
            print(ga)
            # initialize new gameboard
            self.hex_state_manager.new_game()
            Ba = self.hex_state_manager

            # get starting state as root
            sinit = Node(Ba)
            root = sinit
            # initialize new mcts
            mcts = MCTS(sinit, actor_net=self.anet, expansion_threshold=7)

            while not Ba.is_final():

                """
                Pseudo
                for gs in range(num_search_games):

                    # (d) Search from root to leaf
                    leaf = mcts.search(root)

                    # Use ANET to choose rollout actions
                    F = mcts.rollout(leaf)

                    # Perform MCTS backpropagation
                    mcts.backpropagate(leaf, F)
                """
                current_player = self.hex_state_manager.current_player.value

                mcts.search(num_search_games,
                            original_player_value=current_player)

                # update Replay Buffer
                D = mcts.get_distribution(root)

                D_true = create_distribution(D, self.k)
                # print(D_true)
                D_vectorized = vectorize_distribution(D_true, self.k)
                # print(self.hex_state_manager.board)
                # print(D_vectorized)
                vectorized_board = root.state.board.flatten()
                vectorized_state = np.append(vectorized_board, current_player)

                self.replay_buffer.add_case((vectorized_state, D_vectorized))

                # choose actual move based on distribution
                a_star = max(D, key=D.get)
                self.hex_state_manager.make_move(a_star)
                # update Ba and MCT
                Ba = self.hex_state_manager

                root = Node(Ba)
                mcts.update(root)

            # train ANET
            minibatch = self.replay_buffer.get_minibatch(batch_size=128)
            # self.replay_buffer.clear()
            print(len(self.replay_buffer.cases))
            self.anet.train(minibatch)
            if ga % save_interval == 0 or ga == 1:
                # save ANET parameters
                self.anet.save_parameters(
                    f'./trained_networks/anet_{ga}.weights.h5')


if __name__ == "__main__":
    game_manager = GameManager(k=4)
    game_manager.train(num_actual_games=100,
                       num_search_games=300, save_interval=50)
