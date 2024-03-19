import json
from rbuf import ReplayBuffer
from anet import ActorNetwork
from mcts.mcts import MCTS, Node
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

def vectorize_distribution(distribution):

    #initialize the vectorized distribution array
    vectorized_distribution = np.zeros(len(distribution), dtype=float)

    #iterate over the dictionary keys and set the probabilities in the vectorized distribution array
    for move, probability in distribution.items():
        index = move[0] * 3 + move[1]  # Calculate the index based on the tuple (i, j)
        vectorized_distribution[index] = probability

    return vectorized_distribution
class GameManager:
    def __init__(self):
        self.hex_state_manager = HexStateManager(3)
        self.replay_buffer = ReplayBuffer()

        with open('config/anet.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.anet = ActorNetwork(3, config)
        

    def train(self, num_actual_games, num_search_games, save_interval):
        for ga in range(1, num_actual_games + 1):
            print(ga)
            #initialize new gameboard
            self.hex_state_manager.new_game()
            Ba = self.hex_state_manager
            
            #get starting state as root
            sinit = Node(Ba)
            root = sinit
            #initialize new mcts
            mcts = MCTS(sinit, actor_net=self.anet)

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
                mcts.search(num_search_games)

                #update Replay Buffer
                D = mcts.get_distribution(root)
                
                D_true = create_distribution(D, 3)
                #print(D_true)
                D_vectorized = vectorize_distribution(D_true)

                vectorized_board = root.state.board.flatten()
                current_player = self.hex_state_manager.current_player.value
                vectorized_state = np.append(vectorized_board, current_player)
                
                self.replay_buffer.add_case((vectorized_state, D_vectorized))

                #choose actual move based on distribution
                a_star = max(D, key = D.get)
                self.hex_state_manager.make_move(a_star)
                #update Ba and MCT
                Ba = self.hex_state_manager
                
                root = Node(Ba)
                mcts.update(root)

            #train ANET
            minibatch = self.replay_buffer.get_minibatch()
            self.anet.train(minibatch)
            if ga % save_interval == 0:
                #save ANET parameters
                self.anet.save_parameters(f'./trained_networks/anet_{ga}.weights.h5')

if __name__ == "__main__":
    game_manager = GameManager()
    game_manager.train(num_actual_games=20, num_search_games=100, save_interval=10)