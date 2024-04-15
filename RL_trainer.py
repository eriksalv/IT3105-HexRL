import copy
import time

from anet import ActorNetwork
from games.hex import HexStateManager, Player, mark_bridges
from mcts import MCTS, Node
from rbuf import ReplayBuffer

from utils import add_padding, generate_more_cases, evaluate_network
from cnn_network import CNN
import sys
sys.stdout.reconfigure(encoding='utf-8')
import tensorflow as tf
tf.random.set_seed(42)
print(tf.config.experimental.list_physical_devices('GPU'))

class RLTrainer:
    def __init__(self, k: int, anet_config_name: str, contains_bridges = False, padding = False, cnn = False, saved_weights_file = None):
        self.k = k
        self.anet_config_name = anet_config_name
        self.contains_bridges = contains_bridges
        self.padding = padding
        self.state_manager = HexStateManager(self.k)
        self.calculated_batch_size = self.k ** 2 * 4 * 6
        print(self.calculated_batch_size)
        self.replay_buffer = ReplayBuffer(max_size=2048)
        if cnn:
            self.anet = CNN(self.k, self.anet_config_name, contains_bridges = contains_bridges, padding = padding)
        else:
            self.anet = ActorNetwork(self.k, self.anet_config_name, saved_weights_file = saved_weights_file, contains_bridges = contains_bridges, padding = padding)
        
    def train(self, episodes: int, simulations: int, save_interval: int, evaluate_during: bool = False, n_pretrained_episodes = 0,
               increasing_mcts = True, min_simulations = 300, max_simulations = 1000):
        
        # Save untrained network
        self.anet.save_parameters(f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_0.weights.h5')
        starting_player = Player.RED
        if evaluate_during:
            evaluation_network = copy.deepcopy(self.anet)
   
        for episode in range(1, episodes + 1):
            print(f'Episode: {episode}')
            
            start = time.perf_counter()

            # initialize new board
            self.state_manager.new_game(starting_player=starting_player)
            s_init = self.state_manager.get_state()
            root = Node(s_init)
            root.init_actions_and_values(self.state_manager.get_legal_moves())

            
            n_moves = 0
            while not self.state_manager.is_final():
                search_tree = MCTS(self.state_manager, root, self.anet, expansion_threshold=20, c=1.4)
                if increasing_mcts:
                    budget = int((max_simulations - min_simulations) * (n_moves/(self.k**2)/2) + min_simulations) # we gradually increase amount of searches to max_value when half the board is filled
                else: 
                    budget = simulations
 
                best_action, distribution = search_tree.search(budget)
                n_moves += 1
                board_state = root.state[0]
                if self.padding:
                    board_state = add_padding(board_state)
                if self.contains_bridges:
                    board_state = mark_bridges(board_state)
                    case = self.anet.vectorize_case((board_state, root.state[1]), distribution)
                else:
                    case = self.anet.vectorize_case(root.state, distribution)

               
                generated_cases = generate_more_cases(state=root.state[0], distribution=case[1], k = self.k, player_value=root.state[1], contains_bridges = self.contains_bridges, padding = self.padding)
                for case in generated_cases:
          
                    self.replay_buffer.add_case(case)

                self.state_manager.make_move(best_action)
                root = root.children[best_action]
                root.parent = None

            # train ANET
            
            minibatch = self.replay_buffer.get_minibatch(batch_size=2048)
            self.anet.train(minibatch, epochs = 1)

            # starting_player = Player.BLUE if starting_player == Player.RED else Player.RED

            end = time.perf_counter()
            print(f'time: {end - start:.1f}s')
            if episode % 3 == 0 and evaluate_during:
                if evaluate_network(evaluation_network, self.anet, self.k, n_games=400, threshold=0.50, random_move = True):
                    evaluation_network = copy.deepcopy(self.anet)
                else:
                    self.anet = evaluation_network
            if episode % save_interval == 0:
                # save ANET parameters
                self.anet.save_parameters(
                    f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_{episode + n_pretrained_episodes}.weights.h5')
        self.anet.plot_history()


if __name__ == "__main__":
    game_manager = RLTrainer(k=3, anet_config_name='demo_anet', contains_bridges=False, padding=False, cnn=False,
                            )
    game_manager.train(episodes=300,
                       simulations=400,
                       save_interval=10,
                       evaluate_during=True,
                        increasing_mcts = True,
                        min_simulations = 400,
                        max_simulations = 2000)

    #trainer = RLTrainer(k=4, anet_config_name='anet')
    #trainer.train(episodes=200, simulations=500, save_interval=40)
    #trainer = RLTrainer(k=4, anet_config_name='jespee_anet')
    #trainer.train(episodes=200, simulations=500, save_interval=10)
