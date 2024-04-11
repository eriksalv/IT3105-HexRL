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
print(tf.config.experimental.list_physical_devices('GPU'))

class RLTrainer:
    def __init__(self, k: int, anet_config_name: str, contains_bridges = False, padding = False, cnn = False):
        self.k = k
        self.anet_config_name = anet_config_name
        self.contains_bridges = contains_bridges
        self.padding = padding
        self.state_manager = HexStateManager(self.k)
        self.replay_buffer = ReplayBuffer(max_size=1024)
        if cnn:
            self.anet = CNN(self.k, self.anet_config_name, contains_bridges = contains_bridges, padding = padding)
        else:
            self.anet = ActorNetwork(self.k, self.anet_config_name, contains_bridges = contains_bridges, padding = padding)
        
    def train(self, episodes: int, simulations: int, save_interval: int, evaluate_during: bool = False):
        # Save untrained network
        self.anet.save_parameters(f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_0.weights.h5')
        starting_player = Player.RED
        if evaluate_during:
            evaluation_network = copy.deepcopy(self.anet)
            
        for episode in range(1, episodes + 1):
            print(f'Episode: {episode}')
            if episode % 5 == 0 and evaluate_during:
                if evaluate_network(evaluation_network, self.anet, self.k, n_games=400, threshold=0.55):
                    evaluation_network = copy.deepcopy(self.anet)

            start = time.perf_counter()

            # initialize new board
            self.state_manager.new_game(starting_player=starting_player)
            s_init = self.state_manager.get_state()
            root = Node(s_init)
            root.init_actions_and_values(self.state_manager.get_legal_moves())

            
            while not self.state_manager.is_final():
                search_tree = MCTS(self.state_manager, root, self.anet, expansion_threshold=20)
                best_action, distribution = search_tree.search(simulations)
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
            minibatch = self.replay_buffer.get_minibatch(batch_size=1024)
            self.anet.train(minibatch, epochs = 300)

            # starting_player = Player.BLUE if starting_player == Player.RED else Player.RED

            end = time.perf_counter()
            print(f'time: {end - start:.1f}s')

            if episode % save_interval == 0:
                # save ANET parameters
                self.anet.save_parameters(
                    f'./trained_networks/{self.anet_config_name}_no_bridges_{self.k}x{self.k}_{episode}.weights.h5')
        self.anet.plot_history()


if __name__ == "__main__":
    game_manager = RLTrainer(k=3, anet_config_name='jespee_anet', contains_bridges=False, padding=False, cnn=False)
    game_manager.train(episodes=200,
                       simulations=500,
                       save_interval=20,
                       evaluate_during=True)

    #trainer = RLTrainer(k=4, anet_config_name='anet')
    #trainer.train(episodes=200, simulations=500, save_interval=40)
    #trainer = RLTrainer(k=4, anet_config_name='jespee_anet')
    #trainer.train(episodes=200, simulations=500, save_interval=10)
