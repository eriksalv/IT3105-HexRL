import time

from anet import ActorNetwork
from games.hex import HexStateManager, Player
from mcts import MCTS, Node
from rbuf import ReplayBuffer
from utils import generate_more_cases
import sys
sys.stdout.reconfigure(encoding='utf-8')
import tensorflow as tf
print(tf.config.experimental.list_physical_devices('GPU'))
class RLTrainer:
    def __init__(self, k: int, anet_config_name: str):
        self.k = k
        self.anet_config_name = anet_config_name
        self.state_manager = HexStateManager(self.k)
        self.replay_buffer = ReplayBuffer()
        self.anet = ActorNetwork(self.k, self.anet_config_name)

    def train(self, episodes: int, simulations: int, save_interval: int):
        # Save untrained network
        self.anet.save_parameters(f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_0.weights.h5')
        starting_player = Player.RED

        for episode in range(1, episodes + 1):
            print(f'Episode: {episode}')
            start = time.perf_counter()

            # initialize new board
            self.state_manager.new_game(starting_player=starting_player)
            s_init = self.state_manager.get_state()
            root = Node(s_init)
            root.init_actions_and_values(self.state_manager.get_legal_moves())

            
            while not self.state_manager.is_final():
                search_tree = MCTS(self.state_manager, root, self.anet, expansion_threshold=100)
                best_action, distribution = search_tree.search(simulations)

                case = self.anet.vectorize_case(root.state, distribution)
      
                self.replay_buffer.add_case(case)
           
               
                generated_cases = generate_more_cases(state=root.state[0], distribution=case[1], k = self.k, player_value=root.state[1])
                for case in generated_cases:
                    self.replay_buffer.add_case(case)

                self.state_manager.make_move(best_action)
                root = root.children[best_action]
                root.parent = None

            # train ANET
            minibatch = self.replay_buffer.get_minibatch(batch_size=512)
            self.anet.train(minibatch, epochs = 50)

            # starting_player = Player.BLUE if starting_player == Player.RED else Player.RED

            end = time.perf_counter()
            print(f'time: {end - start:.1f}s')

            if episode % save_interval == 0:
                # save ANET parameters
                self.anet.save_parameters(
                    f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_{episode}.weights.h5')
        self.anet.plot_history()


if __name__ == "__main__":
    game_manager = RLTrainer(k=7, anet_config_name='jespee_anet')
    game_manager.train(episodes=200,
                       simulations=1000,
                       save_interval=10)

    trainer = RLTrainer(k=4, anet_config_name='anet')
    trainer.train(episodes=200, simulations=500, save_interval=40)
    #trainer = RLTrainer(k=4, anet_config_name='jespee_anet')
    #trainer.train(episodes=200, simulations=500, save_interval=10)
