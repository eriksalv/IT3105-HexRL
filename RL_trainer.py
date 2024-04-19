import copy
import math
import sys
import time

from games.hex import HexStateManager, Player, mark_bridge_endpoints
from mcts import MCTS, Node
from networks.basic_anet import BasicActorNet
from networks.conv_anet import ConvActorNet
from rbuf import ReplayBuffer
from utils import generate_more_cases, evaluate_network


class RLTrainer:
    def __init__(self, k: int, anet_config_name: str, contains_bridges=False, padding=0, cnn=False,
                 saved_weights_file=None):
        self.k = k
        self.anet_config_name = anet_config_name
        self.contains_bridges = contains_bridges
        self.padding = padding
        self.state_manager = HexStateManager(self.k)
        self.replay_buffer = ReplayBuffer(max_size=2048)
        if cnn:
            self.anet = ConvActorNet(self.k, self.anet_config_name, contains_bridges=contains_bridges, padding=padding)
        else:
            self.anet = BasicActorNet(self.k, self.anet_config_name, saved_weights_file=saved_weights_file,
                                      contains_bridges=contains_bridges)

    def train(self, episodes: int, simulations: int, save_interval: int, evaluate_during: bool = False,
              n_pretrained_episodes=0, increasing_mcts=True, min_simulations=300, max_simulations=1000):
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
                    # we gradually increase amount of searches to max_value when half the board is filled
                    budget = math.floor((max_simulations - min_simulations) * min(1, (2 * n_moves / self.k ** 2))) \
                             + min_simulations
                    print(budget)
                else:
                    budget = simulations

                best_action, distribution = search_tree.search(budget)

                n_moves += 1
                board_state = root.state[0]

                if self.contains_bridges:
                    board_state = mark_bridge_endpoints(board_state)
                    case = self.anet.vectorize_case((board_state, root.state[1]), distribution)
                else:
                    case = self.anet.vectorize_case(root.state, distribution)

                generated_cases = generate_more_cases(case=case, k=self.k, contains_bridges=self.contains_bridges)
                for case in generated_cases:
                    self.replay_buffer.add_case(case)

                self.state_manager.make_move(best_action)
                root = root.children[best_action]
                root.parent = None

            # train ANET

            minibatch = self.replay_buffer.get_minibatch(batch_size=256)
            self.anet.train(minibatch, epochs=5)

            # starting_player = Player.BLUE if starting_player == Player.RED else Player.RED

            end = time.perf_counter()
            print(f'time: {end - start:.1f}s')

            if episode % 3 == 0 and evaluate_during:
                if evaluate_network(evaluation_network, self.anet, self.k, n_games=400, threshold=0.50,
                                    random_move=True):
                    evaluation_network = copy.deepcopy(self.anet)
                else:
                    self.anet = evaluation_network

            if episode % save_interval == 0:
                # save ANET parameters
                self.anet.save_parameters(
                    f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_{episode + n_pretrained_episodes}.weights.h5')

        self.anet.plot_history('loss')
        self.anet.plot_history('accuracy')


if __name__ == "__main__":
    game_manager = RLTrainer(k=4, anet_config_name='cnn', contains_bridges=False, padding=2, cnn=True)
    game_manager.train(episodes=200,
                       simulations=400,
                       save_interval=10,
                       evaluate_during=False,
                       increasing_mcts=True,
                       min_simulations=100,
                       max_simulations=500)

    # game_manager = RLTrainer(k=4, anet_config_name='anet', contains_bridges=False, padding=0, cnn=False)
    # game_manager.train(episodes=200,
    #                    simulations=400,
    #                    save_interval=10,
    #                    evaluate_during=False,
    #                    increasing_mcts=True,
    #                    min_simulations=100,
    #                    max_simulations=500)

    # trainer = RLTrainer(k=4, anet_config_name='anet')
    # trainer.train(episodes=200, simulations=500, save_interval=40)
    # trainer = RLTrainer(k=4, anet_config_name='jespee_anet')
    # trainer.train(episodes=200, simulations=500, save_interval=10)
