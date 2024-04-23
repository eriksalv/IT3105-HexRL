import math
import time

import numpy as np

from games.hex import HexStateManager, Player, mark_bridge_endpoints
from mcts import MCTS, Node
from networks.basic_anet import BasicActorNet
from networks.conv_anet import ConvActorNet
from rbuf import ReplayBuffer
from utils import generate_more_cases, evaluate_network


class RLTrainer:
    def __init__(self, k: int, anet_config_name: str, contains_bridges=False, padding=0, cnn=False,
                 n_pretrained_episodes=0, use_dual_network=False):
        self.k = k
        self.anet_config_name = anet_config_name
        self.contains_bridges = contains_bridges
        self.padding = padding
        self.state_manager = HexStateManager(self.k)
        self.replay_buffer = ReplayBuffer(max_size=2048)
        self.n_pretrained_episodes = n_pretrained_episodes

        saved_weights_file = f'{anet_config_name}_{k}x{k}_{n_pretrained_episodes}' if n_pretrained_episodes > 0 else None

        if cnn:
            self.anet = ConvActorNet(self.k, self.anet_config_name, contains_bridges=contains_bridges, padding=padding,
                                     show_summary=True, saved_weights_file=saved_weights_file,
                                     is_dual_network=use_dual_network)
            self.eval_net = ConvActorNet(self.k, self.anet_config_name, saved_weights_file=saved_weights_file,
                                         contains_bridges=contains_bridges, padding=self.padding,
                                         is_dual_network=use_dual_network)
        else:
            self.anet = BasicActorNet(self.k, self.anet_config_name, saved_weights_file=saved_weights_file,
                                      contains_bridges=contains_bridges, show_summary=True)
            self.eval_net = BasicActorNet(self.k, self.anet_config_name, saved_weights_file=saved_weights_file,
                                          contains_bridges=contains_bridges)

    def train(self, episodes: int, save_interval: int, evaluate_during: bool = False, min_simulations=300,
              max_simulations=1000):
        if self.n_pretrained_episodes == 0:
            # Save untrained network
            self.anet.save_parameters(f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_0.weights.h5')

        starting_player = Player.RED
        sigma = 1.0

        for episode in range(1 + self.n_pretrained_episodes, episodes + 1):
            print(f'Episode: {episode}')

            start = time.perf_counter()

            # initialize new board
            self.state_manager.new_game(starting_player=starting_player)
            s_init = self.state_manager.get_state()
            root = Node(s_init)
            root.init_actions_and_values(self.state_manager.get_legal_moves())

            n_moves = 0
            board_states = []
            distributions = []

            if self.anet.is_dual_network:
                # gradually increase prob. of using critic eval with number of completed episodes
                sigma = max(0.05, 1.0 - 3 * episode / episodes)

            # Actual game
            while not self.state_manager.is_final():
                # decrease exploration bonus closer to final state
                c = np.sqrt(2) * (1 - n_moves / self.k ** 2) if n_moves > self.k else np.sqrt(2)
                search_tree = MCTS(self.state_manager, root, self.eval_net if evaluate_during else self.anet,
                                   expansion_threshold=20, c=c)

                if max_simulations and min_simulations:
                    # we gradually increase amount of searches to max_value when half the board is filled
                    budget = math.floor((max_simulations - min_simulations) * min(1, (2 * n_moves / self.k ** 2))) \
                             + min_simulations
                else:
                    budget = max_simulations

                print(f'Move {n_moves + 1}: running {budget} simulations...')

                best_action, distribution = search_tree.search(budget, sigma=sigma)

                n_moves += 1
                board_state = root.state[0]

                board_states.append(board_state)
                distributions.append(distribution)

                self.state_manager.make_move(best_action)
                root = root.children[best_action]
                root.parent = None

            reward = 1 if self.state_manager.winner == Player.RED else -1

            # Construct training cases
            for board_state, distribution in zip(board_states, distributions):
                if self.contains_bridges:
                    board_state = mark_bridge_endpoints(board_state)
                    case = self.anet.vectorize_case((board_state, root.state[1]), distribution, reward)
                else:
                    case = self.anet.vectorize_case(root.state, distribution, reward)

                generated_cases = generate_more_cases(case=case, k=self.k, contains_bridges=self.contains_bridges)
                for case in generated_cases:
                    self.replay_buffer.add_case(case)

            # train ANET
            if len(self.replay_buffer.cases) > self.replay_buffer.max_size / 2:
                for _ in range(5):
                    minibatch = self.replay_buffer.get_minibatch(batch_size=128, use_sample_weights=False)
                    self.anet.train(minibatch, epochs=1)
            else:
                minibatch = self.replay_buffer.get_minibatch(batch_size=512, use_sample_weights=True)
                self.anet.train(minibatch, epochs=1)

            end = time.perf_counter()
            print(f'time: {end - start:.1f}s')

            if episode % 3 == 0 and evaluate_during:
                if evaluate_network(self.eval_net, self.anet, self.k, n_games=50, threshold=0.50,
                                    random_move=True):
                    self.eval_net.model.set_weights(self.anet.model.get_weights())

            if episode % save_interval == 0:
                # save ANET parameters
                self.anet.save_parameters(
                    f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_{episode}.weights.h5')

        self.anet.plot_history('loss')
        self.anet.plot_history('accuracy')


if __name__ == "__main__":
    # game_manager = RLTrainer(k=7, anet_config_name='oht_cnn2', contains_bridges=True, padding=2, cnn=True,
    #                          use_dual_network=True)
    # game_manager.train(episodes=200,
    #                    simulations=400,
    #                    save_interval=10,
    #                    evaluate_during=True,
    #                    increasing_mcts=True,
    #                    min_simulations=200,
    #                    max_simulations=800)

    game_manager = RLTrainer(k=7, anet_config_name='oht_cnn', contains_bridges=False, padding=2, cnn=True,
                             use_dual_network=True)
    game_manager.train(episodes=200,
                       save_interval=10,
                       evaluate_during=True,
                       max_simulations=500)

    # trainer = RLTrainer(k=4, anet_config_name='anet')
    # trainer.train(episodes=200, simulations=500, save_interval=40)
    # trainer = RLTrainer(k=4, anet_config_name='jespee_anet')
    # trainer.train(episodes=200, simulations=500, save_interval=10)
