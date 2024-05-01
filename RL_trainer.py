import json
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
    def __init__(self, config_name):
        with open('./rl_configs/' + config_name, "r") as file:
            config = json.load(file)

        self.k = config.get("k", 7)
        self.episodes = config.get("episodes", 400)
        self.max_simulations = config.get("max_simulations", 1200)
        self.min_simulations = config.get("min_simulations", 300)
        self.save_interval = config.get("save_interval", 10)
        self.expansion_threshold = config.get("expansion_threshold", None)
        self.anet_config_name = config.get("anet_config_name", "cnn")
        self.contains_bridges = config.get("contains_bridges", False)
        self.padding = config.get("padding", 0)
        self.replay_buffer = ReplayBuffer(max_size=config.get("max_rbuf_size", 2048))

        self.n_pretrained_episodes = config.get("n_pretrained_episodes", 0)
        self.cnn = config.get("cnn", False)
        self.use_dual_network = config.get("use_dual_network", False)

        self.batch_size = config.get("batch_size", 512)
        self.evaluate_during = config.get("evaluate_during", False)

        self.show_board = config.get("show_board", False)

        saved_weights_file = f'{self.anet_config_name}_{self.k}x{self.k}_{self.n_pretrained_episodes}' if self.n_pretrained_episodes > 0 else None

        if self.cnn:
            self.anet = ConvActorNet(self.k, self.anet_config_name, contains_bridges=self.contains_bridges,
                                     padding=self.padding,
                                     show_summary=True, saved_weights_file=saved_weights_file,
                                     is_dual_network=self.use_dual_network)
            self.eval_net = ConvActorNet(self.k, self.anet_config_name, saved_weights_file=saved_weights_file,
                                         contains_bridges=self.contains_bridges, padding=self.padding,
                                         is_dual_network=self.use_dual_network)
        else:
            self.anet = BasicActorNet(self.k, self.anet_config_name, saved_weights_file=saved_weights_file,
                                      contains_bridges=self.contains_bridges, show_summary=True)
            self.eval_net = BasicActorNet(self.k, self.anet_config_name, saved_weights_file=saved_weights_file,
                                          contains_bridges=self.contains_bridges)

    def train(self):
        if self.n_pretrained_episodes == 0:
            # Save untrained network
            self.anet.save_parameters(f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_0.weights.h5')

        starting_player = Player.RED
        sigma = 1.0

        for episode in range(1 + self.n_pretrained_episodes, self.episodes + 1):
            print(f'Episode: {episode}')

            start = time.perf_counter()

            # initialize new board
            state_manager = HexStateManager(self.k)
            state_manager.new_game(starting_player=starting_player)
            s_init = state_manager.get_state()
            root = Node(s_init)
            root.init_actions_and_values(state_manager.get_legal_moves())

            n_moves = 0
            board_states = []
            distributions = []

            if self.anet.is_dual_network:
                # gradually increase prob. of using critic eval with number of completed episodes
                sigma = max(0.05, 1.0 - 2 * episode / self.episodes)

            # Actual game
            while not state_manager.is_final():
                # decrease exploration bonus closer to final state
                c = np.sqrt(2) * (1 - n_moves / self.k ** 2) if n_moves > self.k else np.sqrt(2)
                search_tree = MCTS(state_manager, root, self.eval_net if self.evaluate_during else self.anet,
                                   expansion_threshold=self.expansion_threshold, c=c)

                if self.max_simulations and self.min_simulations:
                    # we gradually increase amount of searches to max_value when half the board is filled
                    budget = math.floor(
                        (self.max_simulations - self.min_simulations) * min(1, (2 * n_moves / self.k ** 2))) \
                             + self.min_simulations
                else:
                    budget = self.max_simulations

                print(f'Move {n_moves + 1}: running {budget} simulations...')

                best_action, distribution = search_tree.search(budget, sigma=sigma)

                n_moves += 1
                board_state = root.state[0]

                board_states.append(board_state)
                distributions.append(distribution)

                state_manager.make_move(best_action)
                root = root.children[best_action]
                root.parent = None

                state_manager.add_frame()

            reward = 1 if state_manager.winner == Player.RED else -1

            cases = [[], [], [], []]

            # Construct training cases
            for board_state, distribution in zip(board_states, distributions):
                if self.contains_bridges:
                    board_state = mark_bridge_endpoints(board_state)
                    case = self.anet.vectorize_case((board_state, root.state[1]), distribution, reward)
                else:
                    case = self.anet.vectorize_case(root.state, distribution, reward)

                generated_cases = generate_more_cases(case=case, k=self.k, contains_bridges=self.contains_bridges)

                for i in range(4):
                    cases[i].append(generated_cases[i])

            # train ANET
            for i in range(4):
                for case in cases[i]:
                    self.replay_buffer.add_case(case)

            minibatch = self.replay_buffer.get_minibatch(batch_size=self.batch_size)
            self.anet.train(minibatch, epochs=1)
            
            end = time.perf_counter()
            print(f'time: {end - start:.1f}s')

            if episode % 3 == 0 and self.evaluate_during:
                if evaluate_network(self.eval_net, self.anet, self.k, n_games=50, threshold=0.50,
                                    random_move=True):
                    self.eval_net.model.set_weights(self.anet.model.get_weights())

            if episode % self.save_interval == 0:
                # save ANET parameters
                self.anet.save_parameters(
                    f'./trained_networks/{self.anet_config_name}_{self.k}x{self.k}_{episode}.weights.h5')

                if self.show_board:
                    state_manager.show_board(filename=f'training/{episode}', animate=True)

        self.anet.plot_history('loss')
        self.anet.plot_history('accuracy')


if __name__ == "__main__":
    game_manager = RLTrainer("live_demo.json")
    game_manager.train()
