import json
from abc import ABC, abstractmethod
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as ks


class ActorNetwork(ABC):
    def __init__(self, k: int, anet_config_name: str, saved_weights_file=None, contains_bridges=False, padding=0,
                 is_dual_network=False):
        self.k = k
        self.padding = padding
        self.contains_bridges = contains_bridges
        self.is_dual_network = is_dual_network
        self.output_size = k ** 2
        self.history = {'loss': [], 'accuracy': []}
        self.anet_config_name = anet_config_name
        self.saved_weights_file = saved_weights_file
        self.model = None
        self.input_shape = None

    @abstractmethod
    def prepare_input_cases(self, inputs: list[np.ndarray]) -> np.ndarray:
        pass

    def build_model(self, show_summary=False) -> ks.models.Model:
        with open(f'config/{self.anet_config_name}.json', 'r', encoding='utf-8') as f:
            anet_config = json.load(f)

        input_layer = ks.layers.Input(shape=self.input_shape)

        x = input_layer

        for layer_conf in anet_config['hidden_layers']:
            layer_class = getattr(ks.layers, layer_conf['type'])
            layer_conf.pop('type', None)
            x = layer_class(**layer_conf)(x)

        distribution_head = x

        for distribution_layer in anet_config['distribution_head']:
            layer_class = getattr(ks.layers, distribution_layer['type'])
            distribution_layer.pop('type', None)
            distribution_head = layer_class(**distribution_layer)(distribution_head)

        outputs = {'distribution': ks.layers.Dense(self.output_size, activation='softmax', name='distribution')(
            distribution_head)}
        loss = {'distribution': anet_config['loss']}
        metrics = {'distribution': 'accuracy'}

        if self.is_dual_network:
            evaluation_head = x

            for evaluation_layer in anet_config['evaluation_head']:
                layer_class = getattr(ks.layers, evaluation_layer['type'])
                evaluation_layer.pop('type', None)
                evaluation_head = layer_class(**evaluation_layer)(evaluation_head)

            outputs['evaluation'] = ks.layers.Dense(1, activation='tanh', name='evaluation')(evaluation_head)
            loss['evaluation'] = 'mse'
            metrics['evaluation'] = 'mae'

        model = ks.models.Model(inputs=input_layer, outputs=outputs)

        optimizer = getattr(ks.optimizers, anet_config['optimizer'])(learning_rate=anet_config['learning_rate'])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if show_summary:
            model.summary()

        return model

    def train(self, minibatch: list[tuple[np.ndarray, np.ndarray]], verbose=1, epochs=1, batch_size=None) -> None:
        """
        Trains the network on the cases in the minibatch.
        Assumes that the minibatch is a list of tuples containing
        the vectorized board state as the first element, and the
        target output distribution as the second
        """
        input_cases = self.prepare_input_cases([case[0] for case in minibatch])
        output_cases = np.stack([case[1] for case in minibatch])

        if self.is_dual_network:
            y = {'distribution': output_cases[:, :-1], 'evaluation': output_cases[:, -1]}
            accuracy = 'distribution_accuracy'
        else:
            y = output_cases
            accuracy = 'accuracy'

        history = self.model.fit(input_cases, y, verbose=verbose, epochs=epochs,
                                 batch_size=batch_size if batch_size else len(minibatch))

        self.history['loss'].extend(history.history['loss'])
        self.history['accuracy'].extend(history.history[accuracy])

    def get_action(self, board_state: np.ndarray, current_player: int) -> tuple[int, int]:
        """
        Vectorizes board state with current player, and feeds the
        input vector into the model to produce an output distribution,
        and selects the move with the highest probability.
        """
        input_case = self.prepare_input_cases([self.vectorize_state(board_state, current_player)])

        output = self.model(input_case)['distribution'].numpy()[0]
        output = self.renormalize_output(output, board_state)

        action_idx = int(np.argmax(output))
        return self.convert_to_move(action_idx)

    def get_eval(self, board_state: np.ndarray, current_player: int) -> float:
        """
        Get critic evaluation for given position
        """
        input_case = self.prepare_input_cases([self.vectorize_state(board_state, current_player)])
        output = self.model(input_case)['evaluation'].numpy()[0][0]
        return output

    def save_parameters(self, filepath):
        """
        Save the parameters (weights) of the ANET model.

        :param filepath: File path where the model weights will be saved.
        """
        self.model.save_weights(filepath)
        print("ANET parameters saved successfully.")

    def vectorize_case(self, state: tuple[np.ndarray, int], distribution: [tuple[int, int], float], reward=None) -> \
            tuple[np.ndarray, np.ndarray]:
        return self.vectorize_state(state[0], state[1]), self.vectorize_distribution(distribution, reward)

    def vectorize_distribution(self, distribution: dict[tuple[int, int], float], reward=None) -> np.ndarray:
        full_distribution = [0.0 for _ in range(self.k ** 2)]
        for action, prob in distribution.items():
            full_distribution[action[0] * self.k + action[1]] = prob

        if reward is not None and self.is_dual_network:
            full_distribution.append(reward)

        return np.array(full_distribution)

    @staticmethod
    def vectorize_state(board_state: np.ndarray, current_player: int) -> np.ndarray:
        return np.append(board_state, current_player).astype(int)

    @staticmethod
    def renormalize_output(output: np.ndarray, board_state: np.ndarray) -> np.ndarray:
        illegal_moves = np.where(np.logical_or(board_state.flatten() == 1, board_state.flatten() == 2))
        output[illegal_moves] = 0

        return output / np.sum(output)

    def convert_to_move(self, action_idx):
        row = action_idx // self.k
        col = action_idx % self.k
        return row, col

    def plot_history(self, metric: Literal['loss', 'accuracy'] = 'loss'):
        plt.figure(figsize=(16, 9))

        plt.plot(self.history[metric])
        plt.grid()

        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        plt.savefig(f'plots/{self.anet_config_name}_{self.k}x{self.k}_{metric}_history.png')
