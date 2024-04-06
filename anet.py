import json
from tensorflow import keras as ks
import numpy as np
from hex import HexStateManager


class ActorNetwork:
    def __init__(self, k: int, anet_config_name: str, saved_weights_file=None) -> None:
        self.k = k
        self.input_size = k ** 2 + 1  # one unit per board node + 1 for current player
        self.output_size = k ** 2
        self.model = self.build_model(anet_config_name)

        if saved_weights_file is not None:
            self.model.load_weights(
                f'./trained_networks/{saved_weights_file}.weights.h5')

    def build_model(self, anet_config_name: str) -> ks.models.Model:
        """
        Builds and compiles a keras Model according to provided config
        """
        with open(f'config/{anet_config_name}.json', 'r', encoding='utf-8') as f:
            anet_config = json.load(f)

        input_layer = ks.layers.Input(shape=(self.input_size,))
        x = input_layer

        hidden_layers = anet_config['hidden_layers']

        for layer in hidden_layers:
            x = ks.layers.Dense(
                units=layer['units'],
                activation=layer['activation'])(x)

        output_layer = ks.layers.Dense(
            units=self.output_size,
            activation='softmax')(x)

        model = ks.models.Model(inputs=input_layer, outputs=output_layer)

        opt = eval('ks.optimizers.' + anet_config['optimizer'])

        model.compile(
            optimizer=opt(
                learning_rate=anet_config['learning_rate'],
                **anet_config['optimizer_kwargs']),
            loss=anet_config['loss']
        )
        # model.summary()

        return model

    def train(self, minibatch: list[tuple[np.ndarray, np.ndarray]], verbose=1) -> None:
        """
        Trains the network on the cases in the minibatch (1 epoch).
        Assumes that the minibatch is a list of tuples containing
        the vectorized board state as the first element, and the
        target output distribution as the second
        """
        input_cases = np.vstack([case[0] for case in minibatch])
        output_cases = np.vstack([case[1] for case in minibatch])

        self.model.fit(input_cases, output_cases, verbose=verbose)

    def get_action(self, board_state: np.ndarray, current_player: int) -> tuple[int, int]:
        """
        Vectorizes board state with current player, and feeds the
        input vector into the model to produce an output distribution,
        and selects the move with the highest probability.
        """
        input_case = self.vectorize_state(board_state, current_player)
        output = self.model(input_case[np.newaxis]).numpy().flatten()
        output = self.renormalize_output(output, board_state)
        action_idx = int(np.argmax(output))

        # Convert flat action index to row and col in 2D board
        row = action_idx // self.k
        col = action_idx % self.k
        return row, col

    def save_parameters(self, filepath):
        """
        Save the parameters (weights) of the ANET model.

        :param filepath: File path where the model weights will be saved.
        """
        self.model.save_weights(filepath)
        print("ANET parameters saved successfully.")

    @staticmethod
    def vectorize_state(board_state: np.ndarray, current_player: int) -> np.ndarray:
        return np.append(board_state, current_player).astype(float)

    @staticmethod
    def renormalize_output(output: np.ndarray, board_state: np.ndarray) -> np.ndarray:
        illegal_moves = np.where(board_state.flatten() != 0)
        output[illegal_moves] = 0
        return output / np.sum(output)


if __name__ == '__main__':
    hsm = HexStateManager(3)
    hsm.make_move((0, 0))
    hsm.make_move((0, 1))
    hsm.make_move((1, 1))

    ANET = ActorNetwork(3, 'anet')
    action = ANET.get_action(hsm.board, hsm.current_player.value)
    print(action)
