import numpy as np

from .anet import ActorNetwork


class ConvActorNet(ActorNetwork):
    def __init__(self, k: int, anet_config_name: str, saved_weights_file=None, contains_bridges=False, padding=2,
                 show_summary=False, is_dual_network=False):
        super().__init__(k, anet_config_name, saved_weights_file, contains_bridges, padding, is_dual_network)
        self.channels = 3 + 1 + (2 if contains_bridges else 0)  # 3 channels per color + 1 for current player (constant)
        self.input_shape = (self.k + self.padding * 2, self.k + self.padding * 2, self.channels)
        self.model = self.build_model(show_summary)

        if saved_weights_file is not None:
            self.model.load_weights(f'./trained_networks/{saved_weights_file}.weights.h5')

    def prepare_input_cases(self, inputs: list[np.ndarray]) -> np.ndarray:
        board_state_inputs = np.array([case[:-1].reshape(self.k, self.k) for case in inputs])
        current_player_inputs = np.stack([np.full((self.k, self.k), case[-1] - 1) for case in inputs])

        input_cases = np.zeros(shape=(len(inputs), *self.input_shape), dtype=int)
        unpadded_idx = slice(self.padding, -self.padding if self.padding > 0 else None)

        if self.padding > 0:
            # pad with red (0) on top and bottom
            input_cases[:, :self.padding, :, :] = 0
            input_cases[:, -self.padding:, :, :] = 0

            # pad with blue (1) on left and right
            input_cases[:, unpadded_idx, :self.padding, :] = 1
            input_cases[:, unpadded_idx, -self.padding:, :] = 1

        # empty, red, and blue pieces
        input_cases[:, unpadded_idx, unpadded_idx, 0] = board_state_inputs == 0
        input_cases[:, unpadded_idx, unpadded_idx, 1] = np.logical_or(board_state_inputs == 1, board_state_inputs == 3)
        input_cases[:, unpadded_idx, unpadded_idx, 2] = np.logical_or(board_state_inputs == 2, board_state_inputs == 4)

        # current player (constant value)
        input_cases[:, unpadded_idx, unpadded_idx, 3] = current_player_inputs

        if self.contains_bridges:
            # red and blue bridge endpoints
            input_cases[:, unpadded_idx, unpadded_idx, 4] = board_state_inputs == 3
            input_cases[:, unpadded_idx, unpadded_idx, 5] = board_state_inputs == 4

        return input_cases
