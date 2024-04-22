import numpy as np

from .anet import ActorNetwork


class BasicActorNet(ActorNetwork):
    def __init__(self, k: int, anet_config_name: str, saved_weights_file=None, contains_bridges=False,
                 show_summary=False):
        super().__init__(k, anet_config_name, saved_weights_file, contains_bridges, padding=0)
        self.input_shape = (k ** 2 + 1,)  # one unit per board node + 1 for current player
        self.model = self.build_model(show_summary=show_summary)

        if saved_weights_file is not None:
            self.model.load_weights(f'./trained_networks/{saved_weights_file}.weights.h5')

    def prepare_input_cases(self, inputs: list[np.ndarray]) -> np.ndarray:
        return np.stack([case for case in inputs])
