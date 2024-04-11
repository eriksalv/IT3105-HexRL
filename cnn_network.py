import copy
import json
from matplotlib import pyplot as plt
from tensorflow import keras as ks
import numpy as np
from games.hex import HexStateManager, mark_bridges
from keras.callbacks import EarlyStopping 
import tensorflow as tf

from utils import add_padding



class CNN():
    def __init__(self, k: int, anet_config_name: str, saved_weights_file=None, contains_bridges = False, padding = False) -> None:
        self.true_k = k
        self.output_size = k ** 2
        self.padding = padding
        if padding:
            self.k = k+2
            self.board_shape = (k+2, k+2, 1)  # one unit per board node + 1 for current player
        else:
            self.board_shape = (k, k, 1)
            self.k = k

        self.player_shape = (1,)
        
        
        self.model = self.build_model(anet_config_name)
        self.history = []
        self.contains_bridges = contains_bridges

        if saved_weights_file is not None:
            self.model.load_weights(
                f'./trained_networks/{saved_weights_file}.weights.h5')
    
    def build_model(self, anet_config_name: dict) -> ks.models.Model:
        """
        Builds and compiles a keras Model according to provided config
        """
        board_input = ks.layers.Input(shape=self.board_shape, name='board_input')
        player_input = ks.layers.Input(shape=self.player_shape, name='player_input')

        with open(f'config/{anet_config_name}.json', 'r', encoding='utf-8') as f:
            anet_config = json.load(f)
        
       
        board_input = ks.layers.Input(shape=self.board_shape, name='board_input')
        player_input = ks.layers.Input(shape=self.player_shape, name='player_input')

        x = board_input

        
        for layer_conf in anet_config['hidden_layers']:
            current_type = layer_conf['type']
            layer_class = getattr(ks.layers, layer_conf['type'])

            layer_conf.pop('type', None)
            x = layer_class(**layer_conf)(x)
            if current_type == 'Flatten':
                x = ks.layers.concatenate([x,player_input])


        

        
        output = ks.layers.Dense(self.output_size, activation='softmax', name='output')(x)

       
        model = ks.models.Model(inputs=[board_input, player_input], outputs=output)

       
        optimizer = getattr(ks.optimizers, anet_config['optimizer'])(learning_rate=anet_config['learning_rate'])
        model.compile(optimizer=optimizer, loss=anet_config['loss'], metrics=['accuracy'])

        model.summary()

        return model
    
    def train(self, minibatch: list[tuple[np.ndarray, np.ndarray]], verbose = 1, epochs = 1) -> None:
        """
        Trains the network on the cases in the minibatch (1 epoch).
        Assumes that the minibatch is a list of tuples containing
        the vectorized board state as the first element, and the
        target output distribution as the second
        """
       
        input_board_states = [case[0][:-1].reshape(self.k, self.k, 1) for case in minibatch]
        input_player_inputs = [case[0][-1] for case in minibatch]
  
        input_board_states = np.array(input_board_states)
      

        input_player_inputs = np.array(input_player_inputs)
        output_cases = np.vstack([case[1] for case in minibatch])
      
        self.model.fit([input_board_states,input_player_inputs], output_cases, verbose = verbose,
                        epochs = epochs, batch_size = len(minibatch))
        
    def get_action(self, board_state: np.ndarray, current_player: int) -> int:
        """
        Vectorizes board state with current player, and feeds the
        input vector into the model to produce an output distribution,
        and selects the move with the highest probability.
        """
        original_board_shape = copy.deepcopy(board_state)
        if self.padding:
            board_state = add_padding(board_state)
        if self.contains_bridges:
            board_state = mark_bridges(board_state)
        board_state = board_state.reshape((1, self.k, self.k, 1))
    
        # Create player input
        player_input = np.array([[current_player]])

        # Obtain the output distribution from the model
        output = self.model.predict([board_state, player_input], verbose=0)[0]
    
        output = self.renormalize_output(output, original_board_shape, contains_bridges=self.contains_bridges, padding = self.padding)
        action_idx = int(np.argmax(output))
        
        return self.convert_to_move(action_idx)

    def save_parameters(self, filepath):
        """
        Save the parameters (weights) of the ANET model.

        :param filepath: File path where the model weights will be saved.
        """
        self.model.save_weights(filepath)
        print("ANET parameters saved successfully.")

    def vectorize_case(self, state: tuple[np.ndarray, int], distribution: [tuple[int, int], float]) -> tuple[np.ndarray, np.ndarray]:
        return self.vectorize_state(state[0], state[1]), self.vectorize_distribution(distribution)

    def vectorize_distribution(self, distribution: dict[tuple[int, int], float]) -> np.ndarray:
        full_distribution = [0.0 for _ in range(self.true_k ** 2)]
        for action, prob in distribution.items():
            full_distribution[action[0] * self.true_k + action[1]] = prob

        return np.array(full_distribution)

    @staticmethod
    def vectorize_state(board_state: np.ndarray, current_player: int) -> np.ndarray:
        return np.append(board_state, current_player).astype(float)

    @staticmethod
    def renormalize_output(output: np.ndarray, board_state: np.ndarray, contains_bridges = False, padding = False) -> np.ndarray:
        
        if contains_bridges:
            illegal_moves = np.where(np.logical_or(board_state.flatten() == 1, board_state.flatten() == 2))
        else:
            illegal_moves = np.where(board_state.flatten() != 0)
        output[illegal_moves] = 0
        
        return output / np.sum(output)

    def convert_to_move(self, action_idx):
        row = action_idx // self.true_k
        col = action_idx % self.true_k
        return row, col

    def plot_history(self):
        plt.figure(figsize=(10, 5))

        plt.plot(self.history)
        plt.grid()

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        plt.savefig(f'plots/anet_{self.true_k}x{self.true_k}_training_progress.png')