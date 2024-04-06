import numpy as np

from anet import ActorNetwork
from ActorClient import ActorClient

actor = ActorNetwork(k=7, anet_config='anet', saved_weights_file='anet_50_7x7')


class MyClient(ActorClient):
    # Import and override the `handle_get_action` hook in ActorClient
    def handle_get_action(self, state):
        current_player = state[0]
        board_state = np.array(state[1:]).reshape((actor.k, actor.k))

        row, col = actor.get_action(board_state, current_player)
        return row, col


# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    # Set env var IT3105_AUTH to authenticate automatically
    # Change qualify to True to attempt qualification
    # Change mode to 'league' to play in ranked league
    client = MyClient(qualify=False)
    client.run(mode='qualifiers')
