import numpy as np

from networks.basic_anet import BasicActorNet
from hex_client.ActorClient import ActorClient

actor = BasicActorNet(k=7, anet_config_name='jespee_anet',
                      saved_weights_file='jespee_anet_7x7_bridges_55', contains_bridges=True)


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
