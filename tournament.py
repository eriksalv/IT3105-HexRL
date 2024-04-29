import json
from games.hex import HexStateManager, Player
from networks.basic_anet import BasicActorNet
from networks.conv_anet import ConvActorNet


class Simulation:
    def __init__(self, anet: str, anet1_weights_file: str, anet2_weights_file: str, k: int, cnn=False, dual=False):
        self.k = k
        # Load ANETs from saved weights files
        if cnn:
            self.anet1 = ConvActorNet(self.k, anet, anet1_weights_file, padding=2, is_dual_network=dual,
                                      contains_bridges=True)
            self.anet2 = ConvActorNet(self.k, anet, anet2_weights_file, padding=2, is_dual_network=dual,
                                      contains_bridges=True)
        else:
            self.anet1 = BasicActorNet(self.k, anet, anet1_weights_file)
            self.anet2 = BasicActorNet(self.k, anet, anet2_weights_file)

    def play_game(self, starting_player, show_board=False):
        hsm = HexStateManager(self.k)
        hsm.new_game(starting_player=starting_player)

        # To make games not identical, start by playing a random move in the middle horizontal line of the hex board
        hsm.make_random_starting_move()

        while not hsm.is_final():
            if hsm.current_player == Player.RED:
                action = self.anet1.get_action(
                    hsm.board, hsm.current_player.value)
            else:
                action = self.anet2.get_action(
                    hsm.board, hsm.current_player.value)

            hsm.make_move(action)

        if show_board:
            hsm.show_board()

        return hsm.winner.value

    def simulate_games(self, n_games=25, show_board=False):
        win_dict = {1: 0, 2: 0}
        starting_player = Player.RED
        for _ in range(n_games):
            winner = self.play_game(
                starting_player=starting_player, show_board=show_board)
            win_dict[winner] += 1

            # alternate if player 1 (red) or player 2 (blue) starts
            starting_player = Player.BLUE if starting_player == Player.RED else Player.RED

        return win_dict


def simulate_tourney(config_name):
    with open('./tournament_configs/' + config_name, "r") as file:
            config = json.load(file)
    epochs = config.get("epochs", [0, 10, 20, 30, 40, 50])
    k = config.get("k", 4)
    anet = config.get("anet", "anet")
    cnn = config.get("cnn", True)
    dual = config.get("dual", True)
    n_games = config.get("n_games", 25)

    model_wins = {epoch: 0 for epoch in epochs}
    for i in epochs:
        for j in epochs:
            if j == i:
                continue

            sim = Simulation(anet, anet1_weights_file=f'{anet}_{k}x{k}_{i}', anet2_weights_file=f'{anet}_{k}x{k}_{j}',
                             k=k, cnn=cnn, dual=dual)
            res = sim.simulate_games(n_games=n_games)

            print(f'player 1: {i}, player 2: {j}')
            print(f'result: {res}')

            model_wins[i] += res[1]
            model_wins[j] += res[2]

    print(f'Total wins: {model_wins}')

    win_rates = {epoch: wins / (2 * n_games * (len(epochs) - 1))
                 for epoch, wins in model_wins.items()}
    print(f'Win rates: {win_rates}')


if __name__ == '__main__':
    simulate_tourney("live_demo.json")
