import json
from anet import ActorNetwork  # Import your ANET class
from hex import HexStateManager


class Simulation:
    def __init__(self, anet1_weights_file, anet2_weights_file, k):
        self.k = k
        # Load ANETs from saved weights files
        self.anet1 = ActorNetwork(self.k, 'anet')
        self.anet1.model.load_weights(
            './trained_networks/' + anet1_weights_file)

        self.anet2 = ActorNetwork(self.k, 'anet')
        self.anet2.model.load_weights(
            './trained_networks/' + anet2_weights_file)

    def play_game(self, starting_player=1, show_board=False):
        hsm = HexStateManager(self.k)

        hsm.new_game()

        current_player = starting_player

        while not hsm.is_final():
            if current_player == 1:
                action = self.anet1.get_action(hsm.board, 1)
                current_player = 2
            else:
                action = self.anet2.get_action(hsm.board, 2)
                current_player = 1

            hsm.make_move(action)

        if show_board:
            hsm.show_board()

        if starting_player == 1:
            winner = hsm.winner.value
        else:
            if hsm.winner.value == 1:
                winner = 2
            else:
                winner = 1

        return winner

    def simulate_games(self, n_games=6, show_board=False):
        win_dict = {1: 0, 2: 0}
        starting_player = 1
        for _ in range(n_games):
            winner = self.play_game(
                starting_player=starting_player, show_board=show_board)

            if starting_player == 1:
                starting_player = 2
            else:
                starting_player = 1

            win_dict[winner] += 1
        # print(win_dict)
        return win_dict


def simulate_tourney(epochs=[1, 10, 20, 30, 40, 50], k=4, show_board=False):
    for i in epochs:
        for j in epochs:
            if i == j:
                continue
            sim = Simulation(
                anet1_weights_file=f'anet_{i}.weights.h5', anet2_weights_file=f'anet_{j}.weights.h5', k=k)
            res = sim.simulate_games()
            print(i, j)
            print(res)


if __name__ == '__main__':
    simulate_tourney(k=3)
