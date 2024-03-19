import json
from anet import ActorNetwork  # Import your ANET class
from hex import HexStateManager
class Simulation:
    def __init__(self, anet1_weights_file, anet2_weights_file):
        # Load ANETs from saved weights files
        with open('config/anet.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.anet1 = ActorNetwork(3, config)
        self.anet1.model.load_weights('./trained_networks/'+anet1_weights_file)
        
        self.anet2 = ActorNetwork(3, config)
        self.anet2.model.load_weights('./trained_networks/'+ anet2_weights_file)

    def play_game(self, starting_player = 1, show_board = False):
        hsm = HexStateManager(3)

        hsm.new_game()
        
        current_player = starting_player  
        
        while not hsm.is_final():
            if current_player == 1:
                action_idx = self.anet1.get_action(hsm.board, 1)
                current_player = 2
            else:
                action_idx = self.anet2.get_action(hsm.board,2)
                current_player = 1
            action = hsm.convert_to_move(action_idx)
            
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
    
    def simulate_games(self, n_games = 5):
        win_dict = {1: 0, 2: 0}
        starting_player = 1
        for _ in range(n_games):
            winner = self.play_game(starting_player=starting_player, show_board=True)

            if starting_player == 1:
                starting_player = 2
            else:
                starting_player = 1

            win_dict[winner] +=1
        print(win_dict)
if __name__ == '__main__':
    sim = Simulation(anet1_weights_file = 'anet_20.weights.h5', anet2_weights_file = 'anet_100.weights.h5')
    sim.simulate_games()