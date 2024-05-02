import numpy as np
class Nim():

    def __init__(self, n, k, starting = True, n_moves = 0) -> None:
        self.k = k
        self.n = n
        self.starting = starting
        self.n_moves = n_moves

    def get_possible_states(self):
        return [Nim(self.n - i, self.k, self.starting, self.n_moves+1) for i in range(1, min(self.k+1, self.n+1))]


    def is_win(self):
        return self.n == 0 and ((self.starting and self.n_moves % 2 == 1) or (not self.starting and self.n_moves % 2 == 0))
                                 

    def my_move(self, k):
        self.n_moves +=1
        self.n -=k
    
    def opponent_move(self, k, random = False):
        self.n_moves +=1
        if random:
            max_k = min(k, self.n)
            self.n -= np.random.choice(max_k)
        else:
            self.n-=k

    def get_state(self):
        print("N moves: ", self.n_moves)
        print("N remaining pieces", self.n)
        #return self.n_moves, self.n
    
    def is_final(self):
        return self.n == 0
    
if __name__ == "__main__":
    nim = Nim(7, 3)
    print(len(nim.get_possible_states()))
    nim.my_move(3)
    nim.opponent_move(3)
    print(len(nim.get_possible_states()))
    nim.my_move(1)
    print(nim.is_win())

    nim = Nim(7, 3)
    nim.my_move(2)
    nim.opponent_move(3)
    nim.my_move(1)
    print(nim.is_win())
    nim.opponent_move(1)
    print(nim.is_final())
    print(nim.is_win())

    nim = Nim(7, 3)
    child_states = nim.get_possible_states()
    for cs in child_states:
        cs.get_state()