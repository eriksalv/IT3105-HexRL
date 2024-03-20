from random import sample
import numpy as np


class ReplayBuffer:
    
    def __init__(self) -> None:
        self.cases: list[tuple[np.ndarray, np.ndarray]] = []

    def add_case(self, case: tuple[np.ndarray, np.ndarray]) -> None:
        self.cases.append(case)

    def get_minibatch(self, batch_size=64) -> list[tuple[np.ndarray, np.ndarray]]:
        if batch_size > len(self.cases):
            return self.cases

        return sample(self.cases, batch_size)
    
    def clear(self):
        self.cases: list[tuple[np.ndarray, np.ndarray]] = []
