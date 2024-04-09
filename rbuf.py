from collections import deque

import numpy as np


class ReplayBuffer:

    def __init__(self, max_size=256) -> None:
        self.cases: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self.max_size = max_size

    def add_case(self, case: tuple[np.ndarray, np.ndarray]) -> None:
        # Remove the oldest case if size of cases is too large
        if len(self.cases) > self.max_size:
            self.cases.popleft()

        self.cases.append(case)

    def get_minibatch(self, batch_size=128) -> list[tuple[np.ndarray, np.ndarray]]:
        if batch_size > len(self.cases):
            return list(self.cases)

        # Recent cases should be given higher weight
        indices = np.arange(len(self.cases))
        sample_weights = (indices + 1) / np.sum(indices + 1)
        indices = np.random.choice(indices, size=batch_size, p=sample_weights, replace=False)
        return [self.cases[i] for i in indices]
