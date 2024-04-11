import random

import numpy as np

from anet import ActorNetwork
from games.hex import HexStateManager, Player


class Node:
    def __init__(self, state: tuple[np.ndarray, int], parent=None):
        self.state = state
        self.parent = parent
        self.selected_action = None
        self.children = {}
        self.visits = 0  # N(s)
        self.actions = {}  # N(s,a)
        self.value = {}  # Q(s,a)

    def expand(self, action: tuple[int, int], child_state: tuple[np.ndarray, int]):
        self.children[action] = Node(child_state, parent=self)

    def init_actions_and_values(self, legal_moves: list[tuple[int, int]]):
        self.actions = {action: 0 for action in legal_moves}
        self.value = {action: 0 for action in legal_moves}


class MCTS:
    """
    Implementation of MCTS with UCT exploration bonus
    """

    def __init__(self, state_manager: HexStateManager, root: Node, actor_net: ActorNetwork = None,
                 expansion_threshold=20, c = np.sqrt(2)) -> None:
        self.state_manager = state_manager
        self.root = root
        self.actor_net = actor_net
        self.expansion_threshold = expansion_threshold
        self.c = c
        #self.root.init_actions_and_values(self.state_manager.get_legal_moves())
    def reset_position(self) -> None:
        """
        Resets board to root position and player
        """
        board = self.root.state[0]
        player = self.root.state[1]
        self.state_manager.new_game(Player(player))
        self.state_manager.board = board.copy()

    def search(self, simulations=1000) -> tuple[tuple[int, int], dict[tuple[int, int], float]]:
        """
        Runs mcts simulations, and returns the best action and
        distribution of actions

        Args:
            simulations: no. of simulations

        Returns:
            best action, distribution
        """
        for i in range(simulations):
            self.simulate()

        self.reset_position()
        distribution = self.get_distribution()
        best_action = max(distribution, key=distribution.get)
        return best_action, distribution

    def simulate(self) -> None:
        """
        Runs a single mcts simulation (tree_policy + rollout + backprop)
        """
        self.reset_position()
        leaf = self.sim_tree_policy()
        reward = self.sim_rollout()
        self.backpropagate(leaf, reward)

    def sim_tree_policy(self, use_expansion_threshold=True) -> Node:
        """
        Runs tree policy until a leaf node is reached.
        If using expansion_threshold then a leaf node will
        expand all of its children when the node has been visited
        enough times. If not, then a single child node will
        always be expanded per simulation (if not final state)

        Returns:
            leaf node
        """
        node = self.root
        while not self.state_manager.is_final():
            action = self.select_move_uct(node)
            node.selected_action = action

            if action in node.children:
                self.state_manager.make_move(action)
                node = node.children[action]
                continue

            if not use_expansion_threshold:
                self.state_manager.make_move(action)
                child_state = self.state_manager.get_state()
                node.expand(action, child_state)
                node = node.children[action]
                node.init_actions_and_values(self.state_manager.get_legal_moves())

            if len(node.children) == 0:
                if use_expansion_threshold and node.visits >= self.expansion_threshold:
                    legal_moves = self.state_manager.get_legal_moves()
                    child_states = self.state_manager.get_possible_states(legal_moves)
                    for move, child_state in zip(legal_moves, child_states):
                        child_legal_moves = [m for m in legal_moves if m != move]
                        node.expand(move, child_state)
                        node.children[move].init_actions_and_values(child_legal_moves)

                    self.state_manager.make_move(action)
                    node = node.children[action]

                return node

        return node

    def select_move_uct(self, node: Node, c=np.sqrt(2)) -> tuple[int, int]:
        """
        Selects the best move from given node according to uct.
        If node has unvisited children, then choose one of those at random

        Args:
            node: current state / tree node
            c: uct c-param

        Returns:
            best action
        """
        unvisited_actions = [action for action, visits in node.actions.items() if visits == 0]
        if len(unvisited_actions) > 0:
            return random.choice(unvisited_actions)

        legal_moves = self.state_manager.get_legal_moves()
        values = []
        ucts = []

        for action in legal_moves:
            values.append(node.value[action])
            ucts.append(self.c * np.sqrt(np.log(node.visits) / (1 + node.actions[action])))

        values = np.array(values)
        ucts = np.array(ucts)

        if self.state_manager.current_player == Player.RED:
            return legal_moves[np.argmax(values + ucts)]
        else:
            return legal_moves[np.argmin(values - ucts)]

    def sim_rollout(self, epsilon=.1) -> int:
        """
        Epsilon greedy rollout from leaf node

        Args:
            epsilon: choose random move with probability=epsilon

        Returns:
            1 if red wins, -1 if blue wins
        """
        state = self.state_manager.get_state()

        while not self.state_manager.is_final():
            legal_moves = self.state_manager.get_legal_moves()

            if random.random() < epsilon or self.actor_net is None:
                action = random.choice(legal_moves)
            else:
                action = self.actor_net.get_action(board_state=state[0], current_player=state[1])
            try:
                self.state_manager.make_move(action)
            except ValueError as e:
                    print(f"Failed move: {action}")
                    print(f"Current state before failure: " )
                    print(self.state_manager.board)
            state = self.state_manager.get_state()

        return 1 if self.state_manager.winner == Player.RED else -1

    def backpropagate(self, node: Node, reward: int) -> None:
        """
        Updates values and visit counts along the path chosen by tree_policy.

        Args:
            node: node to update (starts with leaf node)
            reward: 1 or -1
        """
        node.visits += 1

        # if not leaf node
        if node.selected_action is not None:
            node.actions[node.selected_action] += 1
            node.value[node.selected_action] += (reward - node.value[node.selected_action]) / node.actions[
                node.selected_action]

        if node.parent is not None:
            self.backpropagate(node.parent, reward)

    def get_distribution(self) -> dict[tuple[int, int], float]:
        """
        Compute the distribution of visit counts among the actions from the root node.

        Returns:
            A dictionary mapping actions to their visit counts, normalized to form a probability distribution.
        """
        total_visits = sum(visits for visits in self.root.actions.values())
        
        return {action: visits / total_visits for action, visits in self.root.actions.items()}
