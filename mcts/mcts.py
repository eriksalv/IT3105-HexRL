import math
import random
from anet import ActorNetwork
import numpy as np


def find_last_move(prev_state, current_state):
    diff = current_state - prev_state
    indices = np.argwhere(diff != 0)

    if len(indices) == 1:
        return tuple(indices[0])
    else:
        raise ValueError("Invalid states: More than one difference found.")
    
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.number_of_visits = 0
        self.results = 0

    def get_number_of_visits(self):
        return self.number_of_visits

    def get_results(self):
        return self.results

    def expand(self, child_states):
        """
        for cs in child_states:
            cs.get_state()
        """
        if not self.children:
            #print("Generating children")
            self.children = [Node(child_state, parent=self) for child_state in child_states]

    def update(self, result):
        self.number_of_visits += 1
        self.results += result

    def is_root(self):
        return self.parent is None


class MCTS:
    def __init__(self, root, actor_net: ActorNetwork = None):
        self.root = root
        self.actor_net = actor_net

    def search(self, budget):
        for _ in range(budget):
            leaf = self.traverse(self.root)
            simulation_result = self.rollout(leaf)
            self.backpropagate(leaf, simulation_result)
        return self.best_child(self.root)

    def traverse(self, node: Node):
        """
        Traversal function to tell which child node to do a rollout from

        :node: The node which we wish to explore
        :return: The children node with the most promice according to uct
        """
        node.expand(node.state.get_possible_states())
        while node.children:
            node = self.best_uct(node)
        return node

    def best_uct(self, node: Node, c_param=1.4):
        """
        Function that implements the exploitation vs exploration 

        :node: the node we wish to find the most promising children from
        :return: the node we either have not explored, and the one with the highest uct if all are explored
        """
        unvisited_children = [c for c in node.children if c.number_of_visits == 0]
        # visiting all the nodes before calling the uct calculation
        if unvisited_children:
            return random.choice(unvisited_children)

        choices_weights = []
        for c in node.children:
            uct_value = (c.results / c.number_of_visits) + c_param * math.sqrt(
                math.log(node.number_of_visits) / c.number_of_visits)
            choices_weights.append(uct_value)

        # get the child with the maximum UCT value
        best_child = node.children[choices_weights.index(max(choices_weights))]
        return best_child

    def rollout(self, node: Node, epsilon=.1):
        """
        Rollout method, chooses a next move randomly until we reach a final state

        :node: the node containing our current state which we initiate rollout from
        :return: an integer representing whether we won or not 
        """
        current_state = node.state

        while not current_state.is_final():
            possible_states = current_state.get_possible_states()

            if random.random() < epsilon or self.actor_net is None:
                current_state = random.choice(possible_states)
            else:
                action_idx = self.actor_net.get_action(current_state.board,
                                                       current_state.current_player.value)
                move = current_state.convert_to_move(action_idx)
                current_state = current_state.get_next_state(move)

        return 1 if current_state.is_win() else 0

    def backpropagate(self, node, result):
        if node.parent:
            node.update(result)
            node.parent.update(result)
            self.backpropagate(node.parent, result)

    def best_child(self, node: Node):
        """
        Function to select the most promising child node

        :node: The parent node we wish to decide the best move for
        :return: The child we visited the most during our search
        """
        best_child = None
        best_score = -1
        for child in node.children:
            #print(child.number_of_visits)
            #print(child.results)
            # child.state.get_state()
            if child.number_of_visits > best_score:
                best_score = child.number_of_visits
                best_child = child

        return best_child

    def get_distribution(self, node):
        """
        Compute the distribution of visit counts among the child nodes of the given node.
        
    
        :node: The parent node for which to compute the distribution.
        
        :Returns: distribution: A dictionary mapping child nodes to their visit counts, normalized to form a probability distribution.
        """
        distribution = {}
        total_visits = sum(child.number_of_visits for child in node.children)

        for child in node.children:
            action = find_last_move(node.state.board, child.state.board)
            if total_visits == 0:
                distribution[action] = 0.0
            else:
                distribution[action] = child.number_of_visits / total_visits
        return distribution
    
    def update(self, node):
        self.root = node
        