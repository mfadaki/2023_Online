import random
import copy
import math


class AllocationGame:
    def __init__(self, days_remaining=10, allocation=(0, 0), cum_demands=(0, 0), demands=(0, 0)):
        self.days_remaining = days_remaining
        self.allocation = allocation
        self.cum_demands = cum_demands
        self.demands = demands
        self.demands_range = (2, 5)
        self.inventory = 8

    def play(self, allocation, demands):
        self.allocation = allocation
        self.cum_demands = tuple(sum(x) for x in zip(self.cum_demands, demands))
        self.days_remaining -= 1

    def is_game_over(self):
        return self.days_remaining == 0

    def get_valid_allocations(self):
        max_allocation = min(self.inventory, *self.demands)
        valid_allocations = [(a, max_allocation - a) for a in range(max_allocation + 1)]
        return valid_allocations

    def get_penalty(self):
        target_fill_rate = 0.95
        penalty_per_shortage = 10
        shortages = [max(target_fill_rate * demand - allocated, 0) for demand, allocated in zip(self.cum_demands, self.allocation)]
        penalty = sum(shortage * penalty_per_shortage for shortage in shortages)
        return penalty


class MCTSNode:
    def __init__(self, game_state, parent=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.penalty_sum = 0

    def select(self):
        c = 1 / math.sqrt(2)

        def ucb_value(child):
            if child.visits == 0:
                return float('inf')  # Assign a high value for unvisited nodes
            else:
                return child.penalty_sum / child.visits - c * math.sqrt(math.log(self.visits) / child.visits)

        return max(self.children, key=ucb_value)

    def expand(self):
        allocation_options = self.game_state.get_valid_allocations()
        for allocation in allocation_options:
            game_state_copy = copy.deepcopy(self.game_state)
            game_state_copy.play(allocation, self.game_state.demands)
            self.children.append(MCTSNode(game_state_copy, self))

    def simulate(self, demands):
        game_state_copy = copy.deepcopy(self.game_state)
        while not game_state_copy.is_game_over():
            valid_allocations = game_state_copy.get_valid_allocations()
            random_allocation = random.choice(valid_allocations)
            game_state_copy.play(random_allocation, demands)
        return game_state_copy.get_penalty()

    def backpropagate(self, penalty):
        self.visits += 1
        self.penalty_sum += penalty
        if self.parent:
            self.parent.backpropagate(penalty)


def mcts(game_state, iterations=10000):
    root = MCTSNode(game_state)

    for _ in range(iterations):
        node = root
        while not node.game_state.is_game_over() and node.children:
            node = node.select()

        if not node.game_state.is_game_over() and not node.children:
            node.expand()

        if node.children:
            node = random.choice(node.children)

        demands = [random.randint(*game_state.demands_range) for _ in range(2)]
        penalty = node.simulate(demands)
        node.backpropagate(penalty)

    return min(root.children, key=lambda child: child.penalty_sum / child.visits) if root.children else None


if __name__ == '__main__':
    game = AllocationGame()
    allocations = []
    demands_list = []

    while not game.is_game_over():
        demands = [random.randint(*game.demands_range) for _ in range(2)]
        game.demands = tuple(demands)
        demands_list.append(demands)

        best_child = mcts(game)
        if best_child:
            best_allocation = best_child.game_state.allocation
            allocations.append(best_allocation)
            game.play(best_allocation, demands)
        else:
            break

    print("Allocations:", allocations)
    print("Demands:", demands_list)
    print("Total Penalty:", game.get_penalty())
    print("Fill rates:", game.get_fill_rate())
