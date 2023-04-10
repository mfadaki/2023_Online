import random
import copy
import math


class AllocationGame:

    def __init__(self, days_remaining=3, allocation=(0, 0), customer_demands=(0, 0), cum_demands=(0, 0)):
        self.days_remaining = days_remaining
        self.allocation = allocation
        self.customer_demands = customer_demands
        self.cum_demands = cum_demands
        self.demands_range = (2, 5)
        self.inventory = 8
        # self.demands = demands or [random.randint(*self.demands_range) for _ in range(2)]
    # ...

    # def get_valid_allocations(self):
    #     demands = [random.randint(*self.demands_range) for _ in range(2)]
    #     allocations = [(i, 100 - i) for i in range(0, min(demands[0], 100) + 1)]
    #     return allocations, demands

    def get_valid_allocations(self):
        # demands = [random.randint(*self.demands_range) for _ in range(2)]
        # valid_allocations = [(i, j) for i in range(0, demands[0] + 1) for j in range(0, demands[1] + 1) if i + j <= 8 and i <= demands[0] and j <= demands[1]]
        max_allocation = min(self.inventory, *self.customer_demands)
        valid_allocations = [(a, max_allocation - a) for a in range(max_allocation + 1)]

        if not valid_allocations:
            valid_allocations = [(demands[0], demands[1])]
        return valid_allocations

    def play(self, allocation):
        self.allocation = allocation
        self.cum_demands = [co + alloc for co, alloc in zip(self.cum_demands, self.allocation)]
        self.inventory -= sum(self.allocation)
        self.customer_demands = [cd + demand for cd, demand in zip(self.customer_demands, self.demands)]
        self.days_remaining -= 1

    def is_game_over(self):
        return self.days_remaining == 0

    def get_fill_rate(self):
        return [orders / demands if demands > 0 else 1 for orders, demands in zip(self.cum_demands, self.customer_demands)]

    def get_penalty(self):
        fill_rates = self.get_fill_rate()
        return sum(10 * (0.95 - fill_rate) * demands for fill_rate, demands in zip(fill_rates, self.customer_demands) if fill_rate < 0.95)


class MCTSNode:
    def __init__(self, game_state):
        self.game_state = game_state
        self.children = []
        self.visits = 0
        self.penalty_sum = 0
        self.parent = None

    def select(self):
        exploration_constant = math.sqrt(2)
        best_child = max(self.children, key=lambda child: child.ucb(exploration_constant))
        return best_child

    def ucb(self, exploration_constant):
        if self.visits == 0 or not self.parent:
            return float('inf')
        return self.penalty_sum / self.visits + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self):
        allocation_options = self.game_state.get_valid_allocations()
        for allocation in allocation_options:
            game_state_copy = copy.deepcopy(self.game_state)
            game_state_copy.play(allocation)
            child = MCTSNode(game_state_copy)
            self.children.append(child)

    def backpropagate(self, penalty):
        self.visits += 1
        self.penalty_sum += penalty
        if self.parent:
            self.parent.backpropagate(penalty)

    def simulate(self):
        game_state_copy = copy.deepcopy(self.game_state)
        while not game_state_copy.is_game_over():
            valid_allocations = game_state_copy.get_valid_allocations()
            random_allocation = random.choice(valid_allocations)
            game_state_copy.play(random_allocation)
        return game_state_copy.get_penalty()


def mcts(game_state, iterations=1000):
    root = MCTSNode(game_state)

    for _ in range(iterations):
        node = root
        while not node.game_state.is_game_over() and node.children:  # It is not aleaf node
            node = node.select()  # One of the children with highest UCB is selected as new node

        if not node.game_state.is_game_over() and not node.children:
            node.expand()
        else:
            if node.children:
                node = random.choice(node.children)

        penalty = node.simulate()  # Simulate the game to the end
        node.backpropagate(penalty)

    if root.children:
        best_child = None
        min_penalty = float("inf")
        for child in root.children:
            if child.visits > 0:
                penalty_rate = child.penalty_sum / child.visits
                if penalty_rate < min_penalty:
                    min_penalty = penalty_rate
                    best_child = child
        return best_child
    else:
        return None

    # return min(root.children, key=lambda child: child.penalty_sum / child.visits) if root.children else None


if __name__ == "__main__":
    game = AllocationGame()
    allocations = []
    demands_list = []

    while not game.is_game_over():
        best_child = mcts(game)
        if best_child:
            allocation = best_child.game_state.allocation
            allocations.append(allocation)
            game.play(allocation, game.demands)
            demands_list.append(game.demands)
        else:
            break

    print("Allocations:", allocations)
    print("Demands:", demands_list)
    print("Fill rates customer 1:", round(sum([allocations[x][0] for x in range(len(allocations))])/sum([demands_list[x][0] for x in range(len(demands_list))]), 2))
    print("Fill rates customer 2:", round(sum([allocations[x][1] for x in range(len(allocations))])/sum([demands_list[x][1] for x in range(len(demands_list))]), 2))
