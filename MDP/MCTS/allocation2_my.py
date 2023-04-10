import random
from copy import deepcopy
import math


class AllocationGame:
    def __init__(self):
        self.prp = 6
        self.days_remaining = 6
        self.fill_rate_target = 0.95
        self.demands_range = (2, 7)
        self.inventory = 10
        self.penalty_per_shortage = 10
        self.demands = [tuple([random.randint(*self.demands_range) for _ in range(2)]) for __ in range(self.prp)]
        # self.demands = [(7, 7), (4, 7), (2, 6), (6, 6)]
        self.demands = [(5, 6), (2, 6), (4, 5), (7, 7), (7, 7), (2, 7)]
        self.p = 10
        self.min_profit = 0
        self.max_profit = self.inventory * self.prp

    def is_game_over(self):
        return self.days_remaining == 0

    def clone(self):
        return deepcopy(self)

    def get_valid_allocations(self):
        # Remaining days should not be zero
        valid_allocations = []
        for a1 in range(1, min(self.inventory, self.demands[self.prp-self.days_remaining][0]) + 1):  # Limit allocation for customer 1 based on demand
            for a2 in range(1, min(self.inventory - a1, self.demands[self.prp-self.days_remaining][1]) + 1):  # Limit allocation for customer 2 based on demand
                valid_allocations.append((a1, a2))

        return valid_allocations

    def play(self):
        if self.days_remaining == 0:
            return False
        else:
            self.days_remaining -= 1
            return True

    def get_penalty(self, allocations):
        no_days = len(allocations)  # during the simulation process, from a node onward to the last day
        tot_alloc_1 = sum([_[0] for _ in allocations])
        tot_demands_1 = sum([_[0] for _ in self.demands[:no_days]])
        fill_rate_1 = tot_alloc_1 / tot_demands_1
        tot_alloc_2 = sum([_[1] for _ in allocations])
        tot_demands_2 = sum([_[1] for _ in self.demands[:no_days]])
        fill_rate_2 = tot_alloc_2 / tot_demands_2
        profit = self.p*(tot_alloc_1+tot_alloc_2)-(self.penalty_per_shortage * max(self.fill_rate_target - fill_rate_1, 0) + self.penalty_per_shortage * max(self.fill_rate_target - fill_rate_2, 0))
        norm_profit = (profit-self.min_profit)/(self.max_profit-self.min_profit)
        return norm_profit


class MCTSNode:
    _registry = []

    def __init__(self, game_state, parent=None, allocation=None, state_indice=[0, 0]):
        self._registry.append(self)
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.penalty_sum = 0
        self.allocation = allocation  # Each node has an allocation (a1, a2)
        self.state_indice = state_indice
        print(self.allocation)

    def select(self):
        best_child = max(self.children, key=lambda child: child.ucb())
        return best_child

    def ucb(self):
        if self.visits == 0 or not self.parent:
            return float("inf")
        # return self.penalty_sum + math.sqrt(2) * math.sqrt(
            # math.log(self.parent.visits) / self.visits)
        return self.penalty_sum / self.visits + math.sqrt(2000) * math.sqrt(
            math.log(self.parent.visits) / self.visits)

    def expand(self):
        counter = 0
        new_game_state = self.game_state.clone()
        new_game_state.play()
        for alc in self.game_state.get_valid_allocations():
            child = MCTSNode(new_game_state, parent=self, allocation=alc, state_indice=[self.state_indice[0]+1, counter])
            self.children.append(child)
            counter += 1

    def simulate(self):
        game_state_copy = self.game_state.clone()
        allocations_rollout = []
        while not game_state_copy.is_game_over():
            # print(f'days remaining =  {game_state_copy.days_remaining}')
            valid_allocations = game_state_copy.get_valid_allocations()
            random_allocation = random.choice(valid_allocations)
            allocations_rollout.append(random_allocation)
            # print(allocations_rollout)
            game_state_copy.play()
        penalty_rollout = game_state_copy.get_penalty(allocations_rollout)
        return penalty_rollout

    def backpropagate(self, penalty):
        self.visits += 1
        self.penalty_sum += penalty
        if self.parent:
            self.parent.backpropagate(penalty)


def mcts(game_state, iterations=100000):
    root = MCTSNode(game_state)

    for _ in range(iterations):
        node = root
        # print(node.game_state.days_remaining)
        while not node.game_state.is_game_over() and node.children:
            node = node.select()
            # print(node.game_state.days_remaining)

        if not node.game_state.is_game_over() and not node.children:
            # print(node.game_state.days_remaining)
            node.expand()
            # print(node.game_state.days_remaining)
            node = random.choice(node.children)
        # print(node.game_state.days_remaining)

        if not node.game_state.is_game_over():
            penalty = node.simulate()
            node.backpropagate(penalty)
        else:
            allocation_tree = []
            node_tree = node
            while node_tree.parent != None:
                allocation_tree.append(node_tree.allocation)
                node_tree = node_tree.parent
            allocation_tree.reverse()
            print(f'allocaiton tree = {allocation_tree}')
            penalty = node.game_state.get_penalty(allocation_tree)
            node.backpropagate(penalty)

    if root.children:
        best_child = None
        min_penalty = float("-inf")
        for child in root.children:
            if child.visits > 0:
                # penalty_rate = (child.penalty_sum / child.visits) + math.sqrt(2000) * math.sqrt(
                # math.log(child.parent.visits) / child.visits)
                penalty_rate = child.penalty_sum / child.visits
                if penalty_rate > min_penalty:
                    min_penalty = penalty_rate
                    best_child = child
        return best_child
    else:
        return None


game = AllocationGame()

allocations = []
while not game.is_game_over():
    best_child = mcts(game)
    if best_child:
        # allocation = best_child.game_state.allocation
        allocation = best_child.allocation
        allocations.append(allocation)
        game.play()
    else:
        break

print("Allocations:", allocations)
print("Demands:", game.demands)
print("Fill rates customer 1:", round(sum([allocations[x][0] for x in range(len(allocations))])/sum([game.demands[x][0] for x in range(len(game.demands))]), 2))
print("Fill rates customer 2:", round(sum([allocations[x][1] for x in range(len(allocations))])/sum([game.demands[x][1] for x in range(len(game.demands))]), 2))


# Print information about all nodes
""" 
for i, node in enumerate(MCTSNode._registry):
    print(f"Node {i}:")
    print(f"  Allocation: {node.allocation}")
    print(f"  State Indices: {node.state_indice}")
    print(f"  Visits: {node.visits}")
    print(f"  Penalty Sum: {node.penalty_sum}")
    print()
 """
