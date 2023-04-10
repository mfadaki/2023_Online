import random
from copy import deepcopy
import math
from tqdm import tqdm  # Progress Bar

##################################
# Standard MCTS without augmenting the selecting process or roll-out process
##################################

model_param = {
    'num_iterations': 100000,
    'c': 2000000
}

# for key, value in model_param.items():
# setattr(self, key, value)


class AllocationGame:
    def __init__(self):
        self.prp = 10
        self.days_remaining = 10
        self.fill_rate_target = 0.95
        self.demands_range = (2, 7)
        self.inventory = 10
        self.reward_per_shortage = 10
        # self.demands = [tuple([random.randint(*self.demands_range) for _ in range(2)]) for __ in range(self.prp)]
        # self.demands = [(7, 7), (4, 7), (2, 6), (6, 6)]
        self.demands = [(7, 7), (6, 5), (2, 6), (4, 3), (6, 3), (2, 7), (6, 5), (2, 6), (4, 3), (6, 6)]
        self.p = 10
        self.min_profit = 0
        self.max_profit = self.inventory * self.prp

    def is_game_over(self):
        return self.days_remaining == 0

    def clone(self):
        return deepcopy(self)

    def get_valid_allocations(self):
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

    def get_reward(self, allocations):
        no_days = len(allocations)  # during the simulation process, from a node onward to the last day
        tot_alloc_1 = sum([_[0] for _ in allocations])
        tot_demands_1 = sum([_[0] for _ in self.demands[:no_days]])
        fill_rate_1 = tot_alloc_1 / tot_demands_1
        tot_alloc_2 = sum([_[1] for _ in allocations])
        tot_demands_2 = sum([_[1] for _ in self.demands[:no_days]])
        fill_rate_2 = tot_alloc_2 / tot_demands_2
        profit = self.p*(tot_alloc_1+tot_alloc_2)-(self.reward_per_shortage * max(self.fill_rate_target - fill_rate_1, 0) + self.reward_per_shortage * max(self.fill_rate_target - fill_rate_2, 0))
        # Normalize to be in range [0,1]
        norm_profit = (profit-self.min_profit)/(self.max_profit-self.min_profit)
        return norm_profit


class MCTSNode:
    _registry = []
    opt_allocations = []

    def __init__(self, game_state, parent=None, allocation=None, state_indice=[0, 0]):
        self._registry.append(self)
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward_sum = 0
        self.allocation = allocation  # Each node has an allocation (a1, a2)
        self.state_indice = state_indice

        if parent is None:
            self.path = []
        else:
            self.path = parent.path + [allocation]

    def select(self):
        best_child = max(self.children, key=lambda child: child.ucb())
        return best_child

    def ucb(self):
        if self.visits == 0 or not self.parent:
            return float("inf")
        return self.reward_sum / self.visits + math.sqrt(model_param["c"]) * math.sqrt(
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
            valid_allocations = game_state_copy.get_valid_allocations()
            random_allocation = random.choice(valid_allocations)
            allocations_rollout.append(random_allocation)
            game_state_copy.play()
        reward_rollout = game_state_copy.get_reward(allocations_rollout)
        return reward_rollout

    def backpropagate(self, reward):
        self.visits += 1
        self.reward_sum += reward
        if self.parent:
            self.parent.backpropagate(reward)


def mcts(game_state, iterations):
    root = MCTSNode(game_state)

    for _ in tqdm(range(iterations)):
        node = root
        while not node.game_state.is_game_over() and node.children:
            node = node.select()

        if not node.game_state.is_game_over() and not node.children:
            node.expand()
            node = random.choice(node.children)

        if not node.game_state.is_game_over():
            reward = node.simulate()
            node.backpropagate(reward)
        else:
            allocation_tree = []
            node_tree = node
            while node_tree.parent != None:
                allocation_tree.append(node_tree.allocation)
                node_tree = node_tree.parent
            allocation_tree.reverse()
            reward = node.game_state.get_reward(allocation_tree)
            node.backpropagate(reward)

    if root.children:
        best_child = None
        min_reward = float("-inf")
        for child in root.children:
            if child.visits > 0:
                # reward_rate = (child.reward_sum / child.visits) + math.sqrt(2000) * math.sqrt(
                # math.log(child.parent.visits) / child.visits)
                reward_rate = child.reward_sum / child.visits
                if reward_rate > min_reward:
                    min_reward = reward_rate
                    best_child = child
        return best_child
    else:
        return None


game = AllocationGame()

allocations = []
while not game.is_game_over():
    best_child = mcts(game, model_param["num_iterations"])
    if best_child:
        allocation = best_child.allocation
        allocations.append(allocation)
        MCTSNode.opt_allocations.append(allocation)
        game.play()
    else:
        break


print("Allocations:", allocations)
print("Demands:", game.demands)
print("Fill rates customer 1:", round(sum([allocations[x][0] for x in range(len(allocations))])/sum([game.demands[x][0] for x in range(len(game.demands))]), 2))
print("Fill rates customer 2:", round(sum([allocations[x][1] for x in range(len(allocations))])/sum([game.demands[x][1] for x in range(len(game.demands))]), 2))


################################


""" while not game.is_game_over():
    best_child = mcts(game)
    if best_child:
        game.play()
    else:
        break """

# Print information about all nodes
"""
for i, node in enumerate(MCTSNode._registry):
    print(f"Node {i}:")
    print(f"  Allocation: {node.allocation}")
    print(f"  State Indices: {node.state_indice}")
    print(f"  Visits: {node.visits}")
    print(f"  reward Sum: {node.reward_sum}")
    print()
 """

#######################################################


def find_best_allocation_sequence():
    root_node = MCTSNode._registry[0]
    best_allocations = []
    current_node = root_node

    for i in range(7):
        best_child = None
        max_reward = float("-inf")
        for child in current_node.children:
            if child is None:
                continue
            reward_rate = child.reward_sum / child.visits if child.visits != 0 else 0
            if reward_rate > max_reward:
                max_reward = reward_rate
                best_child = child

        if best_child is not None:
            best_allocations.append(best_child.allocation)
            current_node = best_child
        else:
            break

    return best_allocations


best_sequence = find_best_allocation_sequence()
print("Best allocation sequence:", best_sequence)
