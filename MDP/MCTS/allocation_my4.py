import random
from copy import deepcopy
import math
from tqdm import tqdm  # Progress Bar
from scipy.special import softmax
from sklearn import preprocessing
import json
from json import JSONEncoder
import numpy as np
import time


model_param = {
    'num_iterations': 100000,
    'num_repeats': 10,
    'augmented_valid_allocations': False,
    'augmented_rollout': False,
    'augmented_select': False,
    'augmented_fillrates': False,
    'c': 1e6,  # Exploration Coefficient
    'c1': 2,  # Augmentation Coefficient
    'Big_M': 1e20
}

# for key, value in model_param.items():
# setattr(self, key, value)


def proportional(s, x1, x2):
    return math.floor(min(x1, s*x1/(x1+x2))), math.floor(min(x2, s*x2/(x1+x2)))

# region Class - Working with JSON


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)
# endregion


class AllocationGame:
    def __init__(self):
        self.prp = 10
        self.days_remaining = 10
        self.fill_rate_target = 0.85
        self.demands_range = (2, 8)
        self.inventory = 10
        self.penalty_per_shortage_1 = 100
        self.penalty_per_shortage_2 = 100
        self.demands = [tuple([random.randint(*self.demands_range) for _ in range(2)]) for __ in range(self.prp)]
        while sum(self.demands[0]) <= self.inventory:
            self.demands = [tuple([random.randint(*self.demands_range) for _ in range(2)]) for __ in range(self.prp)]
        # self.demands = [(8, 5), (6, 3), (2, 6), (6, 3), (4, 7), (4, 5), (3, 9), (8, 2), (8, 3), (6, 6)]
        self.p = 10
        # self.min_profit = 0 # Using for nomalising the reward
        # self.max_profit = self.inventory * self.prp # Using for nomalising the reward

    def is_game_over(self):
        return self.days_remaining == 0

    def clone(self):
        return deepcopy(self)

    def get_valid_allocations(self):
        va = []
        valid_allocations = []
        for a1 in range(1, min(self.inventory, self.demands[self.prp-self.days_remaining][0]) + 1):  # Limit allocation for customer 1 based on demand
            for a2 in range(1, min(self.inventory - a1, self.demands[self.prp-self.days_remaining][1]) + 1):  # Limit allocation for customer 2 based on demand
                va.append((a1, a2))
        if model_param['augmented_valid_allocations'] == False:
            valid_allocations = va.copy()
        else:
            for _ in va:
                if sum(self.demands[self.prp-self.days_remaining]) > self.inventory:
                    if sum(_) == self.inventory:
                        valid_allocations.append(_)
                else:
                    if sum(_) == sum(self.demands[self.prp-self.days_remaining]):
                        valid_allocations.append(_)
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
        profit = self.p*(tot_alloc_1+tot_alloc_2)-(self.penalty_per_shortage_1 * max(self.fill_rate_target - fill_rate_1, 0) + self.penalty_per_shortage_2 * max(self.fill_rate_target - fill_rate_2, 0))
        # Normalize to be in range [0,1]
        # norm_profit = (profit-self.min_profit)/(self.max_profit-self.min_profit)
        return profit


class MCTSNode:
    _registry = []
    opt_allocations = []

    @classmethod
    def reset(cls):
        cls.instances = []
        cls._registry = []
        cls.opt_allocations = []

    def __init__(self, game_state, parent=None, allocation=[0, 0], state_indice=[0, 0]):
        self._registry.append(self)
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward_sum = 0
        self.allocation = allocation  # Each node has an allocation (a1, a2)
        self.state_indice = state_indice
        if self.state_indice[0] == 0:
            self.demand = self.game_state.demands[self.state_indice[0]]
        else:
            self.demand = self.game_state.demands[self.state_indice[0]-1]
            # self.demand = self.game_state.demands[self.state_indice[0]]

        self.distance = math.sqrt((self.demand[0]-self.allocation[0])**2 + (self.demand[1]-self.allocation[1])**2)

        if parent is None:
            self.path = []
        else:
            self.path = parent.path + [allocation]

    # If we need a variable is calculated and updated based on new values of above variables in the class.
    @property
    def explotation_score(self):
        return (self.reward_sum / self.visits if self.visits != 0 else 0) + model_param["c1"] * (-self.distance)

    def select(self):
        if model_param["augmented_select"] == False:
            best_child = max(self.children, key=lambda child: child.ucb())
        else:
            exploration = [math.sqrt(math.log(child.parent.visits) / child.visits) if child.visits != 0 else model_param['Big_M'] for child in self.children]
            explotation = [child.explotation_score for child in self.children]
            # explotation_normalized = preprocessing.normalize([explotation])

            select_score = [x + math.sqrt(model_param["c"])*y for x, y, in zip(explotation, exploration)]
            # select_score = [x + math.sqrt(model_param["c"])*y for x, y, in zip(explotation_normalized, exploration)]
            select_score_idx = select_score.index(max(select_score))
            best_child = self.children[select_score_idx]
        return best_child

    def ucb(self):
        if self.visits == 0 or not self.parent:
            return float("inf")
        return self.reward_sum / self.visits + math.sqrt(model_param["c"]) * math.sqrt(
            math.log(self.parent.visits) / self.visits)

    # We did not use this ....
    def augmenting_ucb(self):
        tree_level_no = self.game_state.prp - self.game_state.days_remaining
        # tree_level_no = self.state_indice[0]
        # demands_ = self.game_state.demands[:(tree_level_no+1)]
        today_demand = game_state.demands[tree_level_no]
        valid_allocations = self.game_state.get_valid_allocations()
        probs_select = [0]*len(valid_allocations)
        if sum(today_demand) <= self.game_state.inventory:
            try:
                probs_select[valid_allocations.index(today_demand)] = 1
                random_allocation = random.choices(valid_allocations, weights=probs_select, k=1)[0]
            except:
                print("very very weired!!!!")
                # break
        else:
            allocations_ = MCTSNode.opt_allocations[:(tree_level_no+1)]
            tot_alloc_1 = sum([_[0] for _ in allocations_])
            tot_demands_1 = sum([_[0] for _ in demands_])
            fill_rate_1 = tot_alloc_1 / tot_demands_1
            tot_alloc_2 = sum([_[1] for _ in allocations_])
            tot_demands_2 = sum([_[1] for _ in demands_])
            fill_rate_2 = tot_alloc_2 / tot_demands_2

            opt_a1, opt_a2 = self.AllocationPolicy(today_demand, fill_rate_1, fill_rate_2, game_state_copy.fill_rate_target, game_state_copy.inventory)
            # Euc distance between optimal allocation and valid allocations
            distances = [-math.sqrt((opt_a1-valid_allocations[_][0])**2 + (opt_a2-valid_allocations[_][1])**2) for _ in range(len(valid_allocations))]
            softmax_probs = softmax(distances, axis=0)
            random_allocation = random.choices(valid_allocations, weights=softmax_probs, k=1)[0]

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
            if model_param["augmented_rollout"] == False:
                random_allocation = random.choice(valid_allocations)
            else:  # For simulation, not randomly selecting the nodes for roll-out
                tree_level_no = game_state_copy.prp - game_state_copy.days_remaining
                demands_ = game_state_copy.demands[:(tree_level_no+1)]
                today_demand = game_state_copy.demands[tree_level_no]
                probs_select = [0]*len(valid_allocations)
                if sum(today_demand) <= game_state_copy.inventory:
                    try:
                        probs_select[valid_allocations.index(today_demand)] = 1
                        random_allocation = random.choices(valid_allocations, weights=probs_select, k=1)[0]
                    except:
                        print("very very weired!!!!")
                        break
                else:
                    allocations_ = MCTSNode.opt_allocations[:(tree_level_no+1)]
                    tot_alloc_1 = sum([_[0] for _ in allocations_])
                    tot_demands_1 = sum([_[0] for _ in demands_])
                    fill_rate_1 = tot_alloc_1 / tot_demands_1
                    tot_alloc_2 = sum([_[1] for _ in allocations_])
                    tot_demands_2 = sum([_[1] for _ in demands_])
                    fill_rate_2 = tot_alloc_2 / tot_demands_2

                    opt_a1, opt_a2 = self.AllocationPolicy(child.demand, fill_rate_1, fill_rate_2, game_state_copy.fill_rate_target, game_state_copy.inventory)
                    # Euc distance between optimal allocation and valid allocations
                    distances = [-math.sqrt((opt_a1-valid_allocations[_][0])**2 + (opt_a2-valid_allocations[_][1])**2) for _ in range(len(valid_allocations))]
                    softmax_probs = softmax(distances, axis=0)
                    random_allocation = random.choices(valid_allocations, weights=softmax_probs, k=1)[0]
            allocations_rollout.append(random_allocation)
            game_state_copy.play()
        reward_rollout = game_state_copy.get_reward(allocations_rollout)
        return reward_rollout

    def backpropagate(self, reward):
        self.visits += 1
        self.reward_sum += reward
        if self.parent:
            self.parent.backpropagate(reward)

    # Ick-Hyun Kwon paper
    def AllocationPolicy(self, today_demand, fill_rate_1, fill_rate_2, target_fr, S):
        d1 = today_demand[0]  # Same day t
        d2 = today_demand[1]  # Same day t

        tetha1 = (fill_rate_1 - target_fr) / (abs(fill_rate_1 - target_fr) + abs(fill_rate_2 - target_fr))
        tetha2 = (fill_rate_2 - target_fr) / (abs(fill_rate_1 - target_fr) + abs(fill_rate_2 - target_fr))
        if tetha1 == 0:
            tetha1 += 1e-10

        if tetha2 == 0:
            tetha1 += 1e-10

        if tetha1 >= 0 and tetha2 >= 0 and (tetha1 > 0 and tetha2 > 0):
            p1 = tetha1/(tetha1+tetha2)
            p2 = tetha2/(tetha1+tetha2)
        elif tetha1 <= 0 and tetha2 <= 0 and (tetha1 < 0 and tetha2 < 0):
            p1 = abs(1/tetha1)/(abs(1/tetha1)+abs(1/tetha2))
            p2 = abs(1/tetha2)/(abs(1/tetha1)+abs(1/tetha2))
        elif tetha1 == 0 and tetha2 == 0:
            p1 = 1/2
            p2 = 1/2
        else:
            p1 = max(tetha1, 0)/(max(tetha1, 0)+max(tetha2, 0))
            p2 = max(tetha2, 0)/(max(tetha1, 0)+max(tetha2, 0))

        Q1_t = d1 - max((d1 + d2) - S, 0) * p1
        Q2_t = d2 - max((d1 + d2) - S, 0) * p2

        return Q1_t, Q2_t


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
        reward_arr = []
        for child in root.children:
            if child.visits > 0:
                tree_level_no = len(MCTSNode.opt_allocations)
                # reward_rate = (child.reward_sum / child.visits) + math.sqrt(2000) * math.sqrt(
                # math.log(child.parent.visits) / child.visits)
                if model_param["augmented_select"] == False:
                    reward_rate = child.reward_sum / child.visits
                elif model_param['augmented_fillrates'] == False:
                    reward_rate = child.explotation_score
                elif len(MCTSNode.opt_allocations) == 0:
                    reward_rate = child.explotation_score
                elif child.game_state.inventory >= sum(child.game_state.demands[tree_level_no]):
                    reward_rate = child.explotation_score
                    if child.allocation == (2, 5) and (root.demand == (2, 5) or root.demand == (2, 6)):
                        print('hahaha')
                elif child.game_state.inventory < sum(child.game_state.demands[tree_level_no]) and child.game_state.inventory != sum(child.allocation):
                    reward_rate = float('-inf')
                    # print('hahahahaha')
                elif child.game_state.inventory < sum(child.game_state.demands[tree_level_no]) and child.game_state.inventory == sum(child.allocation):
                    tree_level_no = len(MCTSNode.opt_allocations)
                    allocations_ = MCTSNode.opt_allocations[:(tree_level_no+1)]
                    demands_ = child.game_state.demands[:(tree_level_no+1)]
                    tot_alloc_1 = sum([_[0] for _ in allocations_])
                    tot_demands_1 = sum([_[0] for _ in demands_])
                    fill_rate_1 = tot_alloc_1 / tot_demands_1
                    tot_alloc_2 = sum([_[1] for _ in allocations_])
                    tot_demands_2 = sum([_[1] for _ in demands_])
                    fill_rate_2 = tot_alloc_2 / tot_demands_2

                    opt_a1, opt_a2 = child.AllocationPolicy(child.demand, fill_rate_1, fill_rate_2, child.game_state.fill_rate_target, child.game_state.inventory)
                    # Euc distance between optimal allocation and valid allocations
                    distance = math.sqrt((opt_a1-child.allocation[0])**2 + (opt_a2-child.allocation[1])**2)
                    reward_rate = (child.reward_sum / child.visits) - model_param['c1'] * distance

                # reward_rate = child.explotation_score
                # reward_rate = child.reward_sum / child.visits
                if reward_rate > min_reward:
                    min_reward = reward_rate
                    best_child = child
            reward_arr.append(reward_rate)
        # print(reward_arr)
        return best_child
    else:
        return None


def main_run():
    # Reset the Class (remove all instances)
    MCTSNode.reset()

    game = AllocationGame()
    time.sleep(1)

    allocations = []
    while not game.is_game_over():
        best_child = mcts(game, model_param["num_iterations"])
        if best_child:
            allocation = best_child.allocation
            allocations.append(allocation)
            MCTSNode.opt_allocations.append(allocation)
            # print(allocation)
            game.play()
        else:
            break

    print("Allocations:", allocations)
    print("Demands:", game.demands)
    FR1 = round(sum([allocations[x][0] for x in range(len(allocations))])/sum([game.demands[x][0] for x in range(len(game.demands))]), 2)
    FR2 = round(sum([allocations[x][1] for x in range(len(allocations))])/sum([game.demands[x][1] for x in range(len(game.demands))]), 2)
    print("Fill rates customer 1:", FR1)
    print("Fill rates customer 2:", FR2)

    # Proportional Allocation
    allocations_prop = [proportional(game.inventory, x[0], x[1]) for x in game.demands]
    fr1_prop = round(sum([y[0] for y in allocations_prop]) / sum([x[0] for x in game.demands]), 2)
    fr2_prop = round(sum([y[1] for y in allocations_prop]) / sum([x[1] for x in game.demands]), 2)

    jsResult = {
        "model_param": model_param,
        "Allocations": allocations,
        "Demands": game.demands,
        "FR1": FR1,
        "FR2": FR2,
        "FR1_prop": fr1_prop,
        "FR2_prop": fr2_prop,
        "penalty_per_shortage_1": game.penalty_per_shortage_1,
        "penalty_per_shortage_2": game.penalty_per_shortage_2,
        "s": game.inventory,
        "fill_rate_target": game.fill_rate_target,
        "p": game.p
    }

    return jsResult


for _ in range(model_param["num_repeats"]):
    jsResult = main_run()
    # encodedNumpyData = json.dumps(jsData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    with open("./results/" + str(_) + ".json", "w") as write_file:
        json.dump(jsResult, write_file, cls=NumpyArrayEncoder)
#######################################################

#######################################################


def find_best_allocation_sequence():
    root_node = MCTSNode._registry[0]
    best_allocations = []
    current_node = root_node
    reward_arr = []

    for i in range(7):  # 7 is the number of levels (PRP)
        print('_' * 50)
        print(f'Tree Level = {i}')
        print(f'Demand = {current_node.game_state.demands[i]}')
        reward_arr = []
        allocations_arr = []
        best_child = None
        max_reward = float("-inf")
        print(f"{'Allocation':<10}  {'Reward':<10}")
        for child in current_node.children:
            if child is None:
                continue
            # reward_rate = child.reward_sum / child.visits if child.visits != 0 else 0
            reward_rate = child.explotation_score
            reward_arr.append(reward_rate)
            allocations_arr.append(child.allocation)
            # print(child.allocation)
            if reward_rate > max_reward:
                max_reward = reward_rate
                best_child = child
            print(f'{str(child.allocation): <15}{reward_rate: <10}')
        if best_child is not None:
            best_allocations.append(best_child.allocation)
            current_node = best_child
        else:
            break

    return best_allocations


best_sequence = find_best_allocation_sequence()
print("Best allocation sequence:", best_sequence)
