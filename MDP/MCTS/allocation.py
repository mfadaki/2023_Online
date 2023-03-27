import random
import math
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt


class AllocationGame:
    def __init__(self, days_left=10, inventory=70, fill_rate_target=0.95, demands_range=(30, 80), penalty_per_shortage=10):
        self.inventory = inventory
        self.days_left = days_left
        self.fill_rate_target = fill_rate_target
        self.demands_range = demands_range
        self.penalty_per_shortage = penalty_per_shortage
        self.customer_shortages = [0, 0]
        self.customer_demands = [0, 0]
        self.demands = []  # Added attribute to store the demands

    def play(self, allocation):
        if sum(allocation) > self.inventory or self.days_left == 0:
            return False

        demands = [random.randint(*self.demands_range) for _ in range(2)]
        self.customer_demands = [cd + demand for cd, demand in zip(self.customer_demands, demands)]
        self.demands.append(demands)  # Store the demands for the current day
        shortages = [max(demand - alloc, 0) for demand, alloc in zip(demands, allocation)]
        self.customer_shortages = [cs + shortage for cs, shortage in zip(self.customer_shortages, shortages)]

        self.days_left -= 1
        return True

    def is_game_over(self):
        return self.days_left == 0

    def get_valid_allocations(self):
        return [(i, self.inventory - i) for i in range(self.inventory + 1)]

    def get_penalty(self):
        return sum(self.penalty_per_shortage * shortage for shortage in self.customer_shortages)

    def clone(self):
        return deepcopy(self)

    def get_fill_rates(self):
        return [1 - (shortage / demand) for shortage, demand in zip(self.customer_shortages, self.customer_demands)]


class MCTSNode:
    def __init__(self, game_state, parent=None, allocation=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.penalty_sum = 0
        self.allocation = allocation

    def expand(self):
        for allocation in self.game_state.get_valid_allocations():
            new_game_state = self.game_state.clone()
            new_game_state.play(allocation)
            child = MCTSNode(new_game_state, parent=self, allocation=allocation)
            self.children.append(child)

    def select(self):
        return max(self.children, key=lambda child: -child.penalty_sum/(child.visits + 1e-6) + math.sqrt(2*math.log(self.visits + 1e-6)/(child.visits + 1e-6)))

    def backpropagate(self, penalty):
        self.visits += 1
        self.penalty_sum += penalty
        if self.parent:
            self.parent.backpropagate(penalty)


def mcts(game_state, iterations=1000):
    root = MCTSNode(game_state)

    for _ in range(iterations):
        node = root
        while not node.game_state.is_game_over() and node.children:
            node = node.select()

        if not node.children and not node.game_state.is_game_over():
            node.expand()

        if node.children:
            node = random.choice(node.children)

        penalty = node.game_state.get_penalty()
        node.backpropagate(penalty)

    return min(root.children, key=lambda child: child.penalty_sum/child.visits) if root.children else None


def draw_graph(node, G=None, pos=None):
    if node is None:
        return nx.DiGraph(), {}

    if G is None:
        G = nx.DiGraph()
        pos = {}

    G.add_node(node)
    pos[node] = (node.game_state.inventory, -node.game_state.days_left)

    if node.parent:
        G.add_edge(node.parent, node)

    for child in node.children:
        draw_graph(child, G, pos)

    return G, pos


if __name__ == "__main__":
    game = AllocationGame()
    allocations = []
    demands_list = []

    while not game.is_game_over():
        mcts_result = mcts(game)
        if mcts_result:
            game.play(mcts_result.allocation)
            allocations.append(mcts_result.allocation)
            demands_list.append(game.demands)  # Store the demands
        else:
            break

    fill_rates = game.get_fill_rates()
    print("Fill rates:")
    for i, fill_rate in enumerate(fill_rates, start=1):
        print(f"Customer {i}: {fill_rate:.2%}")

    print("Allocations:")
    for i, alloc in enumerate(allocations, start=1):
        print(f"Day {i}: Customer 1 - {alloc[0]}, Customer 2 - {alloc[1]}")

    print("Total penalty: $", game.get_penalty())

    # Print demands during the performance review period
    print("Demands during the performance review period:")
    for i, demands in enumerate(demands_list, start=1):
        print(f"Day {i}: Customer 1 - {demands[0]}, Customer 2 - {demands[1]}")

    # Generate the graph and draw it
    mcts_result = mcts(game, iterations=500)  # Reduced the number of iterations for easier visualization
    G, pos = draw_graph(mcts_result)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, font_weight='bold', node_color='cyan', edgecolors='black')
    plt.show()

    # Print demands during the performance review period
    print("Demands during the performance review period:")
    for i, demands in enumerate(game.demands, start=1):
        print(f"Day {i}: Customer 1 - {demands[0]}, Customer 2 - {demands[1]}")


#######################################
# Networkx Graph  #####################
#######################################


G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 5)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4, 5)

# explicitly set positions
pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

options = {
    "font_size": 36,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
}
nx.draw_networkx(G, pos, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()
