import random
import math
from copy import deepcopy


class HalvingGame:
    def __init__(self, n=64):
        self.n = n
        self.current_player = 1

    def play(self):
        if self.n > 1:
            self.n = self.n // 2
            self.current_player = 3 - self.current_player
            return True
        return False

    def is_game_over(self):
        return self.n == 1

    def clone(self):
        return deepcopy(self)


class MCTSNode:
    def __init__(self, game_state, parent=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def expand(self):
        new_game_state = self.game_state.clone()
        if new_game_state.play():
            child = MCTSNode(new_game_state, parent=self)
            self.children.append(child)

    def select(self):
        return max(self.children, key=lambda child: child.wins/(child.visits + 1e-6) + math.sqrt(2*math.log(self.visits + 1e-6)/child.visits))

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)


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

        result = 1 if node.game_state.current_player == 1 else 0
        node.backpropagate(result)

    return root.children[0] if root.children else None


if __name__ == "__main__":
    game = HalvingGame(n=64)
    while not game.is_game_over():
        mcts_result = mcts(game)
        if mcts_result:
            game.play()
            print(f"Player {3 - game.current_player} played, value: {game.n}")
        else:
            break
