import random
import math
from copy import deepcopy


class HalvingGame:
    def __init__(self, n=64):
        self.n = n
        self.current_player = 1

    def play(self, action):
        if action == "subtract":
            if self.n > 1:
                self.n -= 1
                self.current_player = 3 - self.current_player
                return True
        elif action == "divide":
            if self.n > 1:
                self.n = (self.n // 2) + (self.n % 2)
                self.current_player = 3 - self.current_player
                return True
        return False

    def is_game_over(self):
        return self.n == 1

    def get_valid_actions(self):
        return ["subtract", "divide"]

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
        for action in self.game_state.get_valid_actions():
            new_game_state = self.game_state.clone()
            new_game_state.play(action)
            child = MCTSNode(new_game_state, parent=self)
            self.children.append(child)

    def select(self):
        return max(self.children, key=lambda child: child.wins/(child.visits + 1e-6) + math.sqrt(2*math.log(self.visits + 1e-6)/(child.visits + 1e-6)))

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

    return max(root.children, key=lambda child: child.visits) if root.children else None


if __name__ == "__main__":
    game = HalvingGame(n=15)
    while not game.is_game_over():
        if game.current_player == 1:
            mcts_result = mcts(game)
            if mcts_result:
                action = mcts_result.game_state.get_valid_actions()[0]
                game.play(action)
                print(f"AI played {action}, value: {game.n}")
            else:
                break
        else:
            action = input("Enter your action (subtract/divide): ").lower()

            while action not in game.get_valid_actions() or not game.play(action):
                print("Invalid action. Please choose 'subtract' or 'divide'.")
                action = input("Enter your action (subtract/divide): ").lower()
            print(f"You played {action}, value: {game.n}")
