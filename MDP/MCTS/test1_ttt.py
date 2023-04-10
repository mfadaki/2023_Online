#
# AI that learns to play Tic Tac Toe using
#        reinforcement learning
#                (MCTS)
#

# packages
from copy import deepcopy
from test_ttt import *
#
# MCTS algorithm implementation
#

# packages
import math
import random

# tree node class definition


class TreeNode():
    # class constructor (create tree node class instance)
    def __init__(self, board, parent):
        # init associated board state
        self.board = board

        # init is node terminal flag
        if self.board.is_win() or self.board.is_draw():
            # we have a terminal node
            self.is_terminal = True

        # otherwise
        else:
            # we have a non-terminal node
            self.is_terminal = False

        # init is fully expanded flag
        self.is_fully_expanded = self.is_terminal

        # init parent node if available
        self.parent = parent

        # init the number of node visits
        self.visits = 0

        # init the total score of the node
        self.score = 0

        # init current node's children
        self.children = {}

# MCTS class definition


class MCTS():
    # search for the best move in the current position
    def search(self, initial_state):
        # create root node
        self.root = TreeNode(initial_state, None)

        # walk through 1000 iterations
        for iteration in range(1000):
            # select a node (selection phase)
            node = self.select(self.root)

            # scrore current node (simulation phase)
            score = self.rollout(node.board)

            # backpropagate results
            self.backpropagate(node, score)

        # pick up the best move in the current position
        try:
            return self.get_best_move(self.root, 0)

        except:
            pass

    # select most promising node
    def select(self, node):
        # make sure that we're dealing with non-terminal nodes
        while not node.is_terminal:
            # case where the node is fully expanded
            if node.is_fully_expanded:
                node = self.get_best_move(node, 2)

            # case where the node is not fully expanded
            else:
                # otherwise expand the node
                return self.expand(node)

        # return node
        return node

    # expand node
    def expand(self, node):
        # generate legal states (moves) for the given node
        states = node.board.generate_states()

        # loop over generated states (moves)
        for state in states:
            # make sure that current state (move) is not present in child nodes
            if str(state.position) not in node.children:
                # create a new node
                new_node = TreeNode(state, node)

                # add child node to parent's node children list (dict)
                node.children[str(state.position)] = new_node

                # case when node is fully expanded
                if len(states) == len(node.children):
                    node.is_fully_expanded = True

                # return newly created node
                return new_node

        # debugging
        print('Should not get here!!!')

    # simulate the game via making random moves until reach end of the game
    def rollout(self, board):
        # make random moves for both sides until terminal state of the game is reached
        while not board.is_win():
            # try to make a move
            try:
                # make the on board
                board = random.choice(board.generate_states())

            # no moves available
            except:
                # return a draw score
                return 0

        # return score from the player "x" perspective
        if board.player_2 == 'x':
            return 1
        elif board.player_2 == 'o':
            return -1

    # backpropagate the number of visits and score up to the root node
    def backpropagate(self, node, score):
        # update nodes's up to root node
        while node is not None:
            # update node's visits
            node.visits += 1

            # update node's score
            node.score += score

            # set node to parent
            node = node.parent

    # select the best node basing on UCB1 formula
    def get_best_move(self, node, exploration_constant):
        # define best score & best moves
        best_score = float('-inf')
        best_moves = []

        # loop over child nodes
        for child_node in node.children.values():
            # define current player
            if child_node.board.player_2 == 'x':
                current_player = 1
            elif child_node.board.player_2 == 'o':
                current_player = -1

            # get move score using UCT formula
            move_score = current_player * child_node.score / child_node.visits + exploration_constant * math.sqrt(math.log(node.visits / child_node.visits))

            # better move has been found
            if move_score > best_score:
                best_score = move_score
                best_moves = [child_node]

            # found as good move as already available
            elif move_score == best_score:
                best_moves.append(child_node)

        # return one of the best moves randomly
        return random.choice(best_moves)

# Tic Tac Toe board class


class Board():
    # create constructor (init board class instance)
    def __init__(self, board=None):
        # define players
        self.player_1 = 'x'
        self.player_2 = 'o'
        self.empty_square = '.'

        # define board position
        self.position = {}

        # init (reset) board
        self.init_board()

        # create a copy of a previous board state if available
        if board is not None:
            self.__dict__ = deepcopy(board.__dict__)

    # init (reset) board
    def init_board(self):
        # loop over board rows
        for row in range(3):
            # loop over board columns
            for col in range(3):
                # set every board square to empty square
                self.position[row, col] = self.empty_square

    # make move
    def make_move(self, row, col):
        # create new board instance that inherits from the current state
        board = Board(self)

        # make move
        board.position[row, col] = self.player_1

        # swap players
        (board.player_1, board.player_2) = (board.player_2, board.player_1)

        # return new board state
        return board

    # get whether the game is drawn
    def is_draw(self):
        # loop over board squares
        for row, col in self.position:
            # empty square is available
            if self.position[row, col] == self.empty_square:
                # this is not a draw
                return False

        # by default we return a draw
        return True

    # get whether the game is won
    def is_win(self):
        ##################################
        # vertical sequence detection
        ##################################

        # loop over board columns
        for col in range(3):
            # define winning sequence list
            winning_sequence = []

            # loop over board rows
            for row in range(3):
                # if found same next element in the row
                if self.position[row, col] == self.player_2:
                    # update winning sequence
                    winning_sequence.append((row, col))

                # if we have 3 elements in the row
                if len(winning_sequence) == 3:
                    # return the game is won state
                    return True

        ##################################
        # horizontal sequence detection
        ##################################

        # loop over board columns
        for row in range(3):
            # define winning sequence list
            winning_sequence = []

            # loop over board rows
            for col in range(3):
                # if found same next element in the row
                if self.position[row, col] == self.player_2:
                    # update winning sequence
                    winning_sequence.append((row, col))

                # if we have 3 elements in the row
                if len(winning_sequence) == 3:
                    # return the game is won state
                    return True

        ##################################
        # 1st diagonal sequence detection
        ##################################

        # define winning sequence list
        winning_sequence = []

        # loop over board rows
        for row in range(3):
            # init column
            col = row

            # if found same next element in the row
            if self.position[row, col] == self.player_2:
                # update winning sequence
                winning_sequence.append((row, col))

            # if we have 3 elements in the row
            if len(winning_sequence) == 3:
                # return the game is won state
                return True

        ##################################
        # 2nd diagonal sequence detection
        ##################################

        # define winning sequence list
        winning_sequence = []

        # loop over board rows
        for row in range(3):
            # init column
            col = 3 - row - 1

            # if found same next element in the row
            if self.position[row, col] == self.player_2:
                # update winning sequence
                winning_sequence.append((row, col))

            # if we have 3 elements in the row
            if len(winning_sequence) == 3:
                # return the game is won state
                return True

        # by default return non winning state
        return False

    # generate legal moves to play in the current position
    def generate_states(self):
        # define states list (move list - list of available actions to consider)
        actions = []

        # loop over board rows
        for row in range(3):
            # loop over board columns
            for col in range(3):
                # make sure that current square is empty
                if self.position[row, col] == self.empty_square:
                    # append available action/board state to action list
                    actions.append(self.make_move(row, col))

        # return the list of available actions (board class instances)
        return actions

    # main game loop
    def game_loop(self):
        print('\n  Tic Tac Toe by Code Monkey King\n')
        print('  Type "exit" to quit the game')
        print('  Move format [x,y]: 1,2 where 1 is column and 2 is row')

        # print board
        print(self)

        # create MCTS instance
        mcts = MCTS()

        # game loop
        while True:
            # get user input
            user_input = input('> ')

            # escape condition
            if user_input == 'exit':
                break

            # skip empty input
            if user_input == '':
                continue

            try:
                # parse user input (move format [col, row]: 1,2)
                row = int(user_input.split(',')[1]) - 1
                col = int(user_input.split(',')[0]) - 1

                # check move legality
                if self.position[row, col] != self.empty_square:
                    print(' Illegal move!')
                    continue

                # make move on board
                self = self.make_move(row, col)

                # print board
                print(self)

                # search for the best move
                best_move = mcts.search(self)

                # legal moves available
                try:
                    # make AI move here
                    self = best_move.board

                # game over
                except:
                    pass

                # print board
                print(self)

                # check if the game is won
                if self.is_win():
                    print('player "%s" has won the game!\n' % self.player_2)
                    break

                # check if the game is drawn
                elif self.is_draw():
                    print('Game is drawn!\n')
                    break

            except Exception as e:
                print('  Error:', e)
                print('  Illegal command!')
                print('  Move format [x,y]: 1,2 where 1 is column and 2 is row')

    # print board state
    def __str__(self):
        # define board string representation
        board_string = ''

        # loop over board rows
        for row in range(3):
            # loop over board columns
            for col in range(3):
                board_string += ' %s' % self.position[row, col]

            # print new line every row
            board_string += '\n'

        # prepend side to move
        if self.player_1 == 'x':
            board_string = '\n--------------\n "x" to move:\n--------------\n\n' + board_string

        elif self.player_1 == 'o':
            board_string = '\n--------------\n "o" to move:\n--------------\n\n' + board_string

        # return board string
        return board_string


# main driver
if __name__ == '__main__':
    # create board instance
    board = Board()

    # start game loop
    board.game_loop()
