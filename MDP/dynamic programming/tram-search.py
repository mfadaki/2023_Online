import sys
import util
import pandas as pd
sys.setrecursionlimit(10000)

# Model (search problem)


class TransportationProblem(object):
    def __init__(self, N):
        # N = number of blocks
        self.N = N

    def startState(self):
        return 1

    def isEnd(self, state):
        return state == self.N

    def succAndCost(self, state):
        # return list of (action, newState, cost) triples
        result = []
        if state+1 <= self.N:
            result.append(('walk', state+1, 1))
        if state*2 <= self.N:
            result.append(('tram', state*2, 2))
        return result

# Algorithms


def printSolution(solution):
    totalCost, history = solution
    print('totalCost: {}'.format(totalCost))
    for item in history:
        print(item)


def printSolution_dp(dp_policy):
    opt_States = []
    opt_nextState = []
    opt_actions = []
    opt_cost = []
    state = problem.startState()
    while state < problem.N:
        opt_States.append(state)
        opt_nextState.append(dp_policy[state])
        if dp_policy[state]-state == 1:
            opt_actions.append('walk')
            opt_cost.append(1)
        else:
            opt_actions.append('tram')
            opt_cost.append(2)

        state = dp_policy[state]

    optimal_path = pd.DataFrame(
        {'Current State': opt_States,
         'Next State': opt_nextState,
         'Action': opt_actions,
         'Cost': opt_cost
         })
    print('######## Dynamic Programming')
    print('This is the Optimal (min) Cost for each state to the end of network:')
    print(cache)
    print('This is the Optimal Actions for each state (from state:to state):')
    print(dp_policy)
    print('This is the optimal path:')
    print(optimal_path)


def backtrackingSearch(problem):
    # Best solution found so far (dictionary because of python scoping technicality)
    best = {
        'cost': float('+inf'),
        'history': None
    }

    def recurse(state, history, totalCost):
        # At state, having undergone history, accumulated
        # totalCost.
        # Explore the rest of the subtree under state.
        if problem.isEnd(state):
            # Update the best solution so far
            if totalCost < best['cost']:
                best['cost'] = totalCost
                best['history'] = history
            return
        # Recurse on children
        for action, newState, cost in problem.succAndCost(state):
            recurse(newState, history+[(action, newState, cost)], totalCost+cost)
    recurse(problem.startState(), history=[], totalCost=0)
    return (best['cost'], best['history'])


def dynamicProgramming(problem):
    global cache  # state -> futureCost(state)
    cache = {}
    global dp_policy
    dp_policy = {}

    def futureCost(state):
        # Base case
        if problem.isEnd(state):
            return 0
        if state in cache:  # Exponential savings
            return cache[state]
        # Actually doing work
        result = min(cost+futureCost(newState)
                     for action, newState, cost in problem.succAndCost(state))
        # argmin min value shows the optimal action
        policy_state = min((cost+futureCost(newState), newState)
                           for action, newState, cost in problem.succAndCost(state))[1]
        cache[state] = result
        dp_policy[state] = policy_state
        # print(dp_policy)
        # print(cache)
        # print(cache)
        return result
    return (futureCost(problem.startState()), [])


def uniformCostSearch(problem):
    frontier = util.PriorityQueue()
    frontier.update(problem.startState(), 0)
    while True:
        # Move from frontier to explored
        state, pastCost = frontier.removeMin()
        if problem.isEnd(state):
            return (pastCost, [])
        # Push out on the frontier
        for action, newState, cost in problem.succAndCost(state):
            frontier.update(newState, pastCost+cost)


# Main
problem = TransportationProblem(N=40)
# print(problem.succAndCost(3))
# print(problem.succAndCost(9))
printSolution(backtrackingSearch(problem))
printSolution(dynamicProgramming(problem))
printSolution_dp(dp_policy)
printSolution(uniformCostSearch(problem))
