import os
import matplotlib.pyplot as plt


def graph(s, pi):
    print(len(s))
    print(len(pi))
    pi_list = [pi[_] for _ in s]
    plt.scatter(s, pi_list)
    plt.xlabel('s')
    plt.ylabel('Policy')
    plt.show()

# Model (MDP problem)


class GamblerMDP(object):
    def __init__(self, N, p_success):
        # N = number of blocks
        self.N = N
        self.p_success = p_success

    def startState(self):
        return 1

    def isEnd(self, state):
        return state == self.N

    def actions(self, state):
        # return list of valid actions
        return [*range(0, min(state, self.N-state)+1)]

    def succProbReward(self, state, action):
        # return list of (newState, prob, reward) triples for the successor (next) state
        # state = s, action = a, newState = s'
        # prob = T(s, a, s'), reward = Reward(s, a, s')
        result = []
        if self.isEnd(state):
            result.append((int(state), 1, 1))
        else:
            if int(state+action) == self.N:
                result.append((int(state+action), self.p_success, 1))
            else:
                result.append((int(state+action), self.p_success, 0))
            result.append((int(state-action), 1-self.p_success, 0))
        return result

    def discount(self):
        return 1

    def states(self):
        return range(1, self.N)


#####################################################

# Inference (Algorithms)


def valueIteration(mdp):
    # initialize
    V = {}  # state -> Vopt[state]
    for state in mdp.states():
        V[state-1] = 0.
    V[state] = 0.
    V[state+1] = 0.

    def Q(state, action):
        return sum(prob*(reward + mdp.discount()*V[newState])
                   for newState, prob, reward in mdp.succProbReward(state, action))

    while True:
        # compute the new values (newV) given the old values (V)
        newV = {}
        # for state in mdp.states():
        for state in range(0, max(mdp.states())+2):
            if state == 0 or state == max(mdp.states())+1:
                newV[state] = 0
            else:
                newV[state] = max(Q(state, action) for action in mdp.actions(state))
        # check for convergence
        if max(abs(V[state]-newV[state]) for state in mdp.states()) < 1e-10:
            print(len(pi))
            print(pi)
            graph(mdp.states(), pi)
            break
        V = newV

        # read out policy
        pi = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                pi[state] = 'none'
            else:
                pi[state] = max((Q(state, action), action) for action in mdp.actions(state))[1]
        # print(pi)
        # print stuff out
        os.system('clear')
        print('{:20} {:20} {:20}'.format('s', 'V(s)', 'pi(s)'))
        for state in mdp.states():
            print('{:20} {:20} {:20}'.format(state, V[state], pi[state]))
        # input()


# Execution
mdp = GamblerMDP(N=100, p_success=0.4)
valueIteration(mdp)
