from StatesActions import *
import random


class ProblemEnv():
    def __init__(self):
        self.height = 4
        self.width = 12
        self.start_state = StateOrg2Oht([self.height-1, 0])
        self.goal_state = [self.height-1, self.width-1]
        self.cliff = [[self.height-1, j] for j in range(1, self.width-1)]

    def reset(self):
        #self.state = StateOrg2Oht([0, 0])
        self.state = random.choice(states_one_hot_tensor)
        return self.state

    def step(self, state, action):
        state_org = StateOht2Org(state)
        action_org = ActionOht2Org(action)

        row, col = state_org
        if action_org == 0:  # Up
            #row = max(row-1, 0)
            row = row-1
        elif action_org == 1:  # Down
            #row = min(row+1, self.height-1)
            row = row+1
        # elif action == 2:  # Left
            # col = max(col-1, 0)
        elif action_org == 2:  # Right
            #col = min(col+1, self.width-1)
            col = col+1
        else:
            raise ValueError("Invalid action")

        next_state_org = [row, col]

        reward = -10
        done = False

        if next_state_org[0] > self.height-1 or next_state_org[0] < 0 or next_state_org[1] > self.width-1 or next_state_org[1] < 0:
            reward = -1000000
            done = True
        if next_state_org in self.cliff:
            reward = -1000000
            done = True
            # next_state = self.start_state
            # done = True
        if next_state_org == self.goal_state:
            done = True
            reward = 1000000

        if done == True:
            next_state = StateOrg2Oht([self.height-1, 0])
        else:
            next_state = StateOrg2Oht(next_state_org)

        return next_state, reward, done, {}
