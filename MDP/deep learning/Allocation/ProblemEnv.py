from StatesActions import *
import random

max_iteration = 50


class ProblemEnv():
    def __init__(self, model_param):
        for key, value in model_param.items():
            setattr(self, key, value)

        self.state = random.choice(states_one_hot_tensor)
        self.time = self.prp1
        self.demand_arr = []
        self.action_arr = []
        self.sc_arr = []
        self.fl_arr = []
        self.state_arr = []

    def reset(self):
        self.state = random.choice(states_one_hot_tensor)
        self.iter = max_iteration
        self.time = self.prp1
        self.demand_arr = []
        self.action_arr = []
        self.sc_arr = []
        self.fl_arr = []
        self.state_arr = []

        return self.state

    def step(self, state, action):
        state_org = StateOht2Org(state)
        action_org = ActionOht2Org(action)

        next_state_org = state_org + action_org

        if next_state_org <= 0:
            done = True
            #next_state = states_one_hot_tensor[0]
            #next_state = state
            next_state = random.choice(states_one_hot_tensor)
            reward = -1e6
        elif next_state_org >= 20:
            done = True
            #next_state = states_one_hot_tensor[len(feasible_states)-1]
            next_state = random.choice(states_one_hot_tensor)
            #next_state = state
            reward = -1e6
        else:
            self.iter -= 1
            next_state = StateOrg2Oht(next_state_org)
            d1 = 5
            d2 = 5
            a1 = min(d1, int(next_state_org*d1/(d1+d2)))
            a2 = min(d2, int(next_state_org*d2/(d1+d2)))
            if abs(self.target_fr1-(a1/d1)) < 0.1 and abs(self.target_fr2-(a2/d2)) < 0.1:
                reward = self.p * (a1+a2)-self.penalty1*self.target_fr1-(a1/d1)**2-self.penalty2*self.target_fr2-(a2/d2)**2
            else:
                reward = -100
            #reward = -self.penalty1*(self.target_fr1-(a1/d1))-self.penalty2*(self.target_fr2-(a2/d2))

            done = False

        if self.iter == 0:
            done = True

        return next_state, reward, done, {}
