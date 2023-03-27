import torch
from ReplayBuffer import *
from DQN import *
from StatesActions import *

lr = 0.1
gamma = 0.99  # Discount Rate

target_update_freq = 10  # No of steps where we set the target NN parameters would be copined (equalized) from the Main NN parameters


class DQNAgent():
    def __init__(self, state_size, action_size, hidden_size, lr, gamma, buffer_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.main_network = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQN(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)
        self.actions_one_hot_tensor = actions_one_hot_tensor

    def q_values_feasibility_update(self, state, q_values):
        state_org = StateOht2Org(state)
        _ = 0
        for action_ in feasible_actions:
            if action_ + state_org < 0 or action_ + state_org > 20:
                q_values[_] = float('-inf')
            _ += 1

        return q_values

    def state_action_feasibility(self, state):
        state_org = StateOht2Org(state)

        feasible_state_actions = []
        for action_ in feasible_actions:
            if action_ + state_org >= 0 and action_ + state_org <= 10:
                action_oht = ActionOrg2Oht(action_)
                feasible_state_actions.append(action_oht)

        return feasible_state_actions

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def softmax_action_selection(self, state):
        state_org = StateOht2Org(state)
        next_state_orgs = [i+state_org for i in feasible_actions]

        #feasible_actions = range(-20, 21)
        SL_target = 0.9
        d1 = 5
        d2 = 5
        SL_MSE = []
        c = 0
        for _ in next_state_orgs:
            if _ < 0 or _ > 20:
                SL_MSE.append(-1e6)
            else:
                a1 = min(d1, int(_*d1/(d1+d2)))
                a2 = min(d2, int(_*d2/(d1+d2)))
                SL_MSE.append(-((a1+a2)/(d1+d2)-SL_target)**2)
            c += 1
        # compute softmax probabilities
        probs = self.softmax(SL_MSE)

        # sample from the resulting probability distribution
        sample_ind = np.random.choice(len(probs), p=probs)
        action_softmax = actions_one_hot_tensor[sample_ind]

        return action_softmax

    def act(self, state, epsilon):  # Select the action based on the state using the e-greedy
        if random.random() > epsilon:
            with torch.no_grad():
                #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.main_network(state)
                #q_values = self.q_values_feasibility_update(state, q_values)
                action_index = torch.argmax(q_values).item()
                action = actions_one_hot_tensor[action_index]
        else:
            #feasible_state_actions_tensor = self.state_action_feasibility(state)
            #action = random.choice(feasible_state_actions_tensor)

            action = random.choice(actions_one_hot_tensor)
            #action = self.softmax_action_selection(state)
        return action

    def train(self):
        if len(self.buffer.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        q_values = torch.sum(self.main_network(states) * actions, dim=1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
        return loss.item()
