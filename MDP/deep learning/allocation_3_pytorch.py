import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

max_iteration = 100
# Model Parameters
model_param = {
    'prp1': 200,
    'penalty1': 20,
    'penalty2': 20,
    'target_fr1': 0.9,
    'target_fr2': 0.9,
    'hc': 10,
    'sc': 0.5,
    's': 10,
    'p': 100,
    'prob_disrupt': 0.1,
    'disr_ratio': 0.1
}

# Define hyperparameters

hidden_size = 64
lr = 0.001
gamma = 0.99  # Discount Rate
buffer_size = 10_000  # Max of transitions that we want to store before overwriting old transitions
batch_size = 64  # No of transitions that we want to sample From Replay buffer to compute the gradient

num_episodes = 1_000

# Exponnetional Epsilon Decay
epsilon_start = 1.0  # 100% selecting the action randomely
epsilon_end = 0.01  # 1% selecting the action randomely
epsilon_decay = 0.995
# Linear Epsilon Decay
epsilon = epsilon_start
epsilon_decay = (epsilon_start - epsilon_end) / num_episodes

target_update_freq = 10  # No of steps where we set the target NN parameters would be copined (equalized) from the Main NN parameters

min_d = 5
max_d = 15

# Create one_hot tensor for Actions
# feasible_actions = [-1, 0, 1]
feasible_actions = range(-15, 16)
# for i in range(0, model_param['s']+1):
#     for j in range(0, model_param['s']+1):
#         if i+j <= model_param['s']:
#             feasible_actions.append([i, j])

actions_one_hot_tensor = torch.eye(len(feasible_actions))
action_size = len(feasible_actions)

# Create one_hot tensor for States
# if model_param['prob_disrupt'] == 0:
#     s_options = [model_param['s']]
# else:
#     s_options = [int(model_param['s']*(1-model_param['disr_ratio'])), model_param['s']]
# min_d = int(model_param['s']/1.8)
# max_d = int(model_param['s']*1.8)
# feasible_states = [[i, j, k] for i in s_options for j in range(min_d, max_d+1) for k in range(min_d, max_d+1)]

feasible_states = range(1, 31)
states_one_hot_tensor = torch.eye(len(feasible_states))
state_size = len(feasible_states)

# ### Define the Deep Q-Network (DQN)


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)  # Input layer. fc1 represents Fully Connected layer 1
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):  # forward is the general function in any class which is callable by default. After initiating an instance, if we call the instance, this function is run.
        # This function actually creates the NN
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the replay buffer


class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.tensor([e[0].tolist() for e in experiences], dtype=torch.float32)
        actions = torch.LongTensor([e[1].tolist() for e in experiences])
        # actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64)
        rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float32)
        next_states = torch.tensor([e[3].tolist() for e in experiences], dtype=torch.float32)
        dones = torch.tensor([e[4] for e in experiences], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

# Define the Agent


class DQNAgent():
    def __init__(self, state_size, action_size, hidden_size, lr, gamma, buffer_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.main_network = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQN(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)
        self.actions_one_hot_tensor = actions_one_hot_tensor

    def q_values_feasibility_update(self, state, q_values):
        state_index = torch.where(state == 1)[0].item()
        state_org = feasible_states[state_index]
        _ = 0
        for action_ in feasible_actions:
            if action_ + state_org < 0 and action_ + state_org > 30:
                q_values[_] = float('-inf')
            _ += 1

        return q_values

    def state_action_feasibility(self, state):
        state_index = torch.where(state == 1)[0].item()
        state_org = feasible_states[state_index]

        feasible_state_actions = []
        for action_ in feasible_actions:
            if action_ + state_org >= 0 and action_ + state_org <= 30:
                action_index_fn = feasible_actions.index(action_)
                action_oh = actions_one_hot_tensor[action_index_fn]
                feasible_state_actions.append(action_oh)

        return feasible_state_actions

    def act(self, state, epsilon):  # Select the action based on the state using the e-greedy
        if random.random() > epsilon:
            with torch.no_grad():
                # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.main_network(state)
                #q_values = self.q_values_feasibility_update(state, q_values)
                action_index = torch.argmax(q_values).item()
                action = actions_one_hot_tensor[action_index]

        else:
            # action = random.choice(np.arange(self.action_size))
            # action = random.randint(0, 2)
            # if state[3] == 0:
            # action_real = [min(state[0], state[1]), min(state[0]-min(state[0], state[1]), state[2])]
            # else:
            # action_real = [min(state[0]-min(state[0], state[2]), state[1]), min(state[0], state[2])]

            # calculate index of action in tensor
            # action_index = feasible_actions.index(action_real)

            # get one-hot encoding vector for action
            # action = actions_one_hot_tensor[action_index]

            feasible_state_actions_tensor = self.state_action_feasibility(state)
            action = random.choice(feasible_state_actions_tensor)

            # action = random.choice(actions_one_hot_tensor)

        return action

    def train(self, i_episode):
        if len(self.buffer.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # states = states.unsqueeze(-1)
        # actions = actions.unsqueeze(1)
        # rewards = rewards.unsqueeze(1)
        # dones = dones.unsqueeze(1)
        # next_states = states.unsqueeze(-1)

        # Use this q_values method is actions are 0,1,2,3...
        # q_values = self.main_network(states).gather(1, actions)
        # To compute q_values, states with shape of [batch_size * S] are passed to the main NN, the result will be q_values for all states. If we multiply this value by the Actions, since actions are one-hot tensors, the result for each action will be the q-value of relevant action since the rest of items in the action tensor are zero. Now, getting sum converts the tensor of many zeros and a q_value to just the q_value for that particular action.
        q_values = torch.sum(self.main_network(states) * actions, dim=1)

        with torch.no_grad():
            # next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        if i_episode % target_update_freq == 0:
            # Method 1 works!
            # for target_param, main_param in zip(self.target_network.parameters(), self.main_network.parameters()):
            # target_param.data.copy_(target_param.data * (1.0 - self.gamma) + main_param.data * self.gamma)
            # Method 2 works!
            self.target_network.load_state_dict(self.main_network.state_dict())
        return loss.item()

# Define the Problem environment


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
        state_index = torch.where(state == 1)[0].item()
        state_org = feasible_states[state_index]

        action_index = torch.where(action == 1)[0].item()
        action_org = feasible_actions[action_index]

        next_state_org = state_org + action_org

        if next_state_org < 0:
            done = True
            next_state = states_one_hot_tensor[0]
            avr_reward = float('-inf')
        elif next_state_org >= 30:
            done = True
            next_state = states_one_hot_tensor[len(feasible_states)-1]
            avr_reward = float('-inf')
        else:
            self.iter -= 1
            next_state_index = feasible_states.index(next_state_org+1)
            next_state = states_one_hot_tensor[next_state_index]

            S = next_state_org

            tot_reward = 0
            for t in range(self.prp1):
                d1 = random.randint(min_d, max_d)
                d2 = random.randint(min_d, max_d)
                a1 = min(d1, int(S*d1/(d1+d2)))
                a2 = min(d2, int(S*d2/(d1+d2)))
                # reward_t = self.p*(a1+a2)-self.sc*(max(0, (d1+d2)-(a1+a2)))-self.hc*(max(0, S-(a1+a2)))
                reward_t = self.p*(a1+a2)-self.hc*(max(0, S-(a1+a2)))
                tot_reward += reward_t

            avr_reward = tot_reward / self.prp1

        if self.iter == 0:
            done = True
        else:
            done = False

        reward = avr_reward

        # state_index = torch.where(state == 1)[0].item()
        # state_org = feasible_states[state_index]
        # self.state_arr.append(np.array(state_org))

        # self.demand_arr.append([state_org[1], state_org[2]])

        # action_index = torch.where(action == 1)[0].item()
        # action_org = feasible_actions[action_index]
        # self.action_arr.append(np.array(action_org))

        # actions_one_hot_tensor_list = actions_one_hot_tensor.tolist()
        # action_index = actions_one_hot_tensor_list.index(action.tolist())
        # action_real1 = feasible_actions[action_index]

        # action_index = feasible_actions.index(action_real)
        # feasible_actions.index([19, 1])
        # actions_one_hot_tensor[229]
        # torch.argmax(actions_one_hot_tensor[229]).item()

        # self.action_arr.append(np.array(action_real1))

        # self.sc_arr.append(self.sc*(max(0, state_org[0]-(sum(action_org)))))
        # # sum(map(sum, m))

        # fl1 = np.sum([item[0] for item in self.action_arr]) / np.sum([item[0] for item in self.demand_arr])
        # fl2 = np.sum([item[1] for item in self.action_arr]) / np.sum([item[1] for item in self.demand_arr])
        # self.fl_arr.append([round(fl1, 2), round(fl2, 2)])

        # d1 = random.randint(min_d, max_d)
        # d2 = random.randint(min_d, max_d)
        # next_state_org = [int(random.choices([self.s, self.s*(1-self.disr_ratio)], weights=((1-self.prob_disrupt), self.prob_disrupt), k=1)[0]),
        #                   d1,
        #                   d2]
        # next_state_index = feasible_states.index(next_state_org)
        # next_state = states_one_hot_tensor[next_state_index]

        # self.time -= 1

        # Check done and reward
        # if self.time <= 0:
        #     done = True
        #     #reward = self.p*(sum(action_org))-self.sc*(max(0, state_org[0]-(sum(action_org)))) - max(0, self.target_fr1-fl1)*self.penalty1 - max(0, self.target_fr2-fl2)*self.penalty2
        #     reward = self.p*(sum(action_org))-self.sc*(max(0, state_org[0]-(sum(action_org))))

        # # elif state_org[1] < action_org[0] or state_org[2] < action_org[1]:
        #     #done = True
        #     #reward = -1e10
        # else:
        #     done = False
        #     reward = self.p*(sum(action_org))-self.sc*(max(0, state_org[0]-(sum(action_org))))
        #     #reward = self.p*(sum(action_org))-self.sc*(max(0, state_org[0]-(sum(action_org)))) - max(0, self.target_fr1-fl1)*self.penalty1 - max(0, self.target_fr2-fl2)*self.penalty2

        #     # reward = self.p*(np.sum(self.action_arr)) - \
        #     #     self.sc*(max(0, np.sum(self.demand_arr) -
        #     #                  np.sum(self.action_arr)))-(max(0, self.target_fr1-(np.sum([item[0] for item in self.action_arr])/np.sum([item[0] for item in self.demand_arr])))*self.penalty1) - \
        #     #     (max(0, self.target_fr1-(np.sum([item[1] for item in self.action_arr])/np.sum([item[1] for item in self.demand_arr])))*self.penalty2)

        return next_state, reward, done, {}


# Initialize the agent and environment
env = ProblemEnv(model_param)
agent = DQNAgent(state_size, action_size, hidden_size, lr, gamma, buffer_size, batch_size)

rewards = []
running_loss = []

for i_episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    episode_loss = 0.0
    # Exponential Decay
    # epsilon = max(epsilon_end, epsilon_start * epsilon_decay ** i_episode)
    # Linear Decay
    epsilon = max(epsilon_end, epsilon - epsilon_decay)

    counter = 0

    while not done:
        action = agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(state, action)
        agent.buffer.add(state, action, reward, next_state, done)
        state = next_state
        counter += 1
        episode_reward += reward
        try:
            episode_loss += float(agent.train(i_episode))
        except:
            agent.train(i_episode)

    rewards.append(episode_reward)
    running_loss.append(round(episode_loss/counter, 1))

    state_index = torch.where(state == 1)[0].item()
    state_org = feasible_states[state_index]

    if i_episode % 100 == 0:
        print(f"Episode {i_episode}: State = {state_org}, reward = {episode_reward}, episode_loss = {episode_loss}, epsilon = {round(epsilon,2)}")

# Saving and loading model
torch.save(agent.main_network, './NN_model.pt')

# Loading
# We should run all part of the code except for the Episodes part and then load the model
NN_model = torch.load('./NN_model.pt')
for state_tensor in states_one_hot_tensor:
    q_values = NN_model(torch.FloatTensor(state_tensor)).detach().numpy()
    action = np.argmax(q_values)
    print(action)


for state_tensor in states_one_hot_tensor:
    q_values = agent.main_network(torch.FloatTensor(state_tensor)).detach().numpy()
    action_ind = np.argmax(q_values)
    action = feasible_actions[action_ind]
    if action == 0:
        print(f'q_values={q_values}')
    print(action)


# Plot
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


"""
# Print the optimal policy
height = 4
width = 12
goal_state = (height-1, width-1)
cliff = [(height-1, j) for j in range(1, width-1)]
for row in range(height):
    for col in range(width):
        state = (row, col)
        if state == goal_state:
            print(" G ", end="")
        elif state in cliff:
            print(" C ", end="")
        else:
            q_values = agent.main_network(torch.FloatTensor(state)).detach().numpy()
            action = np.argmax(q_values)
            if action == 0:
                print(" ^ ", end="")
            elif action == 1:
                print(" v ", end="")
            elif action == 2:
                print(" > ", end="")
    print()
"""
