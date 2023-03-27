import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
torch.set_grad_enabled(True)

# Model Parameters
model_param = {
    'prob_disrupt': 0.2,
    'N': 20
}

# Define hyperparameters

hidden_size = 64
lr = 0.01
gamma = 0.99  # Discount Rate
buffer_size = 10_000  # Max of transitions that we want to store before overwriting old transitions
batch_size = 64  # No of transitions that we want to sample From Replay buffer to compute the gradient

num_episodes = 1000

# Exponnetional Epsilon Decay
epsilon_start = 1.0  # 100% selecting the action randomely
epsilon_end = 0.01  # 1% selecting the action randomely
epsilon_decay = 0.995
# Linear Epsilon Decay
epsilon = epsilon_start
epsilon_decay = (epsilon_start - epsilon_end) / num_episodes

target_update_freq = 5  # No of steps where we set the target NN parameters would be copined (equalized) from the Main NN parameters

# Create one_hot tensor for Actions
feasible_actions = ['walk', 'tram']
actions_one_hot_tensor = torch.eye(len(feasible_actions))
action_size = len(feasible_actions)  # Compute the action size

# Create one_hot tensor for States
feasible_states = np.array(range(0, model_param['N']))
states_one_hot_tensor = torch.eye(len(feasible_states))
state_size = len(feasible_states)  # Compute the state size

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
        #actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64)
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

    def act(self, state, epsilon):  # Select the action based on the state using the e-greedy
        if random.random() > epsilon:
            with torch.no_grad():
                #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.main_network(state)
                action_index = torch.argmax(q_values).item()
                action = actions_one_hot_tensor[action_index]

        else:
            state_org = torch.where(state == 1)[0].item()  # In the state one_hot tensor, the index of '1' shows the original state
            if state_org+1 == model_param['N'] or (state_org * 2 > model_param['N']):
                action = torch.tensor([1., 0.])
            else:
                action = random.choice(actions_one_hot_tensor)
        return action

    def train(self, i_episode):
        if len(self.buffer.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        #states = states.unsqueeze(-1)
        #actions = actions.unsqueeze(1)
        #rewards = rewards.unsqueeze(1)
        #dones = dones.unsqueeze(1)
        #next_states = states.unsqueeze(-1)

        # Use this q_values method is actions are 0,1,2,3...
        # q_values = self.main_network(states).gather(1, actions)
        # To compute q_values, states with shape of [batch_size * S] are passed to the main NN, the result will be q_values for all states. If we multiply this value by the Actions, since actions are one-hot tensors, the result for each action will be the q-value of relevant action since the rest of items in the action tensor are zero. Now, getting sum converts the tensor of many zeros and a q_value to just the q_value for that particular action.
        q_values = torch.sum(self.main_network(states) * actions, dim=1)

        with torch.no_grad():
            #next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        # Shape of both q_values and targets should be the same and generally tensor of [batch-size]
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        if i_episode % target_update_freq == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
        return loss.item()

# Define the Problem environment


class ProblemEnv():
    def __init__(self, model_param):
        for key, value in model_param.items():
            setattr(self, key, value)

        #self.start_state = states_one_hot_tensor[0]
        self.state = states_one_hot_tensor[0]
        self.time = self.N

    def reset(self):
        self.state = states_one_hot_tensor[0]
        self.time = self.N

        return self.state

    def step(self, state, action):

        state_org = torch.where(state == 1)[0].item()
        state_org += 1
        if torch.equal(action, actions_one_hot_tensor[0]):  # if equal for two tensors
            next_state_org = state_org + 1
            reward = -1
        elif torch.equal(action, actions_one_hot_tensor[1]):
            if random.random() < self.prob_disrupt or state_org * 2 > self.N:
                next_state_org = state_org
                reward = -2
            else:
                next_state_org = state_org * 2
                reward = -2
        else:
            raise ValueError("Invalid Action")

        #self.time -= 1

        #state = torch.eye(self.N)[state_org - 1]
        next_state = torch.eye(self.N)[next_state_org - 1]

        # Check done and reward
        if next_state_org < self.N:
            done = False
        else:
            done = True

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
    #epsilon = max(epsilon_end, epsilon_start * epsilon_decay ** i_episode)
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

    if i_episode % 1 == 0:
        print(f"Episode {i_episode}: State = {state}, reward = {episode_reward}, episode_loss = {episode_loss}, epsilon = {round(epsilon,2)}")

# Saving and loading model
torch.save(agent.main_network, './NN_model.pt')

# Loading
NN_model = torch.load('./NN_model.pt')
for state in states_one_hot_tensor:
    q_values = NN_model(torch.FloatTensor(state)).detach().numpy()
    action = np.argmax(q_values)
    print(action)

for param_tensor in NN_model.state_dict():
    print(param_tensor, "\t", NN_model.state_dict()[param_tensor].size())

# Plot
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# Print the optimal policy

# for i in range(1, model_param['N']+1):
for state in states_one_hot_tensor:
    #state = torch.tensor(i, dtype=torch.float32).unsqueeze(-1)
    q_values = agent.main_network(torch.FloatTensor(state)).detach().numpy()
    action = np.argmax(q_values)
    print(action)
