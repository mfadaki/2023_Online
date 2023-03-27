import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define hyperparameters
state_size = 2
action_size = 3
hidden_size = 64
lr = 0.001
gamma = 0.99  # Discount Rate
buffer_size = 50_000  # Max of transitions that we want to store before overwriting old transitions
batch_size = 64  # No of transitions that we want to sample From Replay buffer to compute the gradient

epsilon_start = 1.0  # 100% selecting the action randomely
epsilon_end = 0.01  # 1% selecting the action randomely
epsilon_decay = 0.995
num_episodes = 1000

# ??? TARGET_UPDATE_FREQ = 1000  # No of steps where we set the target parameters equal to the online parameters
# ??? MIN_REPLAY_SIZE = 1000  # No of transitions that we want in the replay buffer before we star to compute the gradient and oing training

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
        states = torch.tensor([e[0] for e in experiences], dtype=torch.float32)
        actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64)
        rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float32)
        next_states = torch.tensor([e[3] for e in experiences], dtype=torch.float32)
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

    def act(self, state, epsilon):  # Select the action based on the state using the e-greedy
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.main_network(state)
                action = torch.argmax(q_values).item()
        else:
            action = random.choice(np.arange(self.action_size))
        return action

    def train(self):
        if len(self.buffer.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        q_values = self.main_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        for target_param, main_param in zip(self.target_network.parameters(), self.main_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.gamma) + main_param.data * self.gamma)

# Define the Cliff Problem environment


class ProblemEnv():
    def __init__(self):
        self.height = 4
        self.width = 12
        self.start_state = (self.height-1, 0)
        self.goal_state = (self.height-1, self.width-1)
        self.cliff = [(self.height-1, j) for j in range(1, self.width-1)]

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, state, action):
        row, col = state
        if action == 0:  # Up
            row = max(row-1, 0)
        elif action == 1:  # Down
            row = min(row+1, self.height-1)
        # elif action == 2:  # Left
            # col = max(col-1, 0)
        elif action == 2:  # Right
            col = min(col+1, self.width-1)
        else:
            raise ValueError("Invalid action")
        next_state = (row, col)
        reward = -10
        done = False
        if next_state[0] > self.height or next_state[1] > self.width:
            reward = -1000
            done = True
        if next_state in self.cliff:
            reward = -1000
            done = True
            # next_state = self.start_state
            # done = True
        elif next_state == self.goal_state:
            done = True
            reward = 100
        return next_state, reward, done, {}


# Initialize the agent and environment
env = ProblemEnv()
agent = DQNAgent(state_size, action_size, hidden_size, lr, gamma, buffer_size, batch_size)

rewards = []
for i_episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    epsilon = max(epsilon_end, epsilon_start * epsilon_decay ** i_episode)

    while not done:
        action = agent.act(state, epsilon)
        # print(f"state={state}")
        # print(f"action={action}")
        next_state, reward, done, _ = env.step(state, action)
        agent.buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        agent.train()

    rewards.append(episode_reward)
    if i_episode % 1 == 0:
        print(f"Episode {i_episode}: reward = {episode_reward}, epsilon = {epsilon}")

# Plot
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

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
