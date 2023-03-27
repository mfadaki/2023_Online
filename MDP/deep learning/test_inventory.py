import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class InventorySystem:
    def __init__(self, capacity, init_state, price, cost, holding_cost, penalty_cost, max_action, min_action, mean):
        self.capacity = capacity
        self.state = init_state
        self.price = price
        self.cost = cost
        self.holding_cost = holding_cost
        self.penalty_cost = penalty_cost
        self.max_action = max_action
        self.min_action = min_action
        self.mean = mean
        self.time = 0

    def reset(self):
        self.state = self.capacity // 2
        self.time = 0

    def step(self, action):
        # Compute demand for this time step
        demand = int(np.random.normal(self.mean, np.sqrt(self.mean)))
        demand = max(min(demand, self.max_action), self.min_action)

        # Update inventory level based on action and demand
        self.state = max(min(self.state + action - demand, self.capacity), 0)

        # Compute reward
        reward = -self.cost * action
        reward += self.price * min(demand, self.state)
        reward -= self.holding_cost * self.state
        if self.state < demand:
            reward -= self.penalty_cost * (demand - self.state)

        # Update time step
        self.time += 1

        # Check if episode is over
        done = self.time == 12

        # Update done flag
        if done:
            self.reset()

        return self.state, reward, done


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, lr, gamma, epsilon, epsilon_decay, min_epsilon):
        self.q_network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.num_actions = 11

    def act(self, state):
        if np.random.uniform() < self.epsilon:
            # Explore: take a random action
            action = np.random.randint(0, self.num_actions)
        else:
            # Exploit: take the best action based on the Q-values
            state_tensor = torch.tensor([[state]], dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        return min(max(action, 0), self.num_actions-1)  # Ensure the action is within valid range

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor([[state]], dtype=torch.float32)
        next_state_tensor = torch.tensor([[next_state]], dtype=torch.float32)
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)
        q_value = q_values[0][action]
        target = reward + (1 - done) * self.gamma * next_q_values.max()
        loss = F.smooth_l1_loss(q_value, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)


def train_agent(agent, env, num_episodes):
    state = env.reset()
    total_reward = 0
    for episode in range(num_episodes):
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.train(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        state = env.reset()
        agent.update_epsilon()
        print("Episode: {:3d} | Reward: {:3d} | Epsilon: {:.3f}".format(episode+1, total_reward, agent.epsilon))
        total_reward = 0


env = InventorySystem(capacity=20, init_state=10, price=1, cost=0.5, holding_cost=0.1, penalty_cost=2, max_action=5, min_action=0, mean=10)
agent = DQNAgent(lr=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01)
train_agent(agent, env, num_episodes=1000)
