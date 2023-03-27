# ##### Imports
import torch
from torch import nn
import gym
from collections import deque  # A lost which we can add an element at both front or end of it
import itertools
import numpy as np
import random

# ##### Hyperparameters
GAMMA = 0.99  # Discount Rate
BATCH_SIZE = 32  # No of transitions that we want to sample From Replay buffer to compute the gradient
BUFFER_SIZE = 50_000  # Max of transitions that we want to store before overwriting old transitions
MIN_REPLAY_SIZE = 1000  # No of transitions that we want in the replay buffer before we star to compute the gradient and oing training
EPSILON_START = 1.0  # 100% selecting the action randomely
EPSILON_END = 0.02  # 2% selecting the action randomely
EPSILON_DECAY = 10000  # Over these many stapes the epsilon decrreases from _start to _end
TARGET_UPDATE_FREQ = 1000  # No of steps where we set the target parameters equal to the online parameters


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))  # No of nurons in the input layer of the NN.
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)  # No of outputs in the NN is the no of available aactions. Note DQL here Only uses discrete actions.
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):  # Selecting an action in q-learning
        obs_t = torch.as_tensor(obs, dtype=torch.float32)  # Convert to Torch Tensor
        q_values = self(obs_t.unsqueeze(0))  # Compute the q_values for a specific observation for all actions that the agent can take. We need to unsqueeze to zero as every single operation in pytorch expoects a batch dimention. In the current problem though, we have no batch dimension.
        max_q_index = torch.argmax(q_values, dim=1)[0]  # Find the action with the highest q-value.
        action = max_q_index.detach().item()  # Tensor to integer
        return action  # an action now is a number between 0 and 1-no of available actions


env = gym.make('CartPole-v0')

replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)  # To store the rewards earned by our agent in a single episode. We do this to track the improvement of the agent as it trains.

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())  # make target the same as online

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# ##### Initialize the Replay Buffer
# Put the MIN_REPLAY_SIZE no of transitions into the Replay buffer before we start
# Observation here is the state of the system
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()  # Selecting a random action

    new_obs, rew, done, truncated, _ = env.step(action)  # Get all info of the new state
    transition = (obs, action, rew, done, new_obs)  # Tuple of all info for this transition
    replay_buffer.append(transition)  # We add this transition to the Replay Buffer
    obs = new_obs

    if done:
        obs = env.reset()

# ##### Training Loop
obs = env.reset()

for step in itertools.count():  # counting which step we are on using the itertools.count
    # Select the action to take. Based on e-greedy policy:
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, done, truncated, _ = env.step(action)  # Get all info of the new state
    transition = (obs, action, rew, done, new_obs)  # Tuple of all info for this transition
    replay_buffer.append(transition)  # We add this transition to the Replay Buffer
    obs = new_obs

    episode_reward += rew  # the current episode reward value is the computed reward

    if done:
        obs = env.reset()

    rew_buffer.append(episode_reward)  # Append the episode reard to the corresponding buffer
    episode_reward = 0.0

    # Start the Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)  # Sampling BATCH_SIZE no of transitions from our replay_buffer (that we added earlier). We already had this in tuple but we now need them individually.

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    # convert them to the Pytorch Tensor
    obses_t = torch.as_tensor(obs, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    # Compute Targets for the Loss Function
    target_q_values = target_net(new_obses_t)  # Get the target q_values for the new obs using the Target_net
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # Find the highest q_value for the observation. Note that in the batch dimension, the q-values of dimension 1. SO it means get the max value in dimension 1. Keepdim=True means keep that dimension around alsthough we just use the first dimentsion.

    targets = rews_t + GAMMA * (1-dones_t) * max_target_q_values

    # Compute Loss
    q_values = online_net(obses_t)  # get q_values for each observation

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)  # This time we get the q_value for the action that we are actually taking.

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()  # compute our gradients
    optimizer.step()  # apply those gradients

    # Update Target Networ every couple of steps
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print(f"step = ")
        print(f"AVG Rew = {np.mean(rew_buffer)}")
