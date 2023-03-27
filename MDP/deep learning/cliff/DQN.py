import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define hyperparameters

hidden_size = 64

buffer_size = 50_000  # Max of transitions that we want to store before overwriting old transitions
batch_size = 64  # No of transitions that we want to sample From Replay buffer to compute the gradient

epsilon_start = 1.0  # 100% selecting the action randomely
epsilon_end = 0.01  # 1% selecting the action randomely
epsilon_decay = 0.995


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
