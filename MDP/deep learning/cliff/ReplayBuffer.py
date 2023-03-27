import random
import torch


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
        actions = torch.tensor([e[1].tolist() for e in experiences], dtype=torch.int64)
        rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float32)
        next_states = torch.tensor([e[3].tolist() for e in experiences], dtype=torch.float32)
        dones = torch.tensor([e[4] for e in experiences], dtype=torch.float32)
        return states, actions, rewards, next_states, dones
