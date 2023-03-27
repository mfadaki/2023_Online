from No_Episodes import *

epsilon_start = 1.0  # 100% selecting the action randomely
epsilon_end = 0.01  # 1% selecting the action randomely
epsilon_decay_exp = 0.995
epsilon = epsilon_start

epsilon_decay_lin = (epsilon_start - epsilon_end) / num_episodes

epsiolon_decay_types = ['exponential', 'linear']
epsiolon_decay_type = epsiolon_decay_types[1]


def EpsilonDecay(epsiolon_decay_type, epsilon, i_episode, epsilon_decay_lin):
    if epsiolon_decay_type == 'exponential':
        epsilon = max(epsilon_end, epsilon_start * epsilon_decay_exp ** i_episode)
    else:
        epsilon = max(epsilon_end, epsilon - epsilon_decay_lin)
    return epsilon
