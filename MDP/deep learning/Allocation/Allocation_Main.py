from ReplayBuffer import *
from DQN import *
from EpsilonDecay import *
from StatesActions import *
from Agent import *
from ProblemEnv import *
from Episodes import *


# epsilon_decay_lin = (epsilon_start - epsilon_end) / num_episodes

# Initialize the agent and environment
env = ProblemEnv(model_param)
agent = DQNAgent(state_size, action_size, hidden_size, lr, gamma, buffer_size, batch_size)

Episodes(num_episodes, env, agent, epsilon, epsiolon_decay_type)


# Print the optimal policy
for state_tensor in states_one_hot_tensor:
    q_values = agent.main_network(torch.FloatTensor(state_tensor)).detach().numpy()
    action_ind = np.argmax(q_values)
    action = feasible_actions[action_ind]
    # if action == -1:
    # print(f'q_values={q_values}')
    print(action)

# Plot
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
