from ReplayBuffer import *
from DQN import *
from EpsilonDecay import *
from StatesActions import *
from Agent import *
from ProblemEnv import *
from Episodes import *


# epsilon_decay_lin = (epsilon_start - epsilon_end) / num_episodes

# Initialize the agent and environment
env = ProblemEnv()
agent = DQNAgent(state_size, action_size, hidden_size, lr, gamma, buffer_size, batch_size)

Episodes(num_episodes, env, agent, epsilon, epsiolon_decay_type)


# Plot
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# Print the optimal policy
height = 4
width = 12
goal_state = [height-1, width-1]
cliff = [[height-1, j] for j in range(1, width-1)]
for row in range(height):
    for col in range(width):
        state_org = [row, col]
        state = StateOrg2Oht(state_org)
        if state_org == goal_state:
            print(" G ", end="")
        elif state_org in cliff:
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
