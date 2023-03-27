from EpsilonDecay import *


def Episodes(num_episodes, env, agent, epsilon, epsiolon_decay_type):
    rewards = []

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        epsilon = EpsilonDecay(epsiolon_decay_type, epsilon, i_episode, epsilon_decay_lin)

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(state, action)
            agent.buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.train()

        rewards.append(episode_reward)
        if i_episode % 1 == 0:
            print(f"Episode {i_episode}: reward = {episode_reward}, epsilon = {round(epsilon,2)}")
