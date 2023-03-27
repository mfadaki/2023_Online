import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
obs = env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
tf.config.list_physical_devices("GPU")

a = 10.345
print(f'The test is {a=:.2f}')
print(f'The test is {a=:.2f}')
