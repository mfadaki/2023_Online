import numpy as np


def softmax(x):
    """Compute softmax values for input x"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


# example input vector
SL = np.array([0.2, 0.8, 0.3, 0.99])
SL_target = 0.95
SL_MSE = [-(i-SL_target)**2 for i in SL]

# compute softmax probabilities
probs = softmax(SL_MSE)

# sample from the resulting probability distribution
sample_ind = np.random.choice(len(probs), p=probs)
print(f'Sample = {SL[sample_ind]}')

print(f"Input vector: {SL}")
print(f"Softmax probabilities: {probs}")
print(f"Sampled index: {SL[sample_ind]}")
