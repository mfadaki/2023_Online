import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import random

# Method 1 old


def encode_action(a, b, a_min, a_max, b_min, b_max):
    # Encoding (a,b)
    num_a = a_max - a_min + 1
    num_b = b_max - b_min + 1
    index = ((a - a_min) * num_b) + (b - b_min)
    one_hot_vec = [0] * (num_a * num_b)
    one_hot_vec[index] = 1
    return one_hot_vec


a, b = 17, 21
a_min, a_max = 10, 30
b_min, b_max = 10, 30

one_hot_vec = encode_action(a, b, a_min, a_max, b_min, b_max)
print(one_hot_vec)


def decode_action(one_hot_vec, a_min, a_max, b_min, b_max):
    num_b = b_max - b_min + 1
    vec = np.array(one_hot_vec)
    index = vec.argmax()
    a = (index // num_b) + a_min
    b = (index % num_b) + b_min
    return a, b


one_hot_vec = one_hot_vec  # one-hot encoded vector for some (a, b) tuple
a_min, a_max = 10, 30
b_min, b_max = 10, 30

a, b = decode_action(one_hot_vec, a_min, a_max, b_min, b_max)
print("a:", a)
print("b:", b)


# Method 2

min_value = 10
max_value = 20

num_actions = (max_value - min_value + 1) ** 2

one_hot_tensor = torch.eye(num_actions)


# For actions in the format of (a,b) if both are within 10 and 20
min_value = 10
max_value = 20

num_actions = (max_value - min_value + 1) ** 2

one_hot_tensor = torch.eye(num_actions)

print(one_hot_tensor)

# To randomly select an action

min_value = 10
max_value = 20

num_actions = (max_value - min_value + 1) ** 2

one_hot_tensor = torch.eye(num_actions)

# randomly select an action
action_index = random.randint(0, num_actions - 1)
action = (action_index // (max_value - min_value + 1) + min_value,
          action_index % (max_value - min_value + 1) + min_value)

print("Randomly selected action:", action)
print("One-hot encoding:", one_hot_tensor[action_index])


# To get the vector of tensor for (a,b)
# example action
a = 15
b = 12

# calculate index of action in tensor
action_index = (a - min_value) * (max_value - min_value + 1) + (b - min_value)

# get one-hot encoding vector for action
action_vector = one_hot_tensor[action_index]

print("Action:", (a, b))
print("One-hot encoding vector:", action_vector)

# To get back the (a,b) from tensor


min_value = 10
max_value = 20

num_actions = (max_value - min_value + 1) ** 2

one_hot_tensor = torch.eye(num_actions)

# example action
a = 15
b = 12

# calculate index of action in tensor
action_index = (a - min_value) * (max_value - min_value + 1) + (b - min_value)

# get one-hot encoding vector for action
action_vector = one_hot_tensor[action_index]

# find index of non-zero element in tensor
action_index_back = torch.nonzero(action_vector).item()

# calculate corresponding values of a and b
a_back = action_index_back // (max_value - min_value + 1) + min_value
b_back = action_index_back % (max_value - min_value + 1) + min_value

print("Original action:", (a, b))
print("Recovered action:", (a_back, b_back))


# Method 3
# If we just keep feasible combinations of (a,b). For example, when adding a+b ot does not exceed the S


# define the input list
input_list = [[0, 0],
              [0, 1],
              [0, 2],
              [0, 3]]

# create an instance of the OneHotEncoder class
encoder = OneHotEncoder()

# fit the encoder to the input list and transform it to one-hot encoded format
one_hot_encoded = encoder.fit_transform(input_list)

# define the vector to be converted back to original value
vector = np.array([0., 0., 1., 0.]).reshape(1, -1)  # reshape the input vector to (1,4)

# use inverse_transform to retrieve the original value
original_value = encoder.inverse_transform(vector)

# print the original value
print(original_value[0])

######
enc = OneHotEncoder(handle_unknown='ignore')
X = [[2, 1], [5, 3], [4, 2]]
enc.fit(X)
enc.n_values_
enc.transform([0, 1, 1]).toarray()
bb = encoder.fit_transform(X).toarray()
enc.transform(X).toarray()

#####


# define the input list
input_list = [[0, 0],
              [0, 1],
              [0, 2],
              [0, 3]]

# create an instance of the OneHotEncoder class with 'if_binary' drop parameter
encoder = OneHotEncoder(drop='if_binary')

# fit the encoder to the input list and transform it to one-hot encoded format
one_hot_encoded = encoder.fit_transform(input_list)

# define the vector to be converted back to original value
vector = np.array([0., 0., 1., 0.]).reshape(1, -1)  # reshape the input vector to (1,4)

# use inverse_transform to retrieve the original value
original_value = encoder.inverse_transform(vector)

# print the original value
print(original_value[0])

# *******
target = np.array(['dog', 'dog', 'cat', 'cat', 'cat', 'dog', 'dog', 'cat', 'cat'])
target = [[0, 0],
          [0, 1],
          [0, 2],
          [0, 3]]
ohe = OneHotEncoder(categories='auto', sparse=False)
arr = ohe.fit_transform(target[:, np.newaxis])
arr

#####

Sure! Here's an example of a two-dimensional NumPy array that can be transformed into one-hot encoded using scikit-learn:

python
Copy code

# Define a sample input array

# Define a sample input array

# define input array
input_array = np.array([[2, 2, 1], [1, 2, 1], [0, 1, 2]])

# create one-hot encoder
encoder = OneHotEncoder(categories='auto', sparse=False)

# fit and transform the input array
one_hot_encoded = encoder.fit_transform(input_array)

# ensure that each row has only a single instance of 1
one_hot_encoded[np.arange(len(one_hot_encoded)), one_hot_encoded.argmax(1)] = 1
one_hot_encoded[one_hot_encoded != 1] = 0

# print the resulting arrays
print(one_hot_encoded)
