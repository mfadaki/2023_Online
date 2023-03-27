### Issue 1
Assume the 
- State size = S
- Action Size = A
- Hidden Layers = H
- batch_size = B

Then, The DGN NN would be like this:

DQN(
  (fc1): Linear(in_features=S, out_features=H, bias=True)
  (fc2): Linear(in_features=H, out_features=H, bias=True)
  (fc3): Linear(in_features=H, out_features=A, bias=True)
)

Under the 'train' function of Agent:
- if we run: states, actions, rewards, next_states, dones = agent.buffer.sample(agent.B)
- then, the shapes for all "states, actions, rewards, next_states, dones" would be (e.g., shape.shape):
  - torch.Size([B]) for any selected B
  - This does not work for the DQN model, running the "main_network(states)" or "target_network(next_states)" in other part of the code will generate error of "mat1 and mat2 shapes cannot be multiplied (?x? and ?x?)"

To rectify this issue, we need to update the shape of "states, actions, rewards, next_states, dones"

```
states = states.unsqueeze(-1)
actions = actions.unsqueeze(1)
rewards = rewards.unsqueeze(1)
dones = dones.unsqueeze(1)
next_states = states.unsqueeze(-1)
```

- Note that the above method is not right for all problem. The final shapes should be like:
- states: torch.Size([B, S])
- actions: torch.Size([B, A])
- rewards: torch.Size([B])
- next_states: torch.Size([B, S])
- dones: torch.Size([B])
- q_values: torch.Size([B])
- target: torch.Size([B])

### Issue 2:
It is related to the gather() function in this line:
```
q_values = self.main_network(states).gather(1, actions)
```
- First, actions should be int64 for this function to work:
```
actions = torch.tensor([e[1] for e in experiences], dtype=torch.int64)
```
- Second, actions in this gather() function are the indexes of the tensor (self.main_network(states)) which should be selected. Therefore, indexes cannot be negative. In case of Shower example, if we select actions as 'temp_down=-1, temp_stay=0, temp_up=1', it does not work as action=-1 cannot be an index of the tensor which is the result of (self.main_network(states)).
- Therefore, the actions should be defined as " 'temp_down=0, temp_stay=1, temp_up=2'.


### Issue 3:
- Converting the States to one-hot tensor makes the index of the first 1 as 0. When we read this in env, we need to be careful that the original value of state becomes zero. In the tram problem, the first state should be 1 not zero. 