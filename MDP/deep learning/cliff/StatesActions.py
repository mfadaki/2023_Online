import torch
# Create one_hot tensor for Actions
# feasible_actions = [-1, 0, 1]
feasible_actions = range(0, 3)
actions_one_hot_tensor = torch.eye(len(feasible_actions))
action_size = len(feasible_actions)

# Create one_hot tensor for States
height = 4
width = 12
feasible_states = [[i, j] for i in range(height) for j in range(width)]

states_one_hot_tensor = torch.eye(len(feasible_states))
state_size = len(feasible_states)


def StateOht2Org(state_oht):
    return feasible_states[torch.where(state_oht == 1)[0].item()]


def StateOrg2Oht(state_org):
    return states_one_hot_tensor[feasible_states.index(state_org)]


def ActionOrg2Oht(action_org):
    return actions_one_hot_tensor[feasible_actions.index(action_org)]


def ActionOht2Org(action_oht):
    return feasible_actions[torch.where(action_oht == 1)[0].item()]
