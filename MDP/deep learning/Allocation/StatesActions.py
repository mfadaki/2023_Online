import torch


# Model Parameters
model_param = {
    'prp1': 100,
    'penalty1': 200,
    'penalty2': 200,
    'target_fr1': 0.9,
    'target_fr2': 0.9,
    'hc': 10,
    'sc': 0.5,
    's': 10,
    'p': 100,
    'prob_disrupt': 0.1,
    'disr_ratio': 0.1
}

# Create one_hot tensor for Actions
# feasible_actions = [-1, 0, 1]
feasible_actions = range(-10, 11)
actions_one_hot_tensor = torch.eye(len(feasible_actions))
action_size = len(feasible_actions)

# Create one_hot tensor for States

feasible_states = range(0, 21)
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
