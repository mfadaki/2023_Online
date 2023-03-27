import pandas as pd
import numpy as np

# define a sample dataset with mixed feature types
data = pd.DataFrame({
    'age': [20, 35, 40, 25, 30],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'income': [50000, 70000, 60000, 80000, 65000],
    'marital_status': ['S', 'M', 'S', 'S', 'M'],
    'purchased': [0, 1, 1, 0, 1]
})

# define a new case
new_case = {'age': 27, 'gender': 'M', 'income': 55000, 'marital_status': 'S'}

# define feature weights
feature_weights = {'age': 0.4, 'gender': 0.1, 'income': 0.3, 'marital_status': 0.2}

# print local similarity scores for the new case
local_similarity_scores = []
for feature in new_case:
    if feature in feature_weights:
        if isinstance(new_case[feature], str):
            # calculate similarity for discrete feature
            local_similarity = 1 if new_case[feature] == data[feature][0] else 0
        else:
            # calculate similarity for continuous feature
            data_range = max(data[feature]) - min(data[feature])
            local_similarity = 1 - (abs(new_case[feature] - data[feature][0]) / data_range)
        local_similarity_scores.append(local_similarity)
local_similarity_df = pd.DataFrame({'feature': new_case.keys(), 'local_similarity': local_similarity_scores})
print(local_similarity_df)

# make decision based on highest similarity score
decision_index = np.argmax(local_similarity_scores)
decision = data.iloc[decision_index]['purchased']
print(f"Decision: {decision}")
