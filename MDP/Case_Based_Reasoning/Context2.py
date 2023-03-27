import pandas as pd
import math

# Define the case library
case_library = [
    {"name": "Car", "wheels": 4, "fuel": "gasoline", "max_speed": 200, "weight": 1500},
    {"name": "Bike", "wheels": 2, "fuel": "none", "max_speed": 50, "weight": 200},
    {"name": "Truck", "wheels": 6, "fuel": "diesel", "max_speed": 150, "weight": 5000},
    {"name": "Motorcycle", "wheels": 2, "fuel": "gasoline", "max_speed": 250, "weight": 400}
]

# Define the new problem
new_vehicle = {"name": "Bus", "wheels": 4, "fuel": "diesel", "max_speed": 120, "weight": 3500}

# Define the weights for the features
weights = {"wheels": 1, "fuel": 3, "max_speed": 2, "weight": 4}

# Calculate the ranges for continuous features
ranges = {}
for key in new_vehicle.keys():
    if key != "name" and isinstance(new_vehicle[key], (int, float)):
        feature_values = [case[key] for case in case_library]
        ranges[key] = max(feature_values) - min(feature_values)

# Define a function to calculate the similarity between two values


def similarity(a, b, feature_weight, feature_range):
    if isinstance(a, str) and isinstance(b, str):
        if a == b:
            return feature_weight
        else:
            return 0
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        data_range = feature_range + 1e-9  # Avoid division by zero
        return feature_weight * (1 - abs(a - b) / data_range)
    else:
        return 0


# Calculate the similarity between the new problem and all past cases
similarity_scores = []
for case in case_library:
    similarity_score = 0
    for key in new_vehicle.keys():
        if key != "name":
            if isinstance(new_vehicle[key], (int, float)):
                similarity_score += similarity(case[key], new_vehicle[key], weights[key], ranges[key])
            else:
                similarity_score += similarity(case[key], new_vehicle[key], weights[key], None)
    similarity_scores.append(similarity_score)

# Create a dataframe of the similarity scores
df = pd.DataFrame({
    "Case": [case["name"] for case in case_library],
    "Similarity": similarity_scores
})

# Print the dataframe
print(df)
