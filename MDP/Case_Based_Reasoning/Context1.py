import pandas as pd
import numpy as np

# Define a function to calculate the similarity between two hotels


def similarity(hotel1, hotel2):
    # Use Euclidean distance for numerical features
    numerical_features = ['Price', 'Distance', 'Rating']
    numerical_similarities = [np.exp(-0.5 * ((hotel1[feature] - hotel2[feature]) / hotel1[feature])**2) for feature in numerical_features]

    # Use Hamming distance for categorical features
    categorical_features = ['City', 'Amenities']
    categorical_similarities = [int(hotel1[feature] == hotel2[feature]) for feature in categorical_features]

    return np.mean(numerical_similarities + categorical_similarities)

# Define a function to retrieve the k most similar hotels to a given query hotel from a dataset


def retrieve_cases(dataset, query, k):
    similarities = []
    for _, hotel in dataset.iterrows():
        similarities.append(similarity(hotel, query))
    most_similar_indices = np.argsort(similarities)[::-1][:k]
    return dataset.iloc[most_similar_indices], similarities

# Define a function to make a decision based on the retrieved cases and the query


def make_decision(retrieved_cases, query):
    recommended = retrieved_cases['Recommended'].value_counts()
    if 'Yes' in recommended and ('No' not in recommended or recommended['Yes'] > recommended['No']):
        return "Recommended"
    else:
        return "Not recommended"


# Example usage:
data = {
    'Name': ['The Ritz-Carlton', 'Hilton Garden Inn', 'Marriott', 'Sheraton', 'Holiday Inn'],
    'City': ['New York', 'Chicago', 'Miami', 'Los Angeles', 'Las Vegas'],
    'Price': [500, 200, 300, 400, 100],
    'Distance': [1.5, 2.0, 1.0, 3.0, 2.5],
    'Rating': [9.0, 8.5, 9.5, 8.0, 7.5],
    'Amenities': ['Spa, Pool, Gym', 'Pool', 'Spa, Gym', 'Gym', 'Pool, Gym'],
    'Recommended': ['Yes', 'Yes', 'No', 'Yes', 'No']
}

hotels = pd.DataFrame(data)
query = hotels.iloc[0]  # Let's say we want to recommend hotels similar to "The Ritz-Carlton"
retrieved_cases, local_similarities = retrieve_cases(hotels, query, 3)
decision = make_decision(retrieved_cases, query)

print("Recommended hotels based on your preferences:")
print(retrieved_cases[['Name', 'City', 'Price', 'Distance', 'Rating', 'Amenities']])
print("Decision:", decision)

# Print a table of local similarities
print("Local similarities:")
for i, row in retrieved_cases.iterrows():
    print(f"{row['Name']}: {local_similarities[i]:.2f}")
