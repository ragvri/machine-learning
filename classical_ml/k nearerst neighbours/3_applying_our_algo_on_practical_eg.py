# applying the k nearest algo we built on the breast cancer classification problem
import numpy as np
import pandas as pd
import random
from collections import Counter
import warnings


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to value less than total groups')
    distances = []
    for group in data:
        for features in data[group]:
            eucledian_distance = np.linalg.norm(
                np.array(features) - np.array(predict))
            distances.append([eucledian_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence

df = pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?', -9999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

full_data = df.astype(float).values.tolist()  # converting the data to float and then getting a list

random.shuffle(full_data)  # shuffles the list

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print("Accuracy", correct / total)
accuracies.append(correct / total)
