"""
Eucledian Distance: (sum i=1 to n where n is number of dimensions ( Qi -Pi )^2 )^(1/2)
Compare for 2-d points, put n=2
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}  # 2 classes k and r
new_features = [5, 7]

[[plt.scatter(j[0], j[1], s=100, color=i) for j in dataset[i]] for i in dataset]

plt.show()


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to value less than total groups')
    distances = []
    for group in data:
        for features in data[group]:
            # eucledian_distance  = np.sqrt(np.sum(((np.array(features)-np.array(predict))**2))
            eucledian_distance = np.linalg.norm(
                np.array(features) - np.array(predict))  # calculates the eucledian distance
            distances.append([eucledian_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)
