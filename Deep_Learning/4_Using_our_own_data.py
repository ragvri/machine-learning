"""
Applying tf on pos and neg sentiments data. The data is in the form of string. Also every string has diff length.
We want the same length input for tf

These are all the unique words in our input data:
['chair' , 'table' , 'spoon' , 'tv']
New sentence:
I pulled the chair upto the table.

[0 , 0, 0, 0]
chair is in the sentence, table is also there
[1,0,0,1]
"""

import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  # running, ran, run are same thing
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


def create_lexicon(pos, neg): # creates a list of words that are important
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    word_counts = Counter(lexicon)  # gives dictionary
    # word_counts = {'the':32322,'a':32134}

    l2 = []
    for w in word_counts:
        if 1000 > word_counts[w] > 50:  # the, an , a not necessory
            l2.append(w)

    print(len(l2))
    return l2


def sample_handling(sample, lexicon, classification): # creates a list of lists where the first element of list
    # denotes if word of lexicon present in our sample and the second tells us if it is pos or neg sampple
    featureset = []
    # [
    #     [ [0 1 0 0 1],[0,1]]
    # ]
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)
    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])  # [:,0] feature of numpy gets the first element
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
