""" Image Augumentation is the process of taking images of the training set and creating altered versions of the same
    image to deal with less training data being available and prevent overfitting.

    we will be using cifar10 dataset available with dataset"""

# the first thing is to load the cifar10 dataset and format the images to prepare them for CNN. We will also take a look
# at some images to see if it worked

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras import backend as K
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# input image dimensions
img_row, img_cols = 32, 32

# the data shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data() # y_train, y_test are uint8 labels from 0 to 9. 3 is for
# cats and 5 for dogs

# only look at cats[=3] and dogs[=5]
train_picks = np.ravel(np.logical_or(y_train == 3, y_train == 5))  # np.ravel flattens an array
test_picks = np.ravel(np.logical_or(y_train == 3, y_train == 5))

y_train = np.array(y_train[train_picks] ==5 , dtype= int)
y_test = np.array(y_test[test_picks]== 5, dtype=int)