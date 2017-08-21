""" Using the IMDB dataset of movie reviews. The Large Movie Review Dataset (often referred to as the IMDB dataset)
contains 25,000 highly polar moving reviews (good or bad) for training and the same amount again for testing. The
problem is to determine whether a given moving review has a positive or negative sentiment.
"""

from keras.datasets import imdb  # keras provides access to the imdb dataset built-in
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# load the dataset
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)  # loading only the top 5000 words
# imdb.load_data(): the words have been replaced by integers
# which represent the absolute popularity of a word in the dataset. so that for instance the integer "3" encodes the
# 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,
# 000 most common words, but eliminate the top 20 most common words". The sentences in each review thus comprises of
# a sequence of integers

max_words_in_a_review = 500

# sequence.pad_sequences() creates a list where each review is of length = max_words_in_review. If length of actual
# review greater than 500, it is truncated, else 0s are padded in the beginning
X_train = sequence.pad_sequences(X_train, maxlen=max_words_in_a_review)
X_test = sequence.pad_sequences(X_test, maxlen=max_words_in_a_review)

# now we will create our model. We will first use Embedding layer setting the vocabulary to be 5000 ,  the output
# vector size is 32 and input length is 500. The output is a 2d matrix of 500*32 size. Next we will Flatten this and
# add a dense layer of 250 outputs and then another dense layer of 1 output unit

# now we do word embeddings: This is a technique where words are encoded as real-valued vectors in a
# high-dimensional space, where the similarity between words in terms of meaning translates to closeness in the
# vector space

# in keras we can turn positive integers into dense vectors of fixed size using embedding
# keras.layers.embeddings.Embedding()

# input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
# output_dim: int >= 0. Dimension of the dense embedding. embeddings_initializer: Initializer for the embeddings
#                        matrix (see initializers).
# embeddings_regularizer: Regularizer function applied to the embeddings matrix (see
#                         regularizer).
# embeddings_constraint: Constraint function applied to the embeddings matrix (see constraints).
# mask_zero: Whether or not the input value 0 is a special "padding" value that should be masked out. This is useful
#            when using recurrent layers which may take variable length input. If this is True then all subsequent
#            layers in the  model need to support masking or an exception will be raised. If mask_zero is set to True,
#            as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).
# input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect
#                Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).


model = Sequential()
model.add(Embedding(input_dim=top_words, output_dim=32, input_length=max_words_in_a_review))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)  # only 1 log line
#  per epoch
scores = model.evaluate(X_test, y_test, verbose=0)
print("\n accuracy %s" % scores[1]*100)

