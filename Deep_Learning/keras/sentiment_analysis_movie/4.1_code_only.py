
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# load the dataset
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words_in_a_review = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words_in_a_review)
X_test = sequence.pad_sequences(X_test, maxlen=max_words_in_a_review)
model = Sequential()
model.add(Embedding(input_dim=top_words, output_dim=32, input_length=max_words_in_a_review))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("\n accuracy %s" % scores[1]*100)

