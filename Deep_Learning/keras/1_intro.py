""" Here we will be looking at the sequential model in keras. The Sequential model is a linear stack of layers"""

from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import numpy as np

model = Sequential()
# add new layers using .add

# Dense implements operation : activation(dot(input,weights)+bias)
model.add(Dense(32, input_dim=100, activation='relu'))  # output array is of the shape(*,32)
model.add(Dense(10, activation='softmax'))  # output is of the shape (*,10), now we don't need to specify input anymore

"""The model needs to know what input shape it should expect. For this reason, the first layer in a  Sequential model 
needs to receive information about its input shape. 1) pass input_shape to first layer: It should be a tuple: None 
indicates any positive integer may be expected. In input_shape, the batch dimension is not included. 

2) Some 2D layers, such as Dense, support the specification of their input shape via the argument  input_dim,

3) If you ever need to specify a fixed batch size for your inputs (this is useful for stateful recurrent networks), 
you can pass a batch_size argument to a layer. If you pass both batch_size=32 and  input_shape=(6, 8) to a layer, 
it will then expect every batch of inputs to have the batch shape  (32, 6, 8) """

# Before training the model it needs to be compiled

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Now we train the model
# Keras models are trained on Numpy arrays of input data and labels

data = np.random.random((1000, 100))  # 1000 rows and 100 cols
labels = np.random.randint(10, size=(1000, 1))  # output can be of 10 classes so random number between 0 to 10 and
# since 1000 inputs so 1000 outputs

# now we need to convert the labels to one hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating through the data in batch size of 32
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
