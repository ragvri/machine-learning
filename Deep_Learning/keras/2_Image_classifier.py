"""dogs vs cats
"""

from keras.preprocessing.image import ImageDataGenerator  # our images are augumented over random transformations so
# that our model never sees the same pic twice. This prevents overfitting
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Activation, Dropout, Flatten
from keras import backend as K
from PIL import Image
import numpy as np


def check_data_augument():
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    """rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures
     
    width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate 
    pictures
     
    vertically or horizontally rescale is a value by which we will multiply the data before any other processing. Our 
    original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process 
    (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor. 
    
    shear_range is for randomly applying shearing transformations 
    
    zoom_range is for randomly zooming inside pictures 
    
    
    horizontal_flip is for randomly flipping half of the images horizontally --relevant when there are no assumptions of 
    
    horizontal assymetry (e.g. real-world pictures). fill_mode is the strategy used for filling in newly created pixels, 
    which can appear after a rotation or a width/height shift. """

    img = load_img('/home/raghav/Documents/kaggle/dogscats/train/cat.0.jpg')
    x = img_to_array(img)  # this is a numpy array with shape (3, 150,150)
    x = x.reshape((1,) + x.shape)  # reshapes to (1,2,150,150)

    # .flow generates batches of randomly transformed images and saves the result to preview/ directory
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='/home/raghav/Desktop', save_prefix='cat',
                              save_format='jpeg'):
        i += 1
        if i > 20:
            break


# for small data number one concern is overfitting. Overfitting happens when a model exposed to too few examples
# learns patterns that do not generalize to new data, i.e. when the model starts using irrelevant features for making
#  prediction

# data augumentation helps but the images generated are highly corelated.
# another way is entropic capacity of the model ie how many features is model allowed to store.
# methods to modulate entropic capacity: number of parameters eg no of layers, nodes
# Another way is weight regularisation ie ensuring small weights.
# dropout: prevents a layer from seeing the same pattern twice

if K.image_data_format() == 'channels_first':
    input_shape = (3, 150, 150)
else:
    input_shape = (150, 150, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))  # no of filters. Input is (batch size, channels,
# rows, cols)
#  Output is 4d tensor (batch size, filter, new rows, new cols)
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))  # pool size: tuple of 2 integers to downscale. (2,2) halfs the row,
# col. Output is 4d tensor (batch size, channels, rows, cols)

model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # Earlier there were 64 filters each being a 2d matrix. flattens our 3d feature maps to 1d
# feature maps. Now only 64*row*cols 1d inputs
model.add(Dense(64))  # 64 outputs
model.add(Activation('relu'))  # f(x) = max(0,x), it can range from [0,inf] So used in hidden layers.
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(
    Activation('sigmoid'))  # for 2 class classification, sigmoid is used. For multiclass, we use softmax. They are
# applied only in the final layer as they give the probability of occurence of different classes

# since binary classifier
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# preparing our data
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1 / 255, zoom_range=0.2, horizontal_flip=True, shear_range=0.2)
test_datagen = ImageDataGenerator(rescale=1 / 255)

# this is a generator that will read images from training portion
train_generator = train_datagen.flow_from_directory(directory="/home/raghav/Desktop/data_image/train",
                                                    target_size=(150, 150), batch_size=batch_size, class_mode='binary')
# since we are using binary_crossentropy loss, we need binary labels

# validation generator
validation_generator = train_datagen.flow_from_directory(directory="/home/raghav/Desktop/data_image/validate",
                                                         target_size=(150, 150), batch_size=batch_size,
                                                         class_mode='binary')
# all images are resized to (150,150)

"""model.fit_generator(train_generator, epochs=50, validation_data=validation_generator,
                    steps_per_epoch=2000 // batch_size,
                    validation_steps=800 // batch_size)
model.save_weights('first_try.h5')
"""
model.load_weights('first_try.h5')
img = load_img("/home/raghav/Desktop/data_image/test/dog/dog.3000.jpg", target_size=(150, 150))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict_classes(x)
prob = model.predict_proba(x)
print(preds, prob)
if preds:
    print("Dog")
else:
    print("cat")
