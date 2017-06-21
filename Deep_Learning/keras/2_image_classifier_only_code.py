"""dogs vs cats
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Activation, Dropout, Flatten
from keras import backend as K

if K.image_data_format() == 'channels_first':
    input_shape = (3, 150, 150)
else:
    input_shape = (150, 150, 3) # 3 because RGB

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 16

train_datagen = ImageDataGenerator(rescale=1 / 255, zoom_range=0.2, horizontal_flip=True, shear_range=0.2)
test_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(directory="/home/raghav/Desktop/data_image/train",
                                                    target_size=(150, 150), batch_size=batch_size, class_mode='binary')

validation_generator = train_datagen.flow_from_directory(directory="/home/raghav/Desktop/data_image/validate",
                                                         target_size=(150, 150), batch_size=batch_size,
                                                         class_mode='binary')

model.fit_generator(train_generator, epochs=50, validation_data=validation_generator,
                    steps_per_epoch=2000 // batch_size,
                    validation_steps=800 // batch_size)
model.save_weights('first_try.h5')
