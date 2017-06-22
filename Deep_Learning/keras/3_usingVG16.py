"""
Using the concept of transfer learning to improve accuracy. VGG16 is a CNN that has been trained on ImageNet data.
We first load this model upto the first fully connected layer.
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import applications
import gc

img_width, img_ht = 150, 150
top_model_wt_path = "bottleneck_fc_model.h5"
train_dir = "/home/raghav/Desktop/data_image/train"
validation_dir = "/home/raghav/Desktop/data_image/validate"
test_dir = "/home/raghav/Desktop/data_image/test"
no_train_samples = 2000
no_validation_samples = 800
epochs = 50
batch_size = 16


def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1 / 255)

    # build the vgg16 model
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(train_dir, target_size=(img_width, img_ht), shuffle=False, class_mode=None,
                                            batch_size=batch_size)  # class_mode=None means our data will only yield
    # batches of data, no labels, shuffle=False means our data will be in order so first 1000 images will be cats and
    #  next 1000 dogs

    # generates predication for a generator. Steps: total no of batches. Returns a numpy array of predictions
    bottleneck_features_train = model.predict_generator(generator=generator, steps=no_train_samples // batch_size)
    # saves an array to a binary file
    np.save(file="bottleneck_features_train.npy", arr=bottleneck_features_train)

    generator = datagen.flow_from_directory(validation_dir, target_size=(img_width, img_ht), batch_size=batch_size,
                                            class_mode=None, shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, no_validation_samples // batch_size)
    np.save(file="bottleneck_features_validate.npy", arr=bottleneck_features_validation)


def train_top_model():
    train_data = np.load(file="bottleneck_features_train.npy")
    train_labels = np.array([0] * (no_train_samples // 2) + [1] * (no_train_samples // 2))

    validation_data = np.load(file="bottleneck_features_validate.npy")
    validation_labels = np.array([0] * (no_validation_samples // 2) + [1] * (no_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))  # don't need to tell batch size in input shape
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    # this gives training data accuracy("acc") and validation data accuracy ("val_acc"). If the "acc" keeps on improving
    # while the "val_acc" keeps on decreasing, then we are likely overfitting the model
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_wt_path)


def predict_image_class(file):
    model = applications.VGG16(include_top=False, weights='imagenet')
    x = load_img(file, target_size=(img_width, img_ht))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x, verbose=0)  # verbose = 0 means no logging
    model = Sequential()
    model.add(Flatten(input_shape=array.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights(top_model_wt_path)
    class_predicted = model.predict_classes(array, verbose=0)
    probability = model.predict(array, verbose=0)[0][0]
    if class_predicted == 1 and probability > 0.5:
        print("dogs")
    elif class_predicted == 0 and probability > 0.5:
        print("cat")
    else:
        print("None")


"""
save_bottleneck_features()
train_top_model()
"""

# predict_image_class(test_dir + "/cat/cat.3120.jpg")
predict_image_class("/home/raghav/Pictures/1.png")
gc.collect()  # resolves an error of session of tensorflow
