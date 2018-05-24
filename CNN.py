from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers


img_width, img_height = 256, 256
input_shape = (img_width, img_height, 3)

train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = 160
nb_validation_samples = 20
batch_size = 5
epochs = 10

model = Sequential()

model.add(Conv2D(3, (11, 11), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(96, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(192, (3, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(192, (3, 3)))

model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()
model.compile(loss="mean_squared_error", optimizer=optimizers.SGD(
    lr=0.001, momentum=0.7, decay=0.0, nesterov=False), metrics=["accuracy"])


# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="binary")


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size

)

model.save('f1.h5')
