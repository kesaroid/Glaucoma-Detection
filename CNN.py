from os import path, environ

from imgaug import augmenters as iaa
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 256, 256

channels = 3

input_shape = channels, img_width, img_height if K.image_data_format() == 'channels_first' \
    else img_width, img_height, channels

train_data_dir = path.join('data', 'train')
validation_data_dir = path.join('data', 'validation')
nb_train_samples = int(environ.get('TRAINING_SAMPLES', 20))
nb_validation_samples = int(environ.get('VALIDATION_SAMPLES', 20))
batch_size = 16
epochs = 100

input_tensor = Input(shape=input_shape)

block1 = BatchNormalization(name='norm_0')(input_tensor)

# Block 1
block1 = Conv2D(8, (3, 3), name='conv_11', activation='relu')(block1)
block1 = Conv2D(16, (3, 3), name='conv_12', activation='relu')(block1)
block1 = Conv2D(32, (3, 3), name='conv_13', activation='relu')(block1)
block1 = Conv2D(64, (3, 3), name='conv_14', activation='relu')(block1)
block1 = MaxPooling2D(pool_size=(2, 2))(block1)
block1 = BatchNormalization(name='norm_1')(block1)

block1 = Conv2D(16, 1)(block1)

# Block 2
block2 = Conv2D(32, (3, 3), name='conv_21', activation='relu')(block1)
block2 = Conv2D(64, (3, 3), name='conv_22', activation='relu')(block2)
block2 = Conv2D(64, (3, 3), name='conv_23', activation='relu')(block2)
block2 = Conv2D(128, (3, 3), name='conv_24', activation='relu')(block2)
block2 = MaxPooling2D(pool_size=(2, 2))(block2)
block2 = BatchNormalization(name='norm_2')(block2)

block2 = Conv2D(64, 1)(block2)

# Block 3
block3 = Conv2D(64, (3, 3), name='conv_31', activation='relu')(block2)
block3 = Conv2D(128, (3, 3), name='conv_32', activation='relu')(block3)
block3 = Conv2D(128, (3, 3), name='conv_33', activation='relu')(block3)
block3 = Conv2D(64, (3, 3), name='conv_34', activation='relu')(block3)
block3 = MaxPooling2D(pool_size=(2, 2))(block3)
block3 = BatchNormalization(name='norm_3')(block3)

# Block 4
block4 = Conv2D(64, (3, 3), name='conv_41', activation='relu')(block3)
block4 = Conv2D(32, (3, 3), name='conv_42', activation='relu')(block4)
block4 = Conv2D(16, (3, 3), name='conv_43', activation='relu')(block4)
block4 = Conv2D(8, (2, 2), name='conv_44', activation='relu')(block4)
block4 = MaxPooling2D(pool_size=(2, 2))(block4)
block4 = BatchNormalization(name='norm_4')(block4)

block4 = Conv2D(2, 1)(block4)

block5 = GlobalAveragePooling2D()(block4)
output = Activation('softmax')(block5)

model = Model(inputs=[input_tensor], outputs=[output])
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

# Initiate the train and test generators with data Augmentation
sometimes = lambda aug: iaa.Sometimes(0.6, aug)
seq = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Sharpen(alpha=1, lightness=0),
    iaa.CoarseDropout(p=0.1, size_percent=0.15),
    sometimes(iaa.Affine(
        scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
        translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
        rotate=(-30, 30),
        shear=(-16, 16)))
])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=seq.augment_image,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode='categorical')

checkpoint = ModelCheckpoint('f1.h5', monitor='acc', verbose=1, save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=0, mode='auto', cooldown=0, min_lr=0)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[checkpoint, reduce_lr]
)
