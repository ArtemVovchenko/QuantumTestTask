import numpy as np

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

import tensorflow as tf
import keras.backend as K

MODEL_NAME = 'unet_model.h5'


def create_unet_model(image_height, image_width, image_channels) -> Model:
    inp = Input((image_height, image_width, image_channels))

    activ = Lambda(lambda x: x / 255)(inp)

    conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(activ)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv5)

    upper6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    upper6 = concatenate([upper6, conv4])
    conv6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(upper6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv6)

    upper7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    upper7 = concatenate([upper7, conv3])
    conv7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(upper7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv7)

    upper8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    upper8 = concatenate([upper8, conv2])
    conv8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(upper8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv8)

    upper9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    upper9 = concatenate([upper9, conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(upper9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv9)

    out = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=[inp], outputs=[out])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.metrics.MeanIoU(2), dice_coef])

    model.summary()
    return model


@tf.function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def make_prediction(model: Model, images: np.array):
    raw_predicted_data = model.predict(images)
    return (raw_predicted_data > 0.5).astype(np.uint8)


def load_unet_model():
    return load_model(MODEL_NAME, custom_objects={'mean_io_u': tf.metrics.MeanIoU(2), 'dice_coef': dice_coef})


def save_unet_model(model: Model):
    model.save('./' + MODEL_NAME)
