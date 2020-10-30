import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.io import imread
from skimage.transform import resize
from model import create_unet_model, make_prediction, save_unet_model, load_unet_model, MODEL_NAME

import tensorflow as tf
import keras.backend as K


IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
TRAIN_DATA_PATH = './input/stage1_train/'
TEST1_DATA_PATH = './input/stage1_test/'
TEST2_DATA_PATH = './input/stage2_test_final/'


def get_images_ids() -> dict:
    train_data_ids = next(os.walk(TRAIN_DATA_PATH))[1]
    test1_data_ids = next(os.walk(TEST1_DATA_PATH))[1]
    test2_data_ids = next(os.walk(TEST2_DATA_PATH))[1]
    return {'train_ids': train_data_ids, 'test1_ids': test1_data_ids, 'test2_ids': test2_data_ids}


def load_train_data() -> tuple:
    ids = get_images_ids()
    x_train = np.zeros((len(ids['train_ids']), IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)
    y_train = np.zeros((len(ids['train_ids']), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    x_train_paths = np.zeros((len(ids['train_ids'])), dtype=object)
    print('Getting train images and masks and resizing them')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(ids['train_ids']), total=len(ids['train_ids'])):
        path = TRAIN_DATA_PATH + id_
        image = imread(path + '/images/' + id_ + '.png')[:, :, :CHANNELS]
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True, mode='constant')
        x_train[n] = image
        mask = np.zeros((IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize
                                   (mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True),
                                   axis=-1)
            mask = np.maximum(mask, mask_)
        y_train[n] = mask
        x_train_paths[n] = path
    print("Train data uploaded")
    return x_train, y_train, x_train_paths


@tf.function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def train(model: Model, x_train_data, y_train_data):
    early_stopper = EarlyStopping(patience=5, verbose=1)
    check_pointer = ModelCheckpoint(MODEL_NAME, verbose=1, save_best_only=True)
    model.fit(x_train_data, y_train_data, validation_split=0.1, batch_size=16, epochs=30, callbacks=[early_stopper, check_pointer])
    return model


def calculate_images_dice_coef(true_masks, predicted_masks):
    coeffs = np.zeros(shape=(len(true_masks)), dtype=tf.Tensor)
    for n in range(0, len(true_masks)):
        coeffs[n] = (2. * np.sum(np.logical_and(true_masks[n], predicted_masks[n])) / (np.sum(true_masks[n]) + np.sum(predicted_masks[n])))
    return coeffs


def save_dice_coefs(paths, dice_coefs):
    dataframe = pd.DataFrame(columns=['image_id', 'dice_score'])
    for i in range(len(paths)):
        dataframe.loc[i, :] = [paths[i], dice_coefs[i]]
    dataframe.to_csv('./training_dice_scores.csv')


def create_and_train_model():
    x_train, y_train, paths = load_train_data()
    unet_model = create_unet_model(IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    unet_model = train(unet_model, x_train_data=x_train, y_train_data=y_train)
    save_unet_model(unet_model)
    unet_model = load_unet_model()
    predicted_masks = make_prediction(unet_model, x_train)
    save_dice_coefs(paths, calculate_images_dice_coef(y_train, predicted_masks))


create_and_train_model()



