import sys
import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_uint
from model import make_prediction, load_unet_model

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


def load_and_prdict_test_data():
    unet_model = load_unet_model()
    ids = get_images_ids()
    x_test1 = np.zeros((len(ids['test1_ids']), IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)
    x_test1_paths = np.zeros((len(ids['test1_ids'])), dtype=object)
    x_test2 = np.zeros((len(ids['test2_ids']), IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)
    x_test2_paths = np.zeros((len(ids['test2_ids'])), dtype=object)
    print('Getting test images and predicting masks, part 1:')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(ids['test1_ids']), total=len(ids['test1_ids'])):
        path = TEST1_DATA_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        x_test1[n] = img
        x_test1_paths[n] = path + '/images/' + 'predicted_mask_' + id_ + '.jpg'
    x_test1_masks = make_prediction(unet_model, x_test1)
    print('Saving masks')
    sys.stdout.flush()
    for n, mask in tqdm(enumerate(x_test1_masks)):
        mask = (mask > 0).astype(np.float32)
        imsave(x_test1_paths[n], img_as_uint(mask))
    print('Getting and resizing test images, part 2:')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(ids['test2_ids']), total=len(ids['test2_ids'])):
        path = TEST2_DATA_PATH + id_
        try:
            img = imread(path + '/images/' + id_ + '.png')[:, :, :CHANNELS]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            x_test2[n] = img
            x_test2_paths[n] = path + '/images/' + 'predicted_mask_' + id_ + '.jpg'
        except IndexError:
            print("Unable to load image: " + path + '/images/' + id_ + '.png')

    x_test2_masks = make_prediction(unet_model, x_test2)
    for n, mask in tqdm(enumerate(x_test2_masks)):
        if isinstance(x_test2_paths[n], str):
            mask = (mask > 0).astype(np.float32)
            imsave(x_test2_paths[n], img_as_uint(mask))

    print("Test data processed")


load_and_prdict_test_data()
