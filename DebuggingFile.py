
import DataSetTool as ds
import os
import random
import threading
import tifffile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import AerialDataGenerator as adGenerator

from threading import Thread
from cv2 import cv2
from PIL import Image
from skimage.io import imread, imshow
from tqdm import tqdm
# Input images size
# original dimensions/16
IMG_WIDTH = 250
IMG_HEIGHT = 250
IMG_CHANNELS = 3
SAMPLE_SIZE = 20000
BATCH_SIZE = 2000
# current labels
labels = {
    0: (1, 1, 1),  # white, paved area/road
    1: (0, 1, 1),  # light blue, low vegetation
    2: (0, 0, 1),  # blue, buildings
    3: (0, 1, 0),  # green, high vegetation
    4: (1, 0, 0),  # red, bare earth
    5: (1, 1, 0)  # yellow, vehicle/car
}

labels_unormalized = {
    0: (255, 255, 255),  # white, paved area/road
    1: (0, 255, 255),  # light blue, low vegetation
    2: (0, 0, 255),  # blue, buildings
    3: (0, 255, 0),  # green, high vegetation
    4: (255, 0, 0),  # red, bare earth
    5: (255, 255, 0)  # yellow, vehicle/car
}

one_hot_labels = {
    0: [1, 0, 0, 0, 0, 0],  # white, paved area/road
    1: [0, 1, 0, 0, 0, 0],  # light blue, low vegetation
    2: [0, 0, 1, 0, 0, 0],  # blue, buildings
    3: [0, 0, 0, 1, 0, 0],  # green, high vegetation
    4: [0, 0, 0, 0, 1, 0],  # red, bare earth
    5: [0, 0, 0, 0, 0, 1]  # yellow, vehicle/car
}

N_OF_LABELS = len(labels)

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# flag to train the model or load pre-trained weights
to_train = 0

DST_PARENT_DIR = './Vaihingen/'
PARENT_DIR = './Vaihingen/'
ORIGINAL_PATH = 'Originals/'
SEGMENTED_PATH = 'SegmentedOriginals/'

DST_SEGMENTED_PATH = "Variation_1_Segmented/"
DST_ORIGINAL_PATH = "Variation_1_Originals/"

ORIGINAL_RESIZED_PATH = "Resized_Originals_Variation_1/"
SEGMENTED_RESIZED_PATH = "Resized_Segmented_Variation_1/"
SEGMENTED_ONE_HOT_PATH = "Resized_Segmented_One_Hot/"

RESULTS_PATH = "./Results/"
LABEL_TYPES_PATH = "results_on_"

# consistent randomness and good-luck charm
seed = 42
np.random.seed = seed
ds_tool = ds.DataSetTool(DST_PARENT_DIR, PARENT_DIR, ORIGINAL_PATH, SEGMENTED_PATH, DST_SEGMENTED_PATH,
                 DST_ORIGINAL_PATH, ORIGINAL_RESIZED_PATH, SEGMENTED_RESIZED_PATH, SEGMENTED_ONE_HOT_PATH,
                 RESULTS_PATH, LABEL_TYPES_PATH)


def one_hot_th_function(train_ids, labeled_images):
    ret_arr = []
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

        mask_ = labeled_images[n]
        encoded_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, N_OF_LABELS), dtype=np.uint8)

        # check each pixel of current mask
        for row_idx in range(0, mask_.shape[0]):
            for col_idx in range(0, mask_.shape[1]):
                for label_idx in range(0, N_OF_LABELS):
                    # if current pixel value has the current label value flag it in result
                    # it uses the fact that all return_array values are initially 0
                    if tuple(mask_[row_idx, col_idx]) == labels_unormalized[label_idx]:
                        encoded_img[row_idx][col_idx] = one_hot_labels[label_idx]
        ret_arr.append(encoded_img)

    return ret_arr


def test_one_hot_algo():
    masks_root_dir = PARENT_DIR + DST_SEGMENTED_PATH
    ids = os.listdir(masks_root_dir)

    test_imgs = []
    test_ids = []
    test_originals = []
    test_imgs.append(imread(DST_PARENT_DIR + DST_SEGMENTED_PATH + ids[2])[:, :, :IMG_CHANNELS])
    test_ids.append(ids[2])
    test_originals.append(imread(DST_PARENT_DIR + DST_ORIGINAL_PATH + ids[2])[:, :, :IMG_CHANNELS])

    test_encoded = one_hot_th_function(test_ids, test_imgs)

    imshow(test_originals[0])
    plt.show()

    imshow(test_imgs[0])
    plt.show()

    imshow(ds_tool.parse_prediction(test_encoded[0], labels_unormalized))
    plt.show()

def test_existing_one_hot_imgs():
    masks_root_dir = PARENT_DIR + SEGMENTED_ONE_HOT_PATH
    ids = os.listdir(masks_root_dir)

    test_imgs = []

    for id in ids[1500:1510]:
        img = imread(DST_PARENT_DIR + SEGMENTED_ONE_HOT_PATH + id)[:, :, :N_OF_LABELS]
        imshow(ds_tool.parse_prediction(img, labels_unormalized))
        plt.show()


# test_one_hot_algo()
test_existing_one_hot_imgs()

