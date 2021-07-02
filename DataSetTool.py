import datetime
import json
import os
import random
import threading
import time

import keras
import tifffile
import numpy as np
import tensorflow as tf
import io

from matplotlib import pyplot as plt

import AerialDataGenerator as adGenerator

from threading import Thread
from cv2 import cv2
from PIL import Image
from skimage.io import imread, imshow
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report

# Input images size
# original dimensions/16
IMG_WIDTH = 1000
IMG_HEIGHT = 1000
IMG_CHANNELS = 3
SAMPLE_SIZE = 20000
BATCH_SIZE = 16
dict_stats_file = 'dict_stats.json'
confusion_matrix_file = 'confusion_matrix.json'
# current labels
labels = {
    0: (255, 255, 255),  # white, paved area/road
    1: (0, 255, 255),  # light blue, low vegetation
    2: (0, 0, 255),  # blue, buildings
    3: (0, 255, 0),  # green, high vegetation
    4: (255, 0, 0),  # red, bare earth
    5: (255, 255, 0)  # yellow, vehicle/car
}

labels_name = {
    0: 'road',  # white, paved area/road
    1: 'grass',  # light blue, low vegetation
    2: 'buildings',  # blue, buildings
    3: 'trees',  # green, high vegetation
    4: 'dirt_water',  # red, bare earth
    5: 'vehicle'  # yellow, vehicle/car
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
seed = np.random.seed(42)
random.seed(42)


# tried to make tf.input pipeline to wrok.. it didn't
def decode_png_img(img):
    img = tf.image.decode_png(img, channels=IMG_CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.uint8)

    return img


# tried to make tf.input pipeline to wrok.. it didn't
def decode_tif_img(img):
    # img = tf.image.decode_image(img, channels=N_OF_LABELS, dtype=tf.dtypes.uint8)
    img = one_hot_enc(img.numpy())
    img = tf.convert_to_tensor(img, tf.uint8)
    img = tf.image.convert_image_dtype(img, tf.uint8)

    return img


# tried to make tf.input pipeline to wrok.. it didn't
def one_hot_enc(input_img):
    encoded_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, N_OF_LABELS), dtype=np.uint8)
    # print(input_img)
    image = np.array(Image.open(io.BytesIO(input_img)))
    # print(image)
    for label_idx in range(0, N_OF_LABELS):
        for row_idx in range(0, image.shape[0]):
            for col_idx in range(0, image.shape[1]):
                # if current pixel value has the current label value flag it in result
                # it uses the fact that all return_array values are initially 0
                if tuple(image[row_idx, col_idx]) == labels[label_idx]:
                    encoded_img[row_idx][col_idx][label_idx] = 1  # or 255
    return encoded_img


# actually loads an image, its maks and returns the pair
# tried to make tf.input pipeline to wrok.. it didn't
def combine_img_masks(original_path: tf.Tensor, segmented_path: tf.Tensor):
    original_image = tf.io.read_file(original_path)
    original_image = decode_png_img(original_image)

    mask_image = tf.io.read_file(segmented_path)
    mask_image = decode_tif_img(mask_image)

    return original_image, mask_image


class DataSetTool:
    def __init__(self, DST_PARENT_DIR, PARENT_DIR, ORIGINAL_PATH, SEGMENTED_PATH, DST_SEGMENTED_PATH,
                 DST_ORIGINAL_PATH, ORIGINAL_RESIZED_PATH, SEGMENTED_RESIZED_PATH, SEGMENTED_ONE_HOT_PATH,
                 RESULTS_PATH, LABEL_TYPES_PATH):
        self.DST_PARENT_DIR = DST_PARENT_DIR
        self.PARENT_DIR = PARENT_DIR
        self.ORIGINAL_PATH = ORIGINAL_PATH
        self.SEGMENTED_PATH = SEGMENTED_PATH
        self.DST_SEGMENTED_PATH = DST_SEGMENTED_PATH
        self.DST_ORIGINAL_PATH = DST_ORIGINAL_PATH
        self.ORIGINAL_RESIZED_PATH = ORIGINAL_RESIZED_PATH
        self.SEGMENTED_RESIZED_PATH = SEGMENTED_RESIZED_PATH
        self.SEGMENTED_ONE_HOT_PATH = SEGMENTED_ONE_HOT_PATH
        self.RESULTS_PATH = RESULTS_PATH
        self.LABEL_TYPES_PATH = LABEL_TYPES_PATH
        pass

    def check_building_vaihingen(self, b, g, r):
        return b == 255 and g == 0 and r == 0

    def check_road_vaihingen(self, b, g, r):
        pass

    def check_road_vaihingen(self, b, g, r):
        return b == 255 and g == 255 and r == 255

    # returns all unique classes (colors) detected in a set of images
    def get_unique_values(self, images):
        return_array = []
        for img in images:
            for i in range(0, img.shape[0]):
                for j in range(0, img.shape[1]):
                    # pick a class "paved area"
                    (b, g, r) = img[i, j]
                    # if it is not a building label or road label, make it 0 (black)
                    if (b, g, r) not in return_array:
                        return_array.append((b, g, r))
        return return_array

    # converts the rgb ground truth to the one-hot version
    def _to_one_hot(self, input_img):
        encoded_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, N_OF_LABELS), dtype=np.uint8)

        # check each pixel of current mask
        for row_idx in range(0, input_img.shape[0]):
            for col_idx in range(0, input_img.shape[1]):
                for label_idx in range(0, N_OF_LABELS):
                    # if current pixel value has the current label value flag it in result
                    # it uses the fact that all return_array values are initially 0
                    if tuple(input_img[row_idx, col_idx]) == labels[label_idx]:
                        encoded_img[row_idx][col_idx] = one_hot_labels[label_idx]
        # save the encoded image as tiff ?

        return encoded_img

    # returns channel index that has the greatest value
    def get_max_channel_idx(self, image_channels):
        max_idx = 0
        for channel in image_channels:
            if image_channels[max_idx] < channel:
                idxs = np.where(image_channels == channel)
                max_idx = idxs[0][0]

        return max_idx

    # prases the model prediction to an rgb image similar to the ground-truth version
    def parse_prediction(self, predicted_image, labels):
        return_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

        for row_idx in range(0, predicted_image.shape[0]):
            for col_idx in range(0, predicted_image.shape[1]):
                try:
                    max_val_idx = self.get_max_channel_idx(predicted_image[row_idx][col_idx])
                    label = labels[max_val_idx]
                    return_array[row_idx][col_idx] = label
                except:
                    print("Exceptie la parsare onehot --> observable image")

        return return_array

    # returns the labeled ids of images that are not encoded yet
    def get_labeled_encoded_difference(self, labeled_img_ids, encoded_img_ids):
        difference_list = []
        # for each label id
        for label_id in labeled_img_ids:
            encoded_label_id = label_id.split('.')[0] + '.tif'
            # if the expected encoded id is not in the encoded images, append to the result
            if encoded_label_id not in encoded_img_ids:
                difference_list.append(label_id)

        return difference_list

    def one_hot_th_function(self, train_ids, labeled_images, lock):
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

            mask_ = imread(self.DST_PARENT_DIR + self.SEGMENTED_RESIZED_PATH + id_)[:, :, :IMG_CHANNELS]
            encoded_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, N_OF_LABELS), dtype=np.uint8)

            # check each pixel of current mask
            for row_idx in range(0, mask_.shape[0]):
                for col_idx in range(0, mask_.shape[1]):
                    for label_idx in range(0, N_OF_LABELS):
                        # if current pixel value has the current label value flag it in result
                        # it uses the fact that all return_array values are initially 0
                        if tuple(mask_[row_idx, col_idx]) == labels[label_idx]:
                            encoded_img[row_idx][col_idx] = one_hot_labels[label_idx]
            # save the encoded image as tiff ?

            new_file_name = self.DST_PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH + id_.split('.')[0] + '.tif'
            tifffile.imwrite(new_file_name, encoded_img)
            # saves the encoded image

    # used to convert an already augmented dataset that has not been one-hot-encoded
    # augmentation funtion already one-hot-encodes images because its more efficient
    def to_one_hot_and_save(self):
        segmented_ids = os.listdir(self.PARENT_DIR + self.SEGMENTED_RESIZED_PATH)
        one_hot_ids = os.listdir(self.PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH)
        # obtains the images that are not encoded in case of an interuption while converting
        not_encoded_ids = self.get_labeled_encoded_difference(segmented_ids, one_hot_ids)

        no_threads = 2
        aug_threads = []
        lock = threading.Lock()
        factor = int(len(not_encoded_ids) / no_threads)
        for thIdx in range(0, len(not_encoded_ids), factor):
            if thIdx == no_threads - 1:
                # give rest of the data to last thread
                th = Thread(target=self.one_hot_th_function,
                            args=(not_encoded_ids[thIdx:], [], lock,))
                aug_threads.append(th)
                th.start()
            else:
                # give thread a chunk of data to process
                limit = thIdx + factor - 1
                th = Thread(target=self.one_hot_th_function, args=(not_encoded_ids[thIdx: limit],
                                                                   [], lock,))
                aug_threads.append(th)
                th.start()

        for th in aug_threads:
            th.join()

    # manual batching, without a generator => multiple train sessions with 2k images at a time
    def batch_data(self, dataset, batch_size):
        data_len = len(dataset)
        nr_of_batches = int(data_len / batch_size)
        last_iter = nr_of_batches * batch_size

        batch_array = []
        for batch_idx in range(0, data_len, batch_size):
            crt_batch = []

            if batch_idx == last_iter:
                for image_name in dataset[batch_idx:]:
                    crt_batch.append(image_name)
            else:
                limit = batch_idx + batch_size
                for image_name in dataset[batch_idx:limit]:
                    crt_batch.append(image_name)
            batch_array.append(crt_batch)

        return batch_array

    # function that a thread uses to augment dataset
    def thread_aug_data_function(self, data_fragment, source_orig_path, source_segm_path, destination_orig_path,
                                 destination_segm_path):
        for n, id_ in tqdm(enumerate(data_fragment), total=len(data_fragment)):

            original_img = imread(source_orig_path + id_)[:, :, :IMG_CHANNELS]

            # reads and encodes the image
            segmented_img = imread(source_segm_path + id_)[:, :, :IMG_CHANNELS]
            segmented_img = self._to_one_hot(segmented_img)

            aug_originals = [original_img]
            aug_segmented = [segmented_img]

            # roatated images
            rot_o_90 = np.rot90(original_img, k=1)
            aug_originals.append(rot_o_90)
            rot_o_180 = np.rot90(original_img, k=2)
            aug_originals.append(rot_o_180)
            rot_o_270 = np.rot90(original_img, k=3)
            aug_originals.append(rot_o_270)

            # horizontally flipped images
            flip_o_h_org = np.flip(original_img, 1)
            aug_originals.append(flip_o_h_org)
            flip_o_h_90 = np.flip(rot_o_90, 1)
            aug_originals.append(flip_o_h_90)

            # same operations for segmented data
            rot_s_90 = np.rot90(segmented_img, k=1)
            aug_segmented.append(rot_s_90)
            rot_s_180 = np.rot90(segmented_img, k=2)
            aug_segmented.append(rot_s_180)
            rot_s_270 = np.rot90(segmented_img, k=3)
            aug_segmented.append(rot_s_270)

            flip_s_h_org = np.flip(segmented_img, 1)
            aug_segmented.append(flip_s_h_org)
            flip_s_h_90 = np.flip(rot_s_90, 1)
            aug_segmented.append(flip_s_h_90)

            for i in range(0, len(aug_segmented)):
                # saves all the augmented originals
                rgb_orig = cv2.cvtColor(aug_originals[i], cv2.COLOR_BGR2RGB)
                cv2.imwrite(destination_orig_path + id_.split('.')[0] + '_' + str((i + 1)) + '.png', rgb_orig)

                # saves all the augmented segmented
                new_file_name = self.DST_PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH + id_.split('.')[0] + '_' + str(
                    (i + 1)) + '.tif'
                tifffile.imwrite(new_file_name, aug_segmented[i])

    # augments dataset and one-hot encodes ground-truth in a multi-thread way
    def augment_data_set(self):
        data_ids = os.listdir(self.PARENT_DIR + self.DST_ORIGINAL_PATH)
        one_hot_ids = os.listdir(self.PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH)
        # obtains the images that are not encoded
        not_encoded_ids = self.get_labeled_encoded_difference(data_ids, one_hot_ids)
        no_threads = 4
        aug_threads = []
        destination_orig_path = self.DST_PARENT_DIR + self.ORIGINAL_RESIZED_PATH
        destination_segm_path = self.DST_PARENT_DIR + self.SEGMENTED_RESIZED_PATH

        source_orig_path = self.DST_PARENT_DIR + self.DST_ORIGINAL_PATH
        source_segm_path = self.DST_PARENT_DIR + self.DST_SEGMENTED_PATH

        factor = int(len(not_encoded_ids) / no_threads)
        for thIdx in range(0, len(not_encoded_ids), factor):
            if thIdx == no_threads - 1:
                # give rest of the data to last thread
                th = Thread(target=self.thread_aug_data_function, args=(not_encoded_ids[thIdx:],
                                                                        source_orig_path, source_segm_path,
                                                                        destination_orig_path, destination_segm_path,))
                aug_threads.append(th)
                th.start()
            else:
                # give thread a chunk of data to process
                th = Thread(target=self.thread_aug_data_function, args=(not_encoded_ids[thIdx: thIdx + factor - 1],
                                                                        source_orig_path, source_segm_path,
                                                                        destination_orig_path, destination_segm_path,))
                aug_threads.append(th)
                th.start()

        for th in aug_threads:
            th.join()

    # resize extracted segmented
    def resize_segmented(self):
        ids = os.listdir(self.PARENT_DIR + self.DST_SEGMENTED_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            path = self.PARENT_DIR + self.DST_SEGMENTED_PATH + id_.split('.')[0] + '.png'
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(self.DST_PARENT_DIR + self.SEGMENTED_RESIZED_PATH + id_, resized_img)

    # resizes extracted segmented data, but keeps only two labels: road and building
    def resize_segmented_building_road(self):
        ids = os.listdir(self.PARENT_DIR + self.DST_SEGMENTED_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            path = self.PARENT_DIR + self.DST_SEGMENTED_PATH + id_.split('.')[0] + '.png'
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

            for i in range(0, resized_img.shape[0]):
                for j in range(0, resized_img.shape[1]):
                    # pick a class "paved area"
                    (b, g, r) = resized_img[i, j]
                    # if it is not a building label or road label, make it 0 (black)
                    if not self.check_road_vaihingen(b, g, r) and not self.check_building_vaihingen(b, g, r):
                        resized_img[i, j] = (0, 0, 0)

            cv2.imwrite(self.DST_PARENT_DIR + self.SEGMENTED_RESIZED_PATH + id_, resized_img)

    # resize extracted original
    def resize_original(self):
        ids = os.listdir(self.PARENT_DIR + self.DST_ORIGINAL_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            path = self.PARENT_DIR + self.DST_ORIGINAL_PATH + id_
            img_ = cv2.imread(path, cv2.IMREAD_COLOR)

            resized_img = cv2.resize(img_, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imwrite(self.DST_PARENT_DIR + self.ORIGINAL_RESIZED_PATH + id_, resized_img)

    # original image data set extraction
    def split_original(self):
        ids = os.listdir(self.PARENT_DIR + self.ORIGINAL_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            count = 0

            path = self.PARENT_DIR + self.ORIGINAL_PATH + id_
            img_ = cv2.imread(path, cv2.IMREAD_COLOR)

            fragmentShape = 1000, 1000, 3
            fragment = np.zeros(fragmentShape, dtype=np.uint8)
            for offset_i in range(0, img_.shape[0] // 1000):
                for offset_j in range(0, img_.shape[0] // 1000):

                    for i in range(0, img_.shape[0] // 6):
                        for j in range(0, img_.shape[1] // 6):
                            fragment[i, j] = img_[i + offset_i * 1000, j + offset_j * 1000]

                    cv2.imwrite(
                        self.DST_PARENT_DIR + self.DST_ORIGINAL_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                        fragment)
                    count = count + 1

            for offset_i in range(0, img_.shape[0] // 1000 - 1):
                for offset_j in range(0, img_.shape[0] // 1000 - 1):

                    for i in range(0, img_.shape[0] // 6):
                        for j in range(0, img_.shape[1] // 6):
                            fragment[i, j] = img_[
                                500 + i + offset_i * 1000, 500 + j + offset_j * 1000]  # 500 pt a porni de la 500, nu 0 si a termina la 5500

                    cv2.imwrite(
                        self.DST_PARENT_DIR + self.DST_ORIGINAL_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                        fragment)
                    count = count + 1

    # segmented image data extractions
    def split_segmented(self):
        ids = os.listdir(self.PARENT_DIR + self.SEGMENTED_PATH)
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            count = 0

            path = self.PARENT_DIR + self.SEGMENTED_PATH + id_
            img_ = cv2.imread(path, cv2.IMREAD_COLOR)

            fragmentShape = 1000, 1000, 3
            fragment = np.zeros(fragmentShape, dtype=np.uint8)
            for offset_i in range(0, img_.shape[0] // 1000):
                for offset_j in range(0, img_.shape[0] // 1000):

                    for i in range(0, img_.shape[0] // 6):
                        for j in range(0, img_.shape[1] // 6):
                            fragment[i, j] = img_[i + offset_i * 1000, j + offset_j * 1000]

                    cv2.imwrite(
                        self.DST_PARENT_DIR + self.DST_SEGMENTED_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                        fragment)
                    count = count + 1

            for offset_i in range(0, img_.shape[0] // 1000 - 1):
                for offset_j in range(0, img_.shape[0] // 1000 - 1):

                    for i in range(0, img_.shape[0] // 6):
                        for j in range(0, img_.shape[1] // 6):
                            fragment[i, j] = img_[
                                500 + i + offset_i * 1000, 500 + j + offset_j * 1000]  # 500 pt a porni de la 500, nu 0 si a termina la 5500

                    cv2.imwrite(
                        self.DST_PARENT_DIR + self.DST_SEGMENTED_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                        fragment)
                    count = count + 1

    # returns two custom AerialDataGenerator, one for training and another for validation
    # by default validation split is 0.2 but if another value is required the parameter must be specified
    # if validation split is less than 0.0 or greater than 1.0 the validation generator shall be None
    def get_generator(self, validation_split=0.2, batch_size=20):

        if not os.path.exists(self.PARENT_DIR + self.ORIGINAL_RESIZED_PATH):
            print('path: "' + self.PARENT_DIR + self.ORIGINAL_RESIZED_PATH + ' " not found')
            exit(1)
        if not os.path.exists(self.PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH):
            print('path: "' + self.PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH + ' " not found')
            exit(1)

        train_ids = os.listdir(self.PARENT_DIR + self.ORIGINAL_RESIZED_PATH)
        random.shuffle(train_ids)
        train_fragment = []
        validation_fragment = []
        train_generator = None
        validation_generator = None

        if 1.0 > validation_split > 0.0:
            # splits train set from validation set
            split_idx = int(len(train_ids) * validation_split)
            train_fragment = train_ids[split_idx:]
            validation_fragment = train_ids[:split_idx]

            validation_generator = adGenerator.AerialDataGenerator(validation_fragment, self.DST_PARENT_DIR,
                                                                   self.ORIGINAL_RESIZED_PATH,
                                                                   self.SEGMENTED_ONE_HOT_PATH, batch_size=batch_size)
        else:
            train_fragment = train_ids

        train_generator = adGenerator.AerialDataGenerator(train_fragment, self.DST_PARENT_DIR,
                                                          self.ORIGINAL_RESIZED_PATH,
                                                          self.SEGMENTED_ONE_HOT_PATH, batch_size=batch_size)

        return train_generator, validation_generator

    # don't uses this, it is just a code backup
    def old_dataset_batching_backup(self):

        train_ids = os.listdir(self.PARENT_DIR + self.ORIGINAL_RESIZED_PATH)
        random.shuffle(train_ids)
        dataset_batches = self.batch_data(train_ids, BATCH_SIZE)
        for batch in dataset_batches:

            X_train = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
            Y_train = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, N_OF_LABELS), dtype=np.uint8)

            for n, id_ in tqdm(enumerate(batch), total=len(batch)):
                img = imread(self.DST_PARENT_DIR + self.ORIGINAL_RESIZED_PATH + train_ids[n])[:, :, :IMG_CHANNELS]
                X_train[n] = img

                mask = imread(self.DST_PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH + train_ids[n].split('.')[0] + '.tif')[
                       :, :,
                       :N_OF_LABELS]
                Y_train[n] = mask

    # creates an input pipeline
    def get_input_pipeline(self):
        # read ids of the input images
        # the images are inside a subfolder with the same name because of the 'get_generator' function
        originals_root_dir = self.PARENT_DIR + self.ORIGINAL_RESIZED_PATH + self.ORIGINAL_RESIZED_PATH
        masks_root_dir = self.PARENT_DIR + self.SEGMENTED_RESIZED_PATH
        # get an array with relative path for each image
        originals_ids = os.listdir(originals_root_dir)
        originals_ids.sort(reverse=False)
        originals_full_paths = [originals_root_dir + id_ for id_ in originals_ids]

        mask_ids = os.listdir(masks_root_dir)
        mask_ids.sort(reverse=False)
        masks_full_paths = [masks_root_dir + id_ for id_ in mask_ids]

        # create dataset using relative path names
        originals_ds = tf.data.Dataset.from_tensor_slices(originals_full_paths)
        masks_ds = tf.data.Dataset.from_tensor_slices(masks_full_paths)

        train_ds = tf.data.Dataset.zip((originals_ds, masks_ds))
        train_ds = train_ds.map(lambda x, y: tf.py_function(func=combine_img_masks,
                                                            inp=[x, y], Tout=(tf.uint8, tf.uint8)),
                                num_parallel_calls=4,
                                deterministic=False)

        train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE).cache()

        return train_ds_batched

    # returns a dict like
    # {'label 1': {'precision': 0.5,
    #              'recall': 1.0,
    #              'f1-score': 0.67,
    #              'support': 1},
    #  'label 2': {...},
    #  ...
    #  }
    # from sklearn classification_report return type
    def _get_statistics_dict(self):
        stats = {
            '0': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '1': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '2': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '3': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '4': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            '5': {'precision': 0.0,
                  'recall': 0.0,
                  'f1-score': 0.0,
                  'support': 0},
            'accuracy': 0.0,
            'macro avg': {'precision': 0.0,
                          'recall': 0.0,
                          'f1-score': 0.0,
                          'support': 0},
            'weighted avg': {'precision': 0.0,
                             'recall': 0.0,
                             'f1-score': 0.0,
                             'support': 0},
        }
        return stats

    def print_per_class_statistics(self, validation_split, model: keras.Model):

        global normalized_conf_matrix, initial_conf_matrix

        train_ids = os.listdir(self.PARENT_DIR + self.ORIGINAL_RESIZED_PATH)
        random.shuffle(train_ids)
        split_idx = int(len(train_ids) * validation_split)
        validation_fragment = train_ids[:split_idx]
        validation_size = len(validation_fragment)

        batch_size = 4
        batch_idx = 0
        no_batches = 0

        images = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        ground_truth = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, N_OF_LABELS), dtype=np.uint8)

        stats = self._get_statistics_dict()
        normalized_conf_matrix = np.zeros((N_OF_LABELS, N_OF_LABELS), dtype=np.float)

        for n, id_ in tqdm(enumerate(validation_fragment), total=len(validation_fragment)):
            img = imread(self.DST_PARENT_DIR + self.ORIGINAL_RESIZED_PATH + train_ids[n])[:, :, :IMG_CHANNELS]
            images[batch_idx] = img

            mask = imread(self.DST_PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH + train_ids[n].split('.')[0] + '.tif')[:, :,
                   :N_OF_LABELS]
            ground_truth[batch_idx] = mask
            batch_idx += 1

            # last image batch or a complete batch
            if batch_idx == batch_size - 1 or n == validation_size - 1:
                batch_idx = 0
                no_batches += 1

                predictions = model.predict(images)
                predictions_max_score = np.argmax(predictions, axis=3).flatten()
                ground_truth_max_score = np.argmax(ground_truth, axis=3).flatten()

                initial_conf_matrix = tf.math.confusion_matrix(num_classes=6,
                                                               labels=ground_truth_max_score,
                                                               predictions=predictions_max_score).numpy()
                normalized_conf_matrix += np.around(normalize(initial_conf_matrix, axis=1, norm='l1'), decimals=2)

                report = classification_report(ground_truth_max_score, predictions_max_score, output_dict=True)

                for label_i in report.keys():
                    if label_i != 'accuracy':
                        for metric in report[label_i].keys():
                            stats[label_i][metric] += report[label_i][metric]
                    else:
                        stats[label_i] += report[label_i]

        # final avg statistics
        for label_i in stats.keys():
            if label_i != 'accuracy':
                for metric in stats[label_i].keys():
                    stats[label_i][metric] /= no_batches
            else:
                stats[label_i] /= no_batches
        print(stats)

        # store it in a file
        with open(dict_stats_file, 'w') as file:
            json.dump(stats, file, indent=4)

        # final confusion matrix
        normalized_conf_matrix = np.around(normalize(normalized_conf_matrix, axis=1, norm='l1'), decimals=2)
        print(normalized_conf_matrix)

        # # store it in a file
        conf_mat_dict = {
            'confusion_matrix': normalized_conf_matrix
        }
        with open(confusion_matrix_file, 'w') as file:
            json.dump(conf_mat_dict, file, indent=4)

    def segment_data(self, model: keras.Model):

        train_ids = os.listdir(self.DST_PARENT_DIR + self.DST_ORIGINAL_PATH)

        validation_size = len(train_ids)

        batch_size = 4
        batch_idx = 0
        no_batches = 0

        images = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        image_name_ids = []

        predictionPath = './Results/000_segmentation_results/'

        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            img = imread(self.PARENT_DIR + self.DST_ORIGINAL_PATH + train_ids[n])[:, :, :IMG_CHANNELS]
            images[batch_idx] = img
            image_name_ids.append(train_ids[n])
            batch_idx += 1
            if batch_idx == batch_size or n == validation_size - 1:
                batch_idx = 0
                no_batches += 1

                # start_time = time.time()
                predictions = model.predict(images)
                # print("--- %s seconds ---" % (time.time() - start_time))

                for i in range(batch_size):
                    parsed_image = self.parse_prediction(predictions[i], labels)
                    parsed_image = cv2.cvtColor(parsed_image, cv2.COLOR_BGR2RGB)

                    cv2.imwrite(predictionPath + image_name_ids[i], parsed_image)

                image_name_ids.clear()

    def get_data_set_class_balance(self):
        train_ids = os.listdir(self.DST_PARENT_DIR + self.DST_SEGMENTED_PATH)

        total_no_pixels = 0
        class_cnt = {
            (255, 255, 255): 0.0,  # white, paved area/road
            (0, 255, 255): 0.0,  # light blue, low vegetation
            (0, 0, 255): 0.0,  # blue, buildings
            (0, 255, 0): 0.0,  # green, high vegetation
            (255, 0, 0): 0.0,  # red, bare earth
            (255, 255, 0): 0.0  # yellow, vehicle/car
        }

        class_aux = {
            (255, 255, 255): 0.0,  # white, paved area/road
            (0, 255, 255): 0.0,  # light blue, low vegetation
            (0, 0, 255): 0.0,  # blue, buildings
            (0, 255, 0): 0.0,  # green, high vegetation
            (255, 0, 0): 0.0,  # red, bare earth
            (255, 255, 0): 0.0  # yellow, vehicle/car
        }

        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            img = imread(self.PARENT_DIR + self.DST_SEGMENTED_PATH + train_ids[n])[:, :, :IMG_CHANNELS]
            for row_idx in range(0, img.shape[0]):
                for col_idx in range(0, img.shape[1]):
                    class_aux[tuple(img[row_idx, col_idx])] += 1.0
                    total_no_pixels += 1.0
            for key in class_aux.keys():
                class_aux[key] /= float(total_no_pixels)
                class_cnt[key] += class_aux[key]

            total_no_pixels = 0.0
            print(class_aux)

        for key in class_cnt.keys():
            class_cnt[key] /= len(train_ids)
        with open('./class_balance.json', 'w') as file:
            json.dump(class_cnt, file, indent=4)
        exit(0)

    def manual_model_testing(self, model: keras.Model):
        current_day = datetime.datetime.now()
        test_set_size = 5
        train_ids = os.listdir(self.PARENT_DIR + self.ORIGINAL_RESIZED_PATH)
        random_images_idx = random.sample(train_ids, test_set_size)

        X_train = np.zeros((test_set_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        ground_truth = np.zeros((test_set_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

        for n, id_ in tqdm(enumerate(random_images_idx), total=len(random_images_idx)):
            img = imread(self.DST_PARENT_DIR + self.ORIGINAL_RESIZED_PATH + train_ids[n])[:, :, :IMG_CHANNELS]
            X_train[n] = img

            mask = imread(self.DST_PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH + train_ids[n].split('.')[0] + '.tif')[:, :,
                   :N_OF_LABELS]
            ground_truth[n] = self.parse_prediction(mask, labels)

        preds_train = model.predict(X_train, verbose=1)

        print("Enter 0 to exit, any other number to predict an image: ")
        continue_flag = input()

        while int(continue_flag) > 0:
            i = random.randint(0, test_set_size - 1)

            trainPath = "%s%sstrain%03d.png" % (self.RESULTS_PATH, self.LABEL_TYPES_PATH, i)
            controlPath = "%s%scontrolMask%03d.png" % (
                self.RESULTS_PATH,
                self.LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(current_day.day).zfill(2) + '/',
                i)
            predictionPath = "%s%sprediction%03d.png" % (
                self.RESULTS_PATH,
                self.LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(current_day.day).zfill(2) + '/',
                i)

            today_result_dir = self.RESULTS_PATH + self.LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(
                current_day.day).zfill(2)
            if not os.path.exists(today_result_dir):
                os.mkdir(today_result_dir)

            imshow(X_train[i])
            plt.savefig(trainPath)
            plt.show()

            imshow(np.squeeze(ground_truth[i]))
            plt.savefig(controlPath)
            plt.show()

            interpreted_prediction = self.parse_prediction(preds_train[i], labels)
            imshow(np.squeeze(interpreted_prediction))
            plt.savefig(predictionPath)
            plt.show()

            print("Enter 0 to exit, any positive number to predict another image: ")
            continue_flag = input()
