import os
import random
import threading
import tifffile
import numpy as np
import tensorflow as tf
import io
import AerialDataGenerator as adGenerator

from threading import Thread
from cv2 import cv2
from PIL import Image
from skimage.io import imread
from tqdm import tqdm

# Input images size
# original dimensions/16
IMG_WIDTH = 250
IMG_HEIGHT = 250
IMG_CHANNELS = 3
SAMPLE_SIZE = 20000
BATCH_SIZE = 16
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


def decode_png_img(img):
    img = tf.image.decode_png(img, channels=IMG_CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.uint8)

    return img


def decode_tif_img(img):
    # img = tf.image.decode_image(img, channels=N_OF_LABELS, dtype=tf.dtypes.uint8)
    img = one_hot_enc(img.numpy())
    img = tf.convert_to_tensor(img, tf.uint8)
    img = tf.image.convert_image_dtype(img, tf.uint8)

    return img


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

    def to_one_hot(self, N_OF_LABELS, label_array):
        return_array = np.zeros((len(label_array), IMG_HEIGHT, IMG_WIDTH, N_OF_LABELS), dtype=np.uint8)

        print('Finding all existing labels')
        # labels = get_unique_values(label_array)
        print('found ' + str(len(labels)) + ' labels')

        if (N_OF_LABELS != len(labels)):
            print('Error, N_OF_LABELS must be equal to the number of unique values in dataset labes')
        else:
            # for each label index
            for label_idx in range(0, N_OF_LABELS):
                # check each pixel of all images in Y_train
                for image_idx in range(0, len(label_array)):
                    image = label_array[image_idx]
                    for row_idx in range(0, image.shape[0]):
                        for col_idx in range(0, image.shape[1]):
                            # if current pixel value has the current label value flag it in result
                            # it uses the fact that all return_array values are initially 0
                            if tuple(image[row_idx, col_idx]) == labels[label_idx]:
                                return_array[image_idx][row_idx][col_idx][label_idx] = 1  # or 255
                                # print("For " + str(image[row_idx, col_idx]) + " encoded " + str(return_array[image_idx][row_idx][col_idx]) + "label idx: " + str(label_idx))
        return labels, return_array

    def get_max_channel_idx(self, image_channels):
        max_idx = 0
        for channel in image_channels:
            if image_channels[max_idx] < channel:
                idxs = np.where(image_channels == channel)
                max_idx = idxs[0][0]

        return max_idx

    def parse_prediction(self, predicted_image, labels):
        return_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        # return_array = np.zeros((1000, 1000, IMG_CHANNELS), dtype=np.uint8)

        for row_idx in range(0, predicted_image.shape[0]):
            for col_idx in range(0, predicted_image.shape[1]):
                try:
                    max_val_idx = self.get_max_channel_idx(predicted_image[row_idx][col_idx])
                    label = labels[max_val_idx]
                    return_array[row_idx][col_idx] = label
                except:
                    print("Exceptie la parsare onehot --> observable image")

        return return_array

    # returns the labeled ids that are not encoded yet
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

            mask_ = labeled_images[n]
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

    def to_one_hot_and_save(self):
        segmented_ids = os.listdir(self.PARENT_DIR + self.SEGMENTED_RESIZED_PATH)
        one_hot_ids = os.listdir(self.PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH)
        # obtains the images that are not encoded
        not_encoded_ids = self.get_labeled_encoded_difference(segmented_ids, one_hot_ids)

        labeled_images = np.zeros((len(not_encoded_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

        for file_index, file_name in tqdm(enumerate(not_encoded_ids), total=len(not_encoded_ids)):
            mask_ = imread(self.DST_PARENT_DIR + self.SEGMENTED_RESIZED_PATH + file_name)[:, :, :IMG_CHANNELS]
            labeled_images[file_index] = mask_
        no_threads = 4
        aug_threads = []
        lock = threading.Lock()
        factor = int(len(not_encoded_ids) / no_threads)
        for thIdx in range(0, len(not_encoded_ids), factor):
            if thIdx == no_threads - 1:
                # give rest of the data to last thread
                th = Thread(target=self.one_hot_th_function,
                            args=(not_encoded_ids[thIdx:], labeled_images[thIdx:], lock,))
                aug_threads.append(th)
                th.start()
            else:
                # give thread a chunk of data to process
                limit = thIdx + factor - 1
                th = Thread(target=self.one_hot_th_function, args=(not_encoded_ids[thIdx: limit],
                                                                   labeled_images[thIdx: limit], lock,))
                aug_threads.append(th)
                th.start()

        for th in aug_threads:
            th.join()

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

    def augment_data_set(self):
        data_ids = os.listdir(self.PARENT_DIR + self.ORIGINAL_RESIZED_PATH)
        no_threads = 8
        aug_threads = []
        root_orig_path = self.DST_PARENT_DIR + self.ORIGINAL_RESIZED_PATH
        root_segm_path = self.DST_PARENT_DIR + self.SEGMENTED_RESIZED_PATH

        factor = int(len(data_ids) / no_threads)
        for thIdx in range(0, len(data_ids), factor):
            if thIdx == no_threads - 1:
                # give rest of the data to last thread
                th = Thread(target=self.thread_aug_data_function, args=(data_ids[thIdx:],
                                                                        root_orig_path, root_segm_path,))
                aug_threads.append(th)
                th.start()
            else:
                # give thread a chunk of data to process
                th = Thread(target=self.thread_aug_data_function, args=(data_ids[thIdx: thIdx + factor - 1],
                                                                        root_orig_path, root_segm_path,))
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

    def thread_aug_data_function(self, data_fragment, root_orig_path, root_segm_path):
        for n, id_ in tqdm(enumerate(data_fragment), total=len(data_fragment)):
            # print(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + id_)
            original_img = imread(root_orig_path + id_)[:, :, :IMG_CHANNELS]
            segmented_img = imread(root_segm_path + id_)[:, :, :IMG_CHANNELS]
            aug_originals = []
            aug_segmented = []

            # roatated images
            rot_o_90 = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
            aug_originals.append(rot_o_90)
            rot_o_180 = cv2.rotate(original_img, cv2.ROTATE_180)
            aug_originals.append(rot_o_180)
            rot_o_270 = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            aug_originals.append(rot_o_270)

            # horizontally flipped images
            flip_o_h_org = cv2.flip(original_img, 1)
            aug_originals.append(flip_o_h_org)
            flip_o_h_90 = cv2.flip(rot_o_90, 1)
            aug_originals.append(flip_o_h_90)
            flip_o_h_180 = cv2.flip(rot_o_180, 1)
            aug_originals.append(flip_o_h_180)
            flip_o_h_270 = cv2.flip(rot_o_270, 1)
            aug_originals.append(flip_o_h_270)

            # brighter images
            brighter_o_ = cv2.add(original_img, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_)
            brighter_o_rot_o_90 = cv2.add(rot_o_90, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_rot_o_90)
            brighter_o_rot_o_180 = cv2.add(rot_o_180, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_rot_o_180)
            brighter_o_rot_o_270 = cv2.add(rot_o_270, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_rot_o_270)
            brighter_o_flip_o_h_org = cv2.add(flip_o_h_org, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_flip_o_h_org)
            brighter_o_flip_o_h_90 = cv2.add(flip_o_h_90, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_flip_o_h_90)
            brighter_o_flip_o_h_180 = cv2.add(flip_o_h_180, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_flip_o_h_180)
            brighter_o_flip_o_h_270 = cv2.add(flip_o_h_270, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(brighter_o_flip_o_h_270)

            # dimmer images
            dimmer_o_ = cv2.subtract(original_img, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_)
            dimmer_o_rot_o_90 = cv2.subtract(rot_o_90, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_rot_o_90)
            dimmer_o_rot_o_180 = cv2.subtract(rot_o_180, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_rot_o_180)
            dimmer_o_rot_o_270 = cv2.subtract(rot_o_270, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_rot_o_270)
            dimmer_o_flip_o_h_org = cv2.subtract(flip_o_h_org, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_flip_o_h_org)
            dimmer_o_flip_o_h_90 = cv2.subtract(flip_o_h_90, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_flip_o_h_90)
            dimmer_o_flip_o_h_180 = cv2.subtract(flip_o_h_180, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_flip_o_h_180)
            dimmer_o_flip_o_h_270 = cv2.subtract(flip_o_h_270, np.array([random.randint(50, 80) / 1.0]))
            aug_originals.append(dimmer_o_flip_o_h_270)

            # same operations for segmented data
            rot_s_90 = cv2.rotate(segmented_img, cv2.ROTATE_90_CLOCKWISE)
            aug_segmented.append(rot_s_90)
            rot_s_180 = cv2.rotate(segmented_img, cv2.ROTATE_180)
            aug_segmented.append(rot_s_180)
            rot_s_270 = cv2.rotate(segmented_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            aug_segmented.append(rot_s_270)

            flip_s_h_org = cv2.flip(segmented_img, 1)
            aug_segmented.append(flip_s_h_org)
            flip_s_h_90 = cv2.flip(rot_s_90, 1)
            aug_segmented.append(flip_s_h_90)
            flip_s_h_180 = cv2.flip(rot_s_180, 1)
            aug_segmented.append(flip_s_h_180)
            flip_s_h_270 = cv2.flip(rot_s_270, 1)
            aug_segmented.append(flip_s_h_270)

            # brighter segmented images
            aug_segmented.append(segmented_img.copy())
            aug_segmented.append(rot_s_90.copy())
            aug_segmented.append(rot_s_180.copy())
            aug_segmented.append(rot_s_270.copy())
            aug_segmented.append(flip_s_h_org.copy())
            aug_segmented.append(flip_s_h_90.copy())
            aug_segmented.append(flip_s_h_180.copy())
            aug_segmented.append(flip_s_h_270.copy())

            # dimmer segmented images
            aug_segmented.append(segmented_img.copy())
            aug_segmented.append(rot_s_90.copy())
            aug_segmented.append(rot_s_180.copy())
            aug_segmented.append(rot_s_270.copy())
            aug_segmented.append(flip_s_h_org.copy())
            aug_segmented.append(flip_s_h_90.copy())
            aug_segmented.append(flip_s_h_180.copy())
            aug_segmented.append(flip_s_h_270.copy())

            for i in range(0, len(aug_segmented)):
                # saves all the augmented originals
                rgb_orig = cv2.cvtColor(aug_originals[i], cv2.COLOR_BGR2RGB)
                cv2.imwrite(root_orig_path + id_.split('.')[0] + '_' + str((i + 1)) + '.png', rgb_orig)
                # saves all the augmented segmented
                rgb_segm = cv2.cvtColor(aug_segmented[i], cv2.COLOR_BGR2RGB)
                cv2.imwrite(root_segm_path + id_.split('.')[0] + '_' + str((i + 1)) + '.png', rgb_segm)

        pass

    # returns two custom AerialDataGenerator, one for training and another for validation
    # by default validation split is 0.2 but if another value is required the parameter must be specified
    # if validation split is less than 0.0 or greater than 1.0 the validation generator shall be None
    def get_generator(self, validation_split=0.2, batch_size=20):

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
        # print(train_ds.element_spec)
        # train_ds.prefetch(10)
        # for image, label in train_ds.take(1):
        #     print(image.shape)
        #     print(label.shape)
        #     print()

        # exit(1)
        return train_ds_batched

    def print_per_class_statistics(self, ground_truth_set, predicition_set):
        class_statistics = {
            (255, 255, 255): (0, 0),  # white, paved area/road
            (0, 255, 255): (0, 0),  # light blue, low vegetation
            (0, 0, 255): (0, 0),  # blue, buildings
            (0, 255, 0): (0, 0),  # green, high vegetation
            (255, 0, 0): (0, 0),  # red, bare earth
            (255, 255, 0): (0, 0)  # yellow, vehicle/car
        }

        for idx in range(0, len(predicition_set)):
            pred = self.parse_prediction(predicition_set[idx], labels)
            for i in range(0, pred.shape[0]):
                for j in range(0, pred.shape[1]):
                    hit, miss = class_statistics[tuple(ground_truth_set[idx][i][j])]
                    if pred[i][j] == ground_truth_set[idx][i][j]:
                        hit += 1
                    else:
                        miss += 1
                    class_statistics[tuple(ground_truth_set[idx][i][j])] = (hit, miss)

        for i in range(0, len(labels)):
            hit, miss = class_statistics[labels[i]]
            print(labels_name[i] + ' accuracy: ' + str(hit / (hit + miss)) + '\n')
