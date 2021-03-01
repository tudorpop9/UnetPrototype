import tensorflow as tf
import os
import numpy as np
import random
from tqdm import tqdm
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from cv2 import cv2
from threading import Thread, Lock
from time import sleep

# Input images size
# original dimensions/16
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3
SAMPLE_SIZE = 20000
N_OF_LABELS = 6

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

RESULTS_PATH = "./Results/"
LABEL_TYPES_PATH = "OnlyRoad/"
# consistent randomness and good-luck charm
seed = 42
np.random.seed = seed


##################################################################### pre-processing data set #######################################################################3

def check_building_vaihingen(b, g, r):
    return b == 255 and g == 0 and r == 0


def check_road_vaihingen(b, g, r):
    pass


def check_road_vaihingen(b, g, r):
    return b == 255 and g == 255 and r == 255


# original image data set extraction
def split_original():
    ids = os.listdir(PARENT_DIR + ORIGINAL_PATH)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        count = 0

        path = PARENT_DIR + ORIGINAL_PATH + id_
        img_ = cv2.imread(path, cv2.IMREAD_COLOR)

        fragmentShape = 1000, 1000, 3
        fragment = np.zeros(fragmentShape, dtype=np.uint8)
        for offset_i in range(0, img_.shape[0] // 1000):
            for offset_j in range(0, img_.shape[0] // 1000):

                for i in range(0, img_.shape[0] // 6):
                    for j in range(0, img_.shape[1] // 6):
                        fragment[i, j] = img_[i + offset_i * 1000, j + offset_j * 1000]

                cv2.imwrite(DST_PARENT_DIR + DST_ORIGINAL_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                            fragment)
                count = count + 1

        for offset_i in range(0, img_.shape[0] // 1000 - 1):
            for offset_j in range(0, img_.shape[0] // 1000 - 1):

                for i in range(0, img_.shape[0] // 6):
                    for j in range(0, img_.shape[1] // 6):
                        fragment[i, j] = img_[
                            500 + i + offset_i * 1000, 500 + j + offset_j * 1000]  # 500 pt a porni de la 500, nu 0 si a termina la 5500

                cv2.imwrite(DST_PARENT_DIR + DST_ORIGINAL_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                            fragment)
                count = count + 1


# segmented image data extractions
def split_segmented():
    ids = os.listdir(PARENT_DIR + SEGMENTED_PATH)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        count = 0

        path = PARENT_DIR + SEGMENTED_PATH + id_
        img_ = cv2.imread(path, cv2.IMREAD_COLOR)

        fragmentShape = 1000, 1000, 3
        fragment = np.zeros(fragmentShape, dtype=np.uint8)
        for offset_i in range(0, img_.shape[0] // 1000):
            for offset_j in range(0, img_.shape[0] // 1000):

                for i in range(0, img_.shape[0] // 6):
                    for j in range(0, img_.shape[1] // 6):
                        fragment[i, j] = img_[i + offset_i * 1000, j + offset_j * 1000]

                cv2.imwrite(DST_PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                            fragment)
                count = count + 1

        for offset_i in range(0, img_.shape[0] // 1000 - 1):
            for offset_j in range(0, img_.shape[0] // 1000 - 1):

                for i in range(0, img_.shape[0] // 6):
                    for j in range(0, img_.shape[1] // 6):
                        fragment[i, j] = img_[
                            500 + i + offset_i * 1000, 500 + j + offset_j * 1000]  # 500 pt a porni de la 500, nu 0 si a termina la 5500

                cv2.imwrite(DST_PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0] + "_" + str(count) + ".png",
                            fragment)
                count = count + 1


# resize extracted original
def resize_original():
    ids = os.listdir(PARENT_DIR + DST_ORIGINAL_PATH)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        path = PARENT_DIR + DST_ORIGINAL_PATH + id_
        img_ = cv2.imread(path, cv2.IMREAD_COLOR)

        resized_img = cv2.resize(img_, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imwrite(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + id_, resized_img)


# resize extracted segmented
def resize_segmented():
    ids = os.listdir(PARENT_DIR + DST_SEGMENTED_PATH)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        path = PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0] + '.png'
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + id_, resized_img)


# resizes extracted segmented data, but keeps only two labels: road and building
def resize_segmented_building_road():
    ids = os.listdir(PARENT_DIR + DST_SEGMENTED_PATH)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        path = PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0] + '.png'
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

        for i in range(0, resized_img.shape[0]):
            for j in range(0, resized_img.shape[1]):
                # pick a class "paved area"
                (b, g, r) = resized_img[i, j]
                # if it is not a building label or road label, make it 0 (black)
                if not check_road_vaihingen(b, g, r) and not check_building_vaihingen(b, g, r):
                    resized_img[i, j] = (0, 0, 0)

        cv2.imwrite(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + id_, resized_img)


def thread_aug_data_function(data_fragment, root_orig_path, root_segm_path):
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
            cv2.imwrite(root_orig_path + id_ + '_' + str((i + 1)), rgb_orig)
            # saves all the augmented segmented
            rgb_segm = cv2.cvtColor(aug_segmented[i], cv2.COLOR_BGR2RGB)
            cv2.imwrite(root_segm_path + id_ + '_' + str((i + 1)), rgb_segm)

    pass


def augment_data_set():
    data_ids = os.listdir(PARENT_DIR + ORIGINAL_RESIZED_PATH)
    no_threads = 8
    aug_threads = []
    root_orig_path = DST_PARENT_DIR + ORIGINAL_RESIZED_PATH
    root_segm_path = DST_PARENT_DIR + SEGMENTED_RESIZED_PATH

    factor = int(len(data_ids) / no_threads)
    for thIdx in range(0, len(data_ids), factor):
        if thIdx == no_threads - 1:
            # give rest of the data to last thread
            th = Thread(target=thread_aug_data_function, args=(data_ids[thIdx:],
                                                               root_orig_path, root_segm_path,))
            aug_threads.append(th)
            th.start()
        else:
            # give thread a chunk of data to process
            th = Thread(target=thread_aug_data_function, args=(data_ids[thIdx: thIdx + factor - 1],
                                                               root_orig_path, root_segm_path,))
            aug_threads.append(th)
            th.start()

    for th in aug_threads:
        th.join()


# create initial model
def create_model():
    # Input layer
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Converts pixel value to float, and normalizes it
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = tf.keras.layers.MaxPooling2D((2, 2))(c5)

    # c61 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    # c61 = tf.keras.layers.Dropout(0.3)(c61)
    # c61 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c61)
    # c61 = tf.keras.layers.Dropout(0.3)(c61)
    # c61 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c61)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    # u6 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(u6)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    # u62 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    # u62 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(u62)
    # u62 = tf.keras.layers.concatenate([u62, c4])
    # c62 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u62)
    # c62 = tf.keras.layers.Dropout(0.2)(c62)
    # c62 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c62)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    # u7 = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(u7)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    # u8 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(u8)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    # u9 = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(u9)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # c9 = tf.keras.layers.Conv2D(6, (1, 1), activation='softmax')(c9)

    outputs = tf.keras.layers.Conv2D(6, (1, 1), activation='softmax')(c9)
    # outputs = tf.keras.layers.Lambda(lambda x: x*255)(outputs)

    adamOptimizer = tf.keras.optimizers.Adam(lr=0.00001)
    # categorical_crossentropy
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=adamOptimizer, loss=dice_coef_loss, metrics=['accuracy'])
    model.summary()

    return model


# another model
def create_another_model():
    model = tf.keras.models.Sequential()

    # Input layer

    model.add(tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    model.add(tf.keras.layers.Lambda(lambda x: x / 255))

    # convolutional layers 1
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # convolutional layers 2
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # convolutional layers 3
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # convolutional layers 4
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # convolutional layers 5
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    # de-convolutional layers 1
    model.add(tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    # model.add(tf.keras.layers.concatenate())
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # de-convolutional layers 2
    model.add(tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    # model.add(tf.keras.layers.concatenate())
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # de-convolutional layers 3
    model.add(tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    # model.add(tf.keras.layers.concatenate())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # de-convolutional layers 4
    model.add(tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'))
    # model.add(tf.keras.layers.concatenate())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # de-convolutional layers 5
    model.add(tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    # model.add(tf.keras.layers.concatenate())
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # de-convolutional layers 6
    model.add(tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same'))
    # model.add(tf.keras.layers.concatenate())
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(3, (3, 3), kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.Lambda(lambda x: x * 255))

    return model


def create_simpler_model():
    # Input layer
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Converts pixel value to float, and normalizes it
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2)
    u9 = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(u9)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(N_OF_LABELS, (1, 1), activation='softmax')(c9)
    # outputs = tf.keras.layers.Lambda(lambda x: x*255)(outputs)

    adamOptimizer = tf.keras.optimizers.Adam(lr=0.000001)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=adamOptimizer, loss=dice_coef_loss, metrics=['accuracy'])
    model.summary()

    return model


# some loss functions form stackoverflow and what not
smooth = 1


def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def get_unique_values(images):
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

def one_hot_th_function(labels, label_array, one_hot_result, resource_lock):
    # for each label index
    for label_idx in range(0, N_OF_LABELS):
        # check each pixel of all images in Y_train
        for image_idx in range(0, len(label_array)):
            image = label_array[image_idx]

            for row_idx in range(0, image.shape[0]):
                for col_idx in range(0, image.shape[1]):
                    # if current pixel value has the current label value flag it in result
                    # it uses the fact that all return_array values are initially 0
                    x = tuple(image[row_idx, col_idx])
                    y = labels[label_idx]
                    if tuple(image[row_idx, col_idx]) == labels[label_idx]:
                        # resource_lock.acquire()

                        one_hot_result[image_idx][row_idx][col_idx][label_idx] = 255  # or 1

                        # resource_lock.release()

def to_one_hot(N_OF_LABELS, label_array):
    return_array = np.zeros((len(label_array), IMG_HEIGHT, IMG_WIDTH, N_OF_LABELS), dtype=np.uint8)

    print('Finding all existing labels')
    labels = get_unique_values(label_array)
    print('found ' + str(len(labels)) + ' labels')

    if (N_OF_LABELS != len(labels)):
        print('Error, N_OF_LABELS must be equal to the number of unique values in dataset labes')
    else:
        # no_threads = 8
        # one_hot_threads = []
        # resource_lock = Lock()
        #
        # factor = int(SAMPLE_SIZE / no_threads)
        # for thIdx in range(0, SAMPLE_SIZE, factor):
        #     if int(thIdx/factor) == no_threads - 1:
        #         # give rest of the data to last thread
        #         th = Thread(target=one_hot_th_function, args=(labels, label_array[thIdx: SAMPLE_SIZE],
        #                                                       return_array, resource_lock,))
        #         one_hot_threads.append(th)
        #         th.start()
        #         print('last started thread')
        #     else:
        #         # give thread a chunk of data to process
        #         th = Thread(target=one_hot_th_function, args=(labels, label_array[thIdx: thIdx + factor - 1],
        #                                                       return_array, resource_lock,))
        #         one_hot_threads.append(th)
        #         th.start()
        #         print('started thread')
        #
        # for th in one_hot_threads:
        #     th.join()

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
                            return_array[image_idx][row_idx][col_idx][label_idx] = 255  # or 1
    return return_array


##################################################################### pre-processing data set #######################################################################

# split_original()
# split_segmented()

# resize_original()
# resize_segmented_building_road()
# resize_segmented()

# augment_data_set()

train_ids = os.listdir(PARENT_DIR + ORIGINAL_RESIZED_PATH)

random_data = random.sample(range(0, len(train_ids)), SAMPLE_SIZE)

X_train = np.zeros((SAMPLE_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((SAMPLE_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

for n, id_ in tqdm(enumerate(random_data), total=len(random_data)):
    img = imread(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + train_ids[id_])[:, :, :IMG_CHANNELS]
    X_train[n] = img

    mask = imread(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + train_ids[id_])[:, :, :IMG_CHANNELS]
    Y_train[n] = mask

image_we = random.randint(0, SAMPLE_SIZE)
# original
imshow(X_train[image_we])
plt.show()
cv2.imshow('img', X_train[image_we])
# control
imshow(np.squeeze(Y_train[image_we]))
plt.show()
cv2.imshow('mask', Y_train[image_we])

# creates the unet
model = create_model()
# model = create_simpler_model()

# waits for a decizion to train, or not to train
print('Enter a number: odd = load weights, or even = train model ')
to_train = input()

# applies one hot encoding on label images
print('One hot encoding labeled images..')
one_hot_y_train = to_one_hot(N_OF_LABELS, Y_train)

# if flag is an even number we perform a fit operation, training the model and save its best results
if int(to_train) % 2 == 0:
    callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

    results = model.fit(X_train, one_hot_y_train, validation_split=0.6, shuffle=True, batch_size=32, epochs=200,
                        callbacks=callbacks,
                        verbose=1)

    print('Training is done, do you want to save (overwrite) weights ?[y/n]')
    save_w_flag = input()
    if save_w_flag.lower() == 'y':
        model.save_weights('model_for_nuclei.h5')

# otherwise we load the weights from another run
else:
    model.load_weights('model_for_nuclei.h5')

idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train, verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0])], verbose=1)
# preds_test = model.predict(X_test, verbose = 1)

# Binarizationing the results
preds_train_t = (preds_train > 0.5).astype(np.uint8)

# random training sample

print("Enter 0 to exit, any other number to predict another image: ")
continue_flag = input()
while int(continue_flag) > 0:
    i = random.randint(0, len(preds_train_t))
    trainPath = "%s%sstrain%03d.png" % (RESULTS_PATH, LABEL_TYPES_PATH, i)
    controlPath = "%s%scontrolMask%03d.png" % (RESULTS_PATH, LABEL_TYPES_PATH, i)
    predictionPath = "%s%sprediction%03d.png" % (RESULTS_PATH, LABEL_TYPES_PATH, i)

    imshow(X_train[i])
    plt.savefig(trainPath)
    plt.show()

    imshow(np.squeeze(Y_train[i]))
    plt.savefig(controlPath)
    plt.show()

    imshow(np.squeeze(preds_train_t[i] * 255))
    plt.savefig(predictionPath)
    plt.show()

    print("Enter 0 to exit, any positive number to predict another image: ")
    continue_flag = input()
