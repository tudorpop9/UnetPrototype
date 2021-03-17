import datetime
import os
import random
import threading
from threading import Thread
import tifffile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cv2 import cv2
from skimage.io import imread, imshow
from tqdm import tqdm
import DataSetTool

# Input images size
# original dimensions/16
IMG_WIDTH = 250
IMG_HEIGHT = 250
IMG_CHANNELS = 3
SAMPLE_SIZE = 20000
BATCH_SIZE = 2000
# current labels
labels = {
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
SEGMENTED_ONE_HOT_PATH = "Resized_Segmented_One_Hot/"

RESULTS_PATH = "./Results/"
LABEL_TYPES_PATH = "results_on_"

# consistent randomness and good-luck charm
seed = 42
np.random.seed = seed
data_set = DataSetTool.DataSetTool(DST_PARENT_DIR, PARENT_DIR, ORIGINAL_PATH, SEGMENTED_PATH, DST_SEGMENTED_PATH,
                 DST_ORIGINAL_PATH, ORIGINAL_RESIZED_PATH, SEGMENTED_RESIZED_PATH, SEGMENTED_ONE_HOT_PATH,
                 RESULTS_PATH, LABEL_TYPES_PATH)

##################################################################### testing functionality area ############################################################################
data_set.get_input_pipeline()

##################################################################### pre-processing data set #######################################################################3




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
    u6 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(u6)
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
    u7 = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(u7)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(u8)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(u9)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # c9 = tf.keras.layers.Conv2D(6, (1, 1), activation='softmax')(c9)

    outputs = tf.keras.layers.Conv2D(6, (1, 1), activation='softmax')(c9)
    # outputs = tf.keras.layers.Lambda(lambda x: x*255)(outputs)

    adamOptimizer = tf.keras.optimizers.Adam(lr=0.0001)
    # categorical_crossentropy
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=adamOptimizer, loss=dice_coef_loss, metrics=['accuracy'])
    model.summary()

    return model


# test purposes
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



##################################################################### pre-processing data set #######################################################################

# data_set.split_original()
# data_set.split_segmented()
#
# data_set.resize_original()
# # data_set.resize_segmented_building_road()
# data_set.resize_segmented()
#
# data_set.augment_data_set()
# data_set.to_one_hot_and_save()
# exit(3)

# creates the unet
model = create_model()
# model = create_simpler_model()

# waits for a decizion to train, or not to train
print('Enter a number: odd = load weights, or even = train model ')
to_train = input()

# applies one hot encoding on label images
print('One hot encoding labeled images..')
# labels, one_hot_y_train = to_one_hot(N_OF_LABELS, Y_train)


current_day = datetime.datetime.now()
# if flag is an even number we perform a fit operation, training the model and save its best results
if int(to_train) % 2 == 0:
    metric = 'val_accuracy'
    callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs' + '/logs_on_' + str(current_day.month).zfill(2) + str(current_day.day).zfill(2)),
        tf.keras.callbacks.ModelCheckpoint(filepath='./model_for_semantic_segmentation.h5', monitor=metric,
                        verbose=2, save_best_only=True, mode='max')
    ]

    train_generator = data_set.get_generator()
    model.fit(train_generator,
              batch_size=18,
              steps_per_epoch=1941, #(floor(dataset_size / batch_size))
              callbacks=callbacks,
              epochs=20,
              verbose=1)
    model.save_weights('model_for_semantic_segmentation.h5')
    # print('Training is done, do you want to save (overwrite) weights ?[y/n]')
    # save_w_flag = input()
    # if save_w_flag.lower() == 'y':
    #     model.save_weights('model_for_semantic_segmentation.h5')

# otherwise we load the weights from another run
else:
    model.load_weights('model_for_semantic_segmentation.h5')


train_ids = os.listdir(PARENT_DIR + ORIGINAL_RESIZED_PATH)
random_images_idx = random.sample(train_ids, 100)
X_train = np.zeros((100, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
ground_truth = np.zeros((100, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

for n, id_ in tqdm(enumerate(random_images_idx), total=len(random_images_idx)):
    img = imread(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + train_ids[n])[:, :, :IMG_CHANNELS]
    X_train[n] = img

    mask = imread(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + train_ids[n].split('.')[0] + '.png')[:, :,
           :N_OF_LABELS]
    ground_truth[n] = mask


preds_train = model.predict(X_train, verbose=1)

print("Enter 0 to exit, any other number to predict another image: ")
continue_flag = input()

while int(continue_flag) > 0:
    i = random.randint(0, len(preds_train))

    trainPath = "%s%sstrain%03d.png" % (RESULTS_PATH, LABEL_TYPES_PATH, i)
    controlPath = "%s%scontrolMask%03d.png" % (
        RESULTS_PATH, LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(current_day.day).zfill(2) + '/', i)
    predictionPath = "%s%sprediction%03d.png" % (
        RESULTS_PATH, LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(current_day.day).zfill(2) + '/', i)

    today_result_dir = RESULTS_PATH + LABEL_TYPES_PATH + str(current_day.month).zfill(2) + str(current_day.day).zfill(2)
    if not os.path.exists(today_result_dir):
        os.mkdir(today_result_dir)

    imshow(X_train[i])
    plt.savefig(trainPath)
    plt.show()

    imshow(np.squeeze(ground_truth[i]))
    plt.savefig(controlPath)
    plt.show()

    interpreted_prediction = data_set.parse_prediction(preds_train[i], labels)
    imshow(np.squeeze(interpreted_prediction))
    plt.savefig(predictionPath)
    plt.show()

    print("Enter 0 to exit, any positive number to predict another image: ")
    continue_flag = input()
