import datetime
import os
import numpy as np
import tensorflow as tf

from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import DataSetTool
import Unet

# IMG_WIDTH = 1000
# IMG_HEIGHT = 1000
weights_file_name = "./semantic_segmentation_all_labels_crossentropy_7mil_090_val_Acc.h5"

IMG_WIDTH = 1000
IMG_HEIGHT = 1000

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

LEARNING_RATE = 0.0001
N_OF_LABELS = len(labels)

gpus = tf.config.experimental.list_physical_devices('GPU')

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
                                   DST_ORIGINAL_PATH, ORIGINAL_RESIZED_PATH, SEGMENTED_RESIZED_PATH,
                                   SEGMENTED_ONE_HOT_PATH,
                                   RESULTS_PATH, LABEL_TYPES_PATH)

unet = Unet.Unet()

##################################################################### pre-processing data set #######################################################################

# used to split the dataset images in 1000x1000
# data_set.split_original()
# data_set.split_segmented()

# used to resize dateset from 1000x1000 to 250x250
# data_set.resize_original()
# data_set.resize_segmented()

# augment dataset and one-hot encodes ground-truth
# data_set.augment_data_set()
# print('Done augmenting dataset')

# class ratio balance, the program will shut down after this function (exit(1))
# data_set.get_data_set_class_balance()


##################################################################### create model #######################################################################
# creates the unet
# model = unet.create_corssentropy_7mil_model(IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, input_channels=IMG_CHANNELS,
#                                             output_channels=N_OF_LABELS, learning_rate=LEARNING_RATE)


model = unet.create_corssentropy_7mil_model(IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, output_channels=N_OF_LABELS,
                                       learning_rate=LEARNING_RATE)
# waits for a decizion to train, or not to train
print('Do you want to train the model first? [y/n]')
to_train = input()

current_day = datetime.datetime.now()

# if flag is an even number we perform a fit operation, training the model and save its best results
if to_train.lower() == 'y':
    if os.path.exists(weights_file_name):
        print('Loading pre-trained weights')
        model.load_weights(weights_file_name)

    metric = 'val_accuracy'
    batch_size = 1

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir='logs' + '/logs_on_' + str(current_day.month).zfill(2) + str(current_day.day).zfill(2)),
        # tf.keras.callbacks.ModelCheckpoint(filepath='./semantic_segmentation_all_labels_crossentropy.h5', monitor=metric,
        #                                    verbose=2, save_best_only=True, mode='max')
        tf.keras.callbacks.ModelCheckpoint(filepath=weights_file_name,
                                           monitor=metric,
                                           verbose=2, save_best_only=True, mode='max')
    ]

    train_generator, validation_generator = data_set.get_generator(validation_split=0.2, batch_size=batch_size)
    try:
        model.fit(train_generator,
                  validation_data=validation_generator,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  epochs=50,
                  verbose=1)
    except ResourceExhaustedError:
        print('\nNot enough resources, decrease batch size or lower gpu usage\n')
        exit(1)

    model.save_weights('./last_epoch_weights_7mil.h5')

# otherwise we load the weights from another run
else:
    if os.path.exists(weights_file_name):
        print('Loading pre-trained weights')
        model.load_weights(weights_file_name)

try:
    # data_set.print_per_class_statistics(validation_split=0.2, model=model)
    # data_set.segment_data(model)
    data_set.get_data_set_class_balance()
    # exit(1)
    # data_set.manual_model_testing(model)
except ResourceExhaustedError:
    print('\nNot enough resources, decrease batch size or lower gpu usage\n')
    exit(1)

