import tensorflow.keras as keras
import numpy as np
from skimage.io import imread, imshow
# import DataSetTool
# import matplotlib.pyplot as plt

# DST_PARENT_DIR = './Vaihingen/'
# PARENT_DIR = './Vaihingen/'
# ORIGINAL_PATH = 'Originals/'
# SEGMENTED_PATH = 'SegmentedOriginals/'
#
# DST_SEGMENTED_PATH = "Variation_1_Segmented/"
# DST_ORIGINAL_PATH = "Variation_1_Originals/"
#
# ORIGINAL_RESIZED_PATH = "Resized_Originals_Variation_1/"
# SEGMENTED_RESIZED_PATH = "Resized_Segmented_Variation_1/"
# SEGMENTED_ONE_HOT_PATH = "Resized_Segmented_One_Hot/"
#
# RESULTS_PATH = "./Results/"
# LABEL_TYPES_PATH = "results_on_"


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# class used to load and dynamically feed data to the model
class AerialDataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, DST_PARENT_DIR, ORIGINAL_RESIZED_PATH, SEGMENTED_ONE_HOT_PATH, batch_size=32,
                 dim=(1000, 1000), n_channels=3,
                 n_classes=6, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.DST_PARENT_DIR = DST_PARENT_DIR
        self.ORIGINAL_RESIZED_PATH = ORIGINAL_RESIZED_PATH
        self.SEGMENTED_ONE_HOT_PATH = SEGMENTED_ONE_HOT_PATH

        # self.ds_tool = DataSetTool.DataSetTool(DST_PARENT_DIR, PARENT_DIR, ORIGINAL_PATH, SEGMENTED_PATH,
        #                                        DST_SEGMENTED_PATH,
        #                                        DST_ORIGINAL_PATH, ORIGINAL_RESIZED_PATH, SEGMENTED_RESIZED_PATH,
        #                                        SEGMENTED_ONE_HOT_PATH,
        #                                        RESULTS_PATH, LABEL_TYPES_PATH)
        # self.labels = {
        #     0: (255, 255, 255),  # white, paved area/road
        #     1: (0, 255, 255),  # light blue, low vegetation
        #     2: (0, 0, 255),  # blue, buildings
        #     3: (0, 255, 0),  # green, high vegetation
        #     4: (255, 0, 0),  # red, bare earth
        #     5: (255, 255, 0)  # yellow, vehicle/car
        # }

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Initialization

        X_train = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        Y_train = np.zeros((self.batch_size, *self.dim, self.n_classes), dtype=np.uint8)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X_train[i] = imread(self.DST_PARENT_DIR + self.ORIGINAL_RESIZED_PATH + ID)[:, :, :self.n_channels]

            # Store class
            Y_train[i] = imread(self.DST_PARENT_DIR + self.SEGMENTED_ONE_HOT_PATH + ID.split('.')[0] + '.tif')[:, :,
                         :self.n_classes]

            # interpreted_prediction = self.ds_tool.parse_prediction(Y_train[i], self.labels)
            # imshow(np.squeeze(interpreted_prediction))
            # plt.show()
        return X_train, Y_train
