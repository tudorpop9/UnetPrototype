import tensorflow as tf
import os
import numpy as np
import random
from tqdm import tqdm
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from cv2 import cv2

# Input images size
# original dimensions/16
IMG_WIDTH = 250
IMG_HEIGHT = 250
IMG_CHANNELS = 3

# Input paths
# DST_PARENT_DIR = './drone_dataset_resized/'
# PARENT_DIR = './semantic_drone_dataset/'
# ORIGINAL_PATH = 'original_images/'
# SEGMENTED_PATH = 'label_images/'
# DST_SEGMENTED_PATH = "label_images_semantic/"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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
#consistent randomness and good-luck charm
seed = 42
np.random.seed = seed
##################################################################### pre-processing data set #######################################################################3
train_ids = os.listdir(PARENT_DIR + ORIGINAL_RESIZED_PATH)

X_train = np.zeros((len(train_ids),IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(train_ids),IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)

def checkVegetation(b, g, r):
    if ((b == 0 and g == 102 and r == 0) or
          (b == 35 and g == 142 and r == 107) or
            (b == 190 and g == 250 and r == 190)):
        return True
    else:
        return False


def split_original():
    ids = os.listdir(PARENT_DIR + ORIGINAL_PATH)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        count = 0

        path = PARENT_DIR + ORIGINAL_PATH + id_
        img_ = cv2.imread(path, cv2.IMREAD_COLOR)

        fragmentShape = 1000, 1000, 3
        fragment = np.zeros(fragmentShape, dtype=np.uint8)
        for offset_i in range(0, img_.shape[0]//1000):
            for offset_j in range(0,img_.shape[0]//1000):

                for i in range(0, img_.shape[0]//6):
                    for j in range(0,img_.shape[1]//6):
                        fragment[i, j] = img_[i + offset_i * 1000, j + offset_j * 1000]

                cv2.imwrite(DST_PARENT_DIR + DST_ORIGINAL_PATH + id_.split('.')[0]+ "_" + str(count) + ".png", fragment)
                count = count + 1

        for offset_i in range(0, img_.shape[0]//1000 - 1):
            for offset_j in range(0, img_.shape[0]//1000 -1):

                for i in range(0, img_.shape[0] // 6):
                    for j in range(0,img_.shape[1] // 6):
                        fragment[i, j] = img_[500 + i + offset_i * 1000, 500 + j + offset_j * 1000] # 500 pt a porni de la 500, nu 0 si a termina la 5500

                cv2.imwrite(DST_PARENT_DIR + DST_ORIGINAL_PATH  + id_.split('.')[0]+ "_" + str(count) + ".png", fragment)
                count = count + 1


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

                cv2.imwrite(DST_PARENT_DIR + DST_SEGMENTED_PATH+ id_.split('.')[0]+ "_" + str(count) + ".png", fragment)
                count = count + 1

        for offset_i in range(0, img_.shape[0] // 1000 - 1):
            for offset_j in range(0, img_.shape[0] // 1000 - 1):

                for i in range(0, img_.shape[0] // 6):
                    for j in range(0, img_.shape[1] // 6):
                        fragment[i, j] = img_[
                            500 + i + offset_i * 1000, 500 + j + offset_j * 1000]  # 500 pt a porni de la 500, nu 0 si a termina la 5500

                cv2.imwrite(DST_PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0]+ "_" + str(count) + ".png", fragment)
                count = count + 1


def resize_original():
    ids = os.listdir(PARENT_DIR + DST_ORIGINAL_PATH)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):

        path = PARENT_DIR + DST_ORIGINAL_PATH + id_
        img_ = cv2.imread(path, cv2.IMREAD_COLOR)

        resized_img = cv2.resize(img_, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imwrite(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + id_, resized_img)

def resize_segmented():
    ids = os.listdir(PARENT_DIR + DST_SEGMENTED_PATH)
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):

        path = PARENT_DIR + DST_SEGMENTED_PATH + id_.split('.')[0] + '.png'
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + id_, resized_img)


# split_original()
# split_segmented()

# resize_segmented()
# resize_original()

##################################################################### pre-processing data set #######################################################################3


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    # print(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + id_)
    img = imread(DST_PARENT_DIR + ORIGINAL_RESIZED_PATH + id_)[:, :, :IMG_CHANNELS]
    X_train[n] = img
    mask = imread(DST_PARENT_DIR + SEGMENTED_RESIZED_PATH + id_)[:, :, :IMG_CHANNELS]
    Y_train[n] = mask


image_we = random.randint(0, len(train_ids))
#original
imshow(X_train[image_we])
plt.show()
#control
imshow(np.squeeze(Y_train[image_we]))
plt.show()

model = tf.keras.models.Sequential()

# Input layer

model.add(tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
#model.add(tf.keras.layers.Lambda(lambda x: x/255))

# convolutional layers 1
model.add(tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

# convolutional layers 2
model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

# convolutional layers 3
model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

# convolutional layers 4
model.add(tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

# convolutional layers 5
model.add(tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))

# de-convolutional layers 1
model.add(tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2, 2), padding='same'))
model.add(tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1))))
#model.add(tf.keras.layers.concatenate())
model.add(tf.keras.layers.Conv2D(128, (3,3),activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(128, (3,3),activation = 'relu', kernel_initializer='he_normal', padding = 'same'))


# de-convolutional layers 2
model.add(tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2, 2), padding='same'))
#model.add(tf.keras.layers.concatenate())
model.add(tf.keras.layers.Conv2D(64, (3,3),activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(64, (3,3),activation = 'relu', kernel_initializer='he_normal', padding = 'same'))

# de-convolutional layers 3
model.add(tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2, 2), padding='same'))
model.add(tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1))))
#model.add(tf.keras.layers.concatenate())
model.add(tf.keras.layers.Conv2D(32, (3,3),activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(32, (3,3),activation = 'relu', kernel_initializer='he_normal', padding = 'same'))

# de-convolutional layers 4
model.add(tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2, 2), padding='same'))
#model.add(tf.keras.layers.concatenate())
model.add(tf.keras.layers.Conv2D(16, (3,3),activation = 'relu', kernel_initializer='he_normal', padding = 'same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same'))

model.add(tf.keras.layers.Conv2D(3, (3,3), activation = 'sigmoid', kernel_initializer='he_normal', padding = 'same'))

model.build()
model.summary()

callbacks =[
   tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
   tf.keras.callbacks.TensorBoard(log_dir='logs')
]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
results = model.fit(X_train, Y_train, validation_split=0.1, shuffle=True, batch_size = 16, epochs = 25, callbacks=callbacks, verbose=1)





#outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(c9)
#
# model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
# model.summary()

#Model checkoints
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose = 1, save_best_only = True )

#Callbacks

idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train, verbose=1)
#preds_val = model.predict(X_train[int(X_train.shape[0])], verbose=1)
# preds_test = model.predict(X_test, verbose = 1)

#Binarizationing the results
preds_train_t = (preds_train > 0.5).astype(np.uint8)
#preds_val_t = (preds_val > 0.1).astype(np.uint8)
# preds_test_t = (preds_test > 0.5).astype(np.uint8)



#random training sample
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


imshow(np.squeeze(preds_train_t[i]))
plt.savefig(predictionPath)
plt.show()





# random validation sample
# i = random.randint(0, len(preds_val_t))
# imshow(X_train[int(X_train.shape[0]*0.9):][i])
# plt.show()
# imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][i]))
# plt.show()
# imshow(np.squeeze(preds_val_t[i]))
# plt.show()

