# pylint: disable=W0611, C0103, C0413, E0401, C0301
"""Working Autoencoder to pretrain the CNN"""
from __future__ import division
import os
from functools import partial
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.keras as keras
from tensorflow.keras import losses
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import UpSampling2D, Reshape, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split

path = os.environ["SCALE_DIR"]
#image_path = os.path.join(path, 'data/glyph/latest/images.npy')
image_path = os.path.join(path, 'scale19/new_image_array.npy')
img_data = np.load(image_path)
img_data = img_data.reshape(-1, 64, 64, 1)
img_data /= 255.0
train_X, valid_X, train_ground, valid_ground = train_test_split(img_data, img_data, test_size=0.2, random_state=13)

model = Sequential()
model.add(Conv2D(32, 3, activation='sigmoid', padding='same', input_shape=(64,64,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(.2))
model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(.2))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(768))
#decoder
model.add(Dense(8192)) #this is the default size
model.add(BatchNormalization())
model.add(Reshape((16, 16, 32)))
model.add(Dropout(.2))
model.add(UpSampling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, 3, strides=(1,1), activation='relu', padding='same'))
model.add(Dropout(.2))
model.add(UpSampling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(1, (3,3), strides=(1,1), activation='sigmoid', padding='same'))

model.compile(loss='mean_squared_error', optimizer=RMSprop())
model_path = os.path.join(path, 'scale19/models/Autoencoder/autoencoder.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                 save_weights_only=True, verbose=40, period=200)
model.fit(train_X, train_ground, batch_size=256, epochs=200,
          verbose=1, validation_data=(valid_X, valid_ground),
          callbacks=[cp_callback])
print(model.summary())
