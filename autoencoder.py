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

def glyph_decoder(hparams):
    """The decoder part of the CNN"""
    activation_str = hparams.get('cnn_activation', ['sigmoid', 'relu'])
    activation1 = getattr(tf.nn, activation_str[0])
    activation2 = getattr(tf.nn, activation_str[1])
    dropout = hparams.get('cnn_dropout', [.3, .5])
    padding = hparams.get('cnn_padding', 'same')
    #Caution: Do not use padding = valid when using the AE
    filters = hparams.get('cnn_filters', 32)
   # If you want different window sizes proceed as before
    kernel_size = hparams.get('cnn_kernel', 3)
    upsample = UpSampling2D(size=(2, 2))
    drop = partial(Dropout)
    batchnorm = partial(BatchNormalization, trainable=True)
    conv = partial(Conv2D, padding=padding)


   # inputs=tf.keras.Input(shape=(64,64,1))
    layers = [
        Dense(8192),
        batchnorm(),
        Reshape((16, 16, 32)),
        drop(dropout[1]),
        upsample,
        batchnorm(),
        conv(filters, kernel_size, strides=(1, 1),
             activation=activation2),
        drop(dropout[0]),
        upsample,
        batchnorm(),
        conv(1, kernel_size, strides=(1, 1), activation=activation1)
    ]
    return layers

model = cnn_glyph_encoder(hparams=glyph_crf_arijit())
for layer in glyph_decoder(hparams=glyph_crf_arijit()):
    model.add(layer)

path = os.environ["SCALE_DIR"]
#image_path = os.path.join(path, 'data/glyph/latest/images.npy')
image_path = os.path.join(path, 'scale19/new_image_array.npy')
img_data = np.load(image_path)
img_data = img_data.reshape(-1, 64, 64, 1)
img_data /= 255.0
train_X, valid_X, train_ground, valid_ground = train_test_split(img_data, img_data, test_size=0.2, random_state=13)
model.compile(loss='mean_squared_error', optimizer=RMSprop())
model_path = os.path.join(path, 'scale19/models/Autoencoder/autoencoder.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_path,
                                                 save_weights_only=True, verbose=40, period=200)
model.fit(train_X, train_ground, batch_size=256, epochs=200,
          verbose=1, validation_data=(valid_X, valid_ground),
          callbacks=[cp_callback])
print(model.summary())
