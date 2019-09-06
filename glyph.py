""" GLYPH CNNS """

from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda


def _layer_norm(x):
    return tf.contrib.layers.layer_norm(x)

def strided_glyph_encoder(embed):
    """Strided CNN"""
    conv = partial(Conv2D, strides=(2, 2), activation='leaky_relu)
    layers = [
        conv(64, kernel_size=3), # 64 -> 32
        conv(64, kernel_size=3), # 32 -> 16
        conv(64, kernel_size=3), # 16 -> 8
        conv(64, kernel_size=3), # 8  -> 4
        Flatten(),
        Dense(64),
        Lambda(_layer_norm)
    ]

    return tf.keras.Sequential(layers)

def cnn_glyph_encoder(embed):
    """GLYNN CNN"""

    conv = partial(Conv2D, padding='same', kernel_size=3)
    drop = partial(Dropout)
    batchnorm = partial(BatchNormalization, trainable=True)
    layers = [
        conv(32, strides=(2, 2), activation='sigmoid'),
        batchnorm(),
        MaxPooling2D((2, 2)),
        drop(.3),
        conv(32, strides=(1, 1), activation='relu'),
        batchnorm(),
        MaxPooling2D((2, 2)),
        drop(.5),
        Flatten(),
        batchnorm(),
        Dense(256)
    ]
    return tf.keras.Sequential(layers)
