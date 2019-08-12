# pylint: disable=W0511, C0103, E0401
""" Neural networks to encode images into vectors """

from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda


def _layer_norm(x):
    return tf.contrib.layers.layer_norm(x)


def strided_glyph_encoder(embed):
    """Default CNN"""
    #activation_str = hparams.get('glyph_activation', 'leaky_relu')
    #activation = getattr(tf.nn, activation_str)
    #filters = hparams.get('glyph_filters', 64)
    #kernel_size = hparams.get('glyph_kernel_size', 3)
    #output_dim = hparams.get('glyph_embed_dim', 64)
    norm_type = hparams.get('glyph_norm', 'layer')
    conv = partial(Conv2D, strides=(2, 2), activation='leaky_relu)
    layers = [
        conv(64, kernel_size=3), # 64 -> 32
        conv(64, kernel_size=3), # 32 -> 16
        conv(64, kernel_size=3), # 16 -> 8
        conv(64, kernel_size=3), # 8  -> 4
        Flatten(),
        Dense(64)
    ]
    if norm_type == 'layer':
        layers.append(Lambda(_layer_norm))
    else:
        raise ValueError(norm_type)
    return tf.keras.Sequential(layers)

def cnn_glyph_encoder(embed):
    """Slightly more complicated CNN"""
    #activation_str = hparams.get('cnn_activation', ['sigmoid', 'relu'])
    #activation1 = getattr(tf.nn, activation_str[0])
    #activation2 = getattr(tf.nn, activation_str[1])
    #dropout = hparams.get('cnn_dropout', [.3, .5])
    #padding = hparams.get('cnn_padding', ['same', 'valid'])
    #Caution: Do not use padding = valid when using the AE
    #filters = hparams.get('cnn_filters', 32)
    #If you want different window sizes proceed as before
    #kernel_size = hparams.get('cnn_kernel', 3)
    #output_dim = hparams.get('glyph_cnn_embed', 256)

    conv = partial(Conv2D, padding='same', kernel_size=3)
# Do not change padding or any hyperparameters without retraining the CNN
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
