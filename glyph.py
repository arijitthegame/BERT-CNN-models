# Copyright 2019 Johns Hopkins University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


def strided_glyph_encoder(hparams):
    """Default CNN"""
    activation_str = hparams.get('glyph_activation', 'leaky_relu')
    activation = getattr(tf.nn, activation_str)
    filters = hparams.get('glyph_filters', 64)
    kernel_size = hparams.get('glyph_kernel_size', 3)
    output_dim = hparams.get('glyph_embed_dim', 64)
    norm_type = hparams.get('glyph_norm', 'layer')
    conv = partial(Conv2D, strides=(2, 2), activation=activation)
    layers = [
        conv(filters, kernel_size), # 64 -> 32
        conv(filters, kernel_size), # 32 -> 16
        conv(filters, kernel_size), # 16 -> 8
        conv(filters, kernel_size), # 8  -> 4
        Flatten(),
        Dense(output_dim)
    ]
    if norm_type == 'layer':
        layers.append(Lambda(_layer_norm))
    else:
        raise ValueError(norm_type)
    return tf.keras.Sequential(layers)

def cnn_glyph_encoder(hparams):
    """Slightly more complicated CNN"""
    activation_str = hparams.get('cnn_activation', ['sigmoid', 'relu'])
    activation1 = getattr(tf.nn, activation_str[0])
    activation2 = getattr(tf.nn, activation_str[1])
    dropout = hparams.get('cnn_dropout', [.3, .5])
    padding = hparams.get('cnn_padding', ['same', 'valid'])
    #Caution: Do not use padding = valid when using the AE
    filters = hparams.get('cnn_filters', 32)
    #If you want different window sizes proceed as before
    kernel_size = hparams.get('cnn_kernel', 3)
    output_dim = hparams.get('glyph_cnn_embed', 256)

    conv = partial(Conv2D, padding=padding[0])
# Do not change padding or any hyperparameters without retraining the CNN
    drop = partial(Dropout)
    batchnorm = partial(BatchNormalization, trainable=True)
    layers = [
        conv(filters, kernel_size,
             strides=(2, 2), activation=activation1),
        batchnorm(),
        MaxPooling2D((2, 2)),
        drop(dropout[0]),
        conv(filters, kernel_size,
             strides=(1, 1), activation=activation2),
        batchnorm(),
        MaxPooling2D((2, 2)),
        drop(dropout[1]),
        Flatten(),
        batchnorm(),
        Dense(output_dim)
    ]
    return tf.keras.Sequential(layers)
