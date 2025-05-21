"""
implementation of a ResNet-34 model architecture
"""

import sys 
assert sys.version_info >= (3, 7)

import tensorflow as tf
from packaging import version 

assert version.parse(tf.__version__) >= version.parse("2.8.0")

from functools import partial

DefaultConv2D  = partial(tf.keras.layers.Conv2D, kernel_size = 3, strides =1 , kernel_initializer = "he_normal", padding = "same")

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides = 1, activation = "relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main = [
            DefaultConv2D(filters, strides = strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]

        self.skipLayer = []
        if strides > 1:
            self.skipLayer = [
              DefaultConv2D(filters, strides = strides, kernel_size = 1),
              tf.keras.layers.BatchNormalization()  
            ]

    def call(self, inputs):
        z = inputs 
        for layer in self.main:
            z = layer(z)
        skipZ = inputs 
        for layer in self.skipLayer:
            skipZ = layer(skipZ)
        return self.activation(z + skipZ)

#creating the model 
model = tf.keras.Sequential([
    DefaultConv2D(64, kernel_size = 7, strides = 2, input_shape = [224, 224, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.Activation("relu"),
    tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = "same")
])
prevFilters = 64

for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    if filters == prevFilters:
        strides = 1
    else:
        strides = 2
    model.add(ResidualUnit(filters, strides = strides))
    prevFilters = filters 

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation= "softmax"))


