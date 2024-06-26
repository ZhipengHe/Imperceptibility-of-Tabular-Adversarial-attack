import tensorflow as tf
import numpy as np

import os

def get_models(X_train, y_train):
    nn = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(24,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(12,activation='relu'),
                tf.keras.layers.Dense(2),
                tf.keras.layers.Activation(tf.nn.softmax),
            ]
        )
    nn.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    nn.fit(X_train, y_train, batch_size=64, epochs=20, shuffle=True)

    return nn