import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import sklearn as sk


def get_data():
    # define train set
    df_train = tf.keras.preprocessing.image_dataset_from_directory(
        directory='.\\data\\',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(128, 128),
        batch_size=32
    )

    # train validation set
    df_test = tf.keras.preprocessing.image_dataset_from_directory(
        directory='.\\data\\',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(128, 128),
        batch_size=32
    )

    # use autotune to optimize consumption and producing data
    AUTOTUNE = tf.data.AUTOTUNE
    df_train = df_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    df_test = df_test.cache().prefetch(buffer_size=AUTOTUNE)

    return df_train, df_test


def create_model():
    # define number of output classes
    num_classes = len(os.listdir('./data/'))

    # create model
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(128, 128, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy',
                           # tf.keras.metrics.TrueNegatives(),
                           # tf.keras.metrics.FalseNegatives(),
                           # tf.keras.metrics.TruePositives(),
                           # tf.keras.metrics.FalsePositives()
                           ])

    return model


def run(model, df_train, df_test, epochs=10):
    history = model.fit(
        df_train,
        validation_data=df_test,
        epochs=epochs
    )

    return history, model
