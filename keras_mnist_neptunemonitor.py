from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import datetime
import os
from tempfile import gettempdir

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from clearml import Dataset, Task
from mlxtend.data import loadlocal_mnist

import neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor

PARAMS = {"batch_size": 128, "epochs": 6, "learning_rate": 0.001}

neptune.init(
    project_qualified_name="mon3/sandbox",
    api_token="",
)

# Create experiment
neptune.create_experiment("tensorflow_mnist_keras_JB", params=PARAMS)


tf.enable_v2_behavior()


(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# TODO: check with tensorboard integration
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(PARAMS["batch_size"])
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(PARAMS["batch_size"])
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(PARAMS["batch_size"], activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(PARAMS["learning_rate"]),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


tf.keras.utils.plot_model(
    model,
    to_file="neptune_mnist_model_JB.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)

model.fit(
    ds_train,
    epochs=PARAMS["epochs"],
    validation_data=ds_test,
    callbacks=[NeptuneMonitor()],
)


model.save("neptune_models/neptune_mnist_model_JB")

neptune.log_artifact("neptune_models/neptune_mnist_model_JB")
