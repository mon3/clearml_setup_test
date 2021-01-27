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


# task = Task.init(
#     project_name="test_mnist_data_keras_sequential_260121_tf2",
#     task_name="Tensorflow v2 mnist tensorboard enabled",
# )


tf.enable_v2_behavior()
dataset_folder = Dataset.get(
    dataset_id="86b5e2bbec4340318621f6286058960a"
).get_local_copy()

print(type(dataset_folder))

# (ds_train, ds_test), ds_info = tfds.load(
#     "mnist",
#     split=["train", "test"],
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True,
# )

x_train, y_train = loadlocal_mnist(
    images_path="./mnist_local_data/train-images-idx3-ubyte",
    labels_path="./mnist_local_data/train-labels-idx1-ubyte",
)

x_test, y_test = loadlocal_mnist(
    images_path="./mnist_local_data/t10k-images-idx3-ubyte",
    labels_path="./mnist_local_data/t10k-labels-idx1-ubyte",
)

ds_train = (x_train, y_train)
ds_test = (x_test, y_test)
# print("TRAIN SHAPE", y_train.shape[0])

# (ds_train, ds_test) = tf.keras.datasets.mnist.load_data(dataset_folder)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(y_train.shape[0])
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
    callbacks=[tensorboard_callback],
)
