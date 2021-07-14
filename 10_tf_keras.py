import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#  KERAS SEQUENTIAL MODEL


# model = keras.models.Sequential()
# model.add(keras.layers.Dense(units=13, input_dim=7, activation='relu'))
# model.add(keras.layers.Dense(units=7, activation='relu'))
# model.add(keras.layers.Dense(units=1, activation='sigmoid'))


# model = keras.models.Sequential([
#     keras.layers.Dense(units=13, input_dim=7, activation='relu'),
#     keras.layers.Dense(units=7, activation='relu'),
#     keras.layers.Dense(units=1, activation='sigmoid')
# ])


#  KERAS FUNCTIONAL MODEL


# input_ = keras.Input(shape=(2, ))
# layer1 = keras.layers.Dense(units=10, activation='relu')(input_)
# layer2 = keras.layers.Dense(units=10, activation='relu')(layer1)
# output_ = keras.layers.Dense(units=1, activation='sigmoid')(layer2)
# model = keras.Model(input_, output_)
# model.compile(
#     loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']
# )


#  MNIST DIGIT CLASSIFICATION USING TENSORFLOW  ###########################


mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#  normalizing data
x_train, x_test = x_train / 255., x_test / 255.

#  normalizing data casting it into an EagerTensor, then use tf.keras not keras
# x_train, x_test = tf.cast(x_train / 255., tf.float32), tf.cast(x_test / 255., tf.float32)
# y_train, y_test = tf.cast(y_train, tf.int64), tf.cast(y_test, tf.int64)


#  vanilla dnn
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(256, activation="relu"))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10)
# model.fit(x_train, y_train, steps_per_epoch=32, epochs=10)
model.evaluate(x_test, y_test)






































































































































































































































































