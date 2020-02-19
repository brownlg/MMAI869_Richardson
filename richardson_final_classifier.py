from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


from sklearn import datasets
import numpy as np
import pandas as pd
import os

import cv2

from sklearn.utils import shuffle


#get paths for files
import richardson_path as my_paths
import richardson_file_handlers as file_handler
import richardson_image_handlers as image_handler
from Richardson_Logger import r_logger

import os

#dimensions of input image
WINDOW_X = 200
WINDOW_Y = 200

max_images = 1000
input_shape = (WINDOW_Y, WINDOW_X, 3)

print("Loading dictionary for y_train...")
my_logger = r_logger.R_logger(os.path.join(my_paths.INFO_PATH, "shapes_info.csv"))
my_y_values = my_logger.load_dictionary(key_index = 1, value_index = 2)

images, image_file_names = file_handler.load_images_for_keras("simple_gen_images", "jpg", max_images, WINDOW_X, WINDOW_Y, num_channels=3, scale=True)

shuffle_image, shuffle_image_file_names = shuffle(images, image_file_names, random_state=42)

image_count = len(shuffle_image_file_names)
train_count = int(image_count * 0.8)
test_count = int(image_count * 0.2)

print("Loading images for train...")
x_train = shuffle_image[:train_count]
x_train_files = shuffle_image_file_names[:train_count]
x_train = x_train / 255.0

print("Loading images for test...")
x_test = shuffle_image[train_count:]
x_test_files = shuffle_image_file_names[train_count:]
x_test = x_test / 255.0

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

#image_index = 7777 # You may select anything up to 60,000
#print(y_train[image_index]) # The label is 8
#plt.imshow(x_train[image_index], cmap='Greys')

# Reshaping the array to 4-dims so that it can work with the Keras API
#x_train = x_train.reshape(x_train.shape[0], WINDOW_X, WINDOW_Y, 1)
#x_test = x_test.reshape(x_test.shape[0], WINDOW_X, WINDOW_Y, 1)


print("Get y values...")

# build dataframe
df = pd.DataFrame(data=np.zeros((test_count, 1)), dtype='float64', columns=["Y"])
i=0
for file_name in x_train_files:
    y_value = my_y_values[file_name]    
    #df.loc[i, "Image file"] = file_name
    df.loc[i, "Y"] = y_value
    i=i+1
 # one hot encoding
#y_train = pd.get_dummies(df)
y_train = df.Y.astype("category").cat.codes
print(y_train.describe())

df = pd.DataFrame(data=np.zeros((test_count, 1)), columns=["Y"])
i=0
for file_name in x_test_files:
    y_value = my_y_values[file_name]    
    df.loc[i, "Y"] = y_value    
    i=i+1
# one hot encoding
#y_test = pd.get_dummies(df)
y_test = df.Y.astype("category").cat.codes


print(y_train.head())
print(y_test.head())

#create y_train and y_test values
print('Number of images in y_train', y_train.shape[0])
print('Number of images in y_test', y_test.shape[0])

mnist = tf.keras.datasets.mnist

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.asarray(x_train, dtype=float)
y_train = np.asarray(y_train, dtype=float)
x_test = np.asarray(x_test, dtype=float)
y_test = np.asarray(y_test, dtype=float)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')        
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Create an instance of the model
model = MyModel()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))
