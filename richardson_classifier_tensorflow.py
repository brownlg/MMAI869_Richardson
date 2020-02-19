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

#images, image_file_names = file_handler.load_images_for_keras("simple_gen_images", "jpg", max_images, WINDOW_X, WINDOW_Y, num_channels=3, scale=True)

#Preprocess the images using InceptionV3
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    #we are not using inception here!
    #img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

image_file_names = file_handler.get_file_list("simple_gen_images", "jpg", True)

# Get unique images
encode_train = sorted(set(image_file_names))
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)  # slice the tensor
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

#image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

for img, path in image_dataset:
  #batch_features = image_features_extract_model(img)
  batch_features = img
  batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())


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


# we will cache images so we dont run out of memory
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
#dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

# Use map to load the numpy files in parallel
train_ds = train_ds.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
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


EPOCHS = 20

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for (batch, (img_tensor, target)) in enumerate(train_ds):
      train_step(img_tensor, target)

  #for images, labels in train_ds:
  #  train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

