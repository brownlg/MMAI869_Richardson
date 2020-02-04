# making an image classifier
# for shapes to learn what is needed
# 

from sklearn import datasets
import tensorflow as tf
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

#dimensions of input image
WINDOW_X = 80
WINDOW_Y = 80

from keras import backend as K
#print(K.tensorflow_backend._get_available_gpus())
import matplotlib.pyplot as plt

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
x_train = x_train / 255

print("Loading images for test...")
x_test = shuffle_image[train_count:]
x_test_files = shuffle_image_file_names[train_count:]
x_test = x_test / 255

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
df = pd.DataFrame(data=np.zeros((test_count, 1)), columns=["Y"])
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


# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()
#model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding = 'same', input_shape=input_shape))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(16, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))  

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy' ])

model.fit(x=x_train, y=y_train, batch_size = 10, epochs = 20)

print(model.evaluate(x_test, y_test))
print(model.metrics_names)

print("Saving model...")
model.save('my_classifier_soft_max_2.h5')

print("deleting old file 1")
if os.path.exists(os.path.join("trained_model_results", "test_results_summary.csv")):
	os.remove(os.path.join("trained_model_results", "test_results_summary.csv"))

print("Predicting results")
results = model.predict(x_test)

print("Saving results")
#np.savetxt(os.path.join("trained_model_results", "test_results_summary.csv"), results, delimiter = ',')
files_txt = r_logger.R_logger(os.path.join("trained_model_results", "test_results_summary.csv"))

result_table = []
for i in range(0, results.shape[0]):
    files_txt.write_line(x_test_files[i] + "," + str(results[i, 0]) + "," + str(results[i, 1]) + "," + str(results[i, 2]) + '\n')
    result_table.append((results[i,0], results[i, 1], x_test_files[i]))
files_txt.close()
