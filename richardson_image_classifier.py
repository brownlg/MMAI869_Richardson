# first attempt at an image classifier
from sklearn import datasets
import tensorflow as tf
import numpy as np
import pandas as pd
import os
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

max_images = 20000
input_shape = (WINDOW_Y, WINDOW_X, 3)

print("Loading dictionary for y_train...")
my_logger = r_logger.R_logger(os.path.join(my_paths.INFO_PATH, "data.csv"))
my_y_values = my_logger.load_dictionary()

print("Loading images for train...")
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_train_files = file_handler.load_images_for_keras(my_paths.TRAIN_PATH, "png", max_images, WINDOW_X, WINDOW_Y, num_channels=3)
x_train = x_train / 255

print("Loading images for test...")
x_test, x_test_files = file_handler.load_images_for_keras(my_paths.VALIDATION_PATH, "png",max_images, WINDOW_X, WINDOW_Y, num_channels=3)
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
y_train = image_handler.get_y_value(x_train_files, my_y_values, my_paths.human_labels)
y_test = image_handler.get_y_value(x_test_files, my_y_values, my_paths.human_labels)

#create y_train and y_test values
print('Number of images in y_train', y_train.shape[0])
print('Number of images in y_test', y_test.shape[0])


# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()
#model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding = 'same', input_shape=input_shape))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))  

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy' ])

model.fit(x=x_train, y=y_train, batch_size = 100, epochs = 20)

#print(model.evaluate(x_test, y_test))
#print(model.metrics_names)

print("Saving model...")
model.save('path_1_richardson_V1.h5')

print("deleting old file 1")
if os.path.exists(os.path.join("trained_model_results", "test_results_summary.csv")):
	os.remove(os.path.join("trained_model_results", "test_results_summary.csv"))

print("deleting old result files")
PATH_CORRECT = "prediction_is_correct"
for filename in os.listdir(os.path.join("trained_model_results" , PATH_CORRECT)):
	os.remove(os.path.join("trained_model_results", PATH_CORRECT, filename))

print("Predicting results")
results = model.predict(x_test)

print("Saving results")
np.savetxt(os.path.join("trained_model_results", "test_results_summary.csv"), results, delimiter = ',')
files_txt = r_logger.R_logger(os.path.join("trained_model_results", "test_results_summary.csv"))

result_table = []
for i in range(0, results.shape[0]):
    files_txt.write_line(x_test_files[i] + "," + str(results[i, 0]) + "," + str(results[i, 1]) + '\n')
    result_table.append((results[i,0], results[i, 1], x_test_files[i]))
files_txt.close()

print("Copying clips to folders for easy breezy viewing")
import shutil 
import operator

result_table.sort(key = operator.itemgetter(0))

for iter in result_table:
    file_name = iter[2]
    prob_correct = str(int(round(iter[0]*100, 0))).zfill(3)
    shutil.copy(os.path.join(my_paths.VALIDATION_PATH, file_name), os.path.join("trained_model_results", PATH_CORRECT, prob_correct + "_" + file_name))
