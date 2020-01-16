import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet #ImageNet's model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


# first attempt at an image classifier
from sklearn import datasets
import tensorflow as tf
import numpy as np
import pandas as pd

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

max_images = 1000000

print("Loading dictionary for y_train...")
my_logger = r_logger.R_logger(my_paths.INFO_PATH + '\\' + "data.csv")
my_y_values = my_logger.load_dictionary()

print("Loading images for train...")
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_train_files = file_handler.load_images_for_keras(my_paths.TRAIN_PATH, "png", max_images, WINDOW_X, WINDOW_Y)
x_train = x_train / 255

print("Loading images for test...")
x_test, x_test_files = file_handler.load_im ages_for_keras(my_paths.VALIDATION_PATH, "png",max_images, WINDOW_X, WINDOW_Y)
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
input_shape = (WINDOW_Y, WINDOW_X, 1)


print("Get y values...")
y_train = image_handler.get_y_value(x_train_files, my_y_values, my_paths.human_labels)
y_test = image_handler.get_y_value(x_test_files, my_y_values, my_paths.human_labels)

## ---------- TRANSFER LEARNING  ------------

#Import the MobileNet model with Imagenet's weights and without the output layer

base_model=MobileNet(weights='imagenet',include_top=False, input_shape= input_shape) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output

#Include a few dense layers and an output layer with softmax and two outputs (human head or no human head)

x=GlobalAveragePooling2D()(x)
x=Dense(500,activation='relu')(x) 
x=Dense(500,activation='relu')(x)
x=Dense(500,activation='relu')(x) 
preds=Dense(2,activation='softmax')(x) 

model=Model(inputs=x_train,outputs=y_train)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture

for i,layer in enumerate(model.layers):
  print(i,layer.name)

