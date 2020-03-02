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
import numpy as npa
import pandas as pd

#get paths for files
import richardson_path as my_paths
import richardson_file_handlers as file_handler
import richardson_image_handlers as image_handler
from Richardson_Logger import r_logger

from keras import backend as K
#print(K.tensorflow_backend._get_available_gpus())
import matplotlib.pyplot as plt


#dimensions of input image
WINDOW_X = 128
WINDOW_Y = 128

max_images = 5000

print("Loading dictionary for y_train...")
my_logger = r_logger.R_logger(my_paths.INFO_PATH + '\\' + "data.csv")
my_y_values = my_logger.load_dictionary()

print("Loading images for train...")
x_train, x_train_files = file_handler.load_images_for_keras(my_paths.TRAIN_PATH, "png", max_images, WINDOW_X, WINDOW_Y, num_channels=3,scale = True)
x_train = x_train / 255

print("Loading images for test...")
x_test, x_test_files = file_handler.load_images_for_keras(my_paths.VALIDATION_PATH, "png",max_images, WINDOW_X, WINDOW_Y, num_channels=3, scale = True)
x_test = x_test / 255


print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

print("Get y values...")
y_train = image_handler.get_y_value(x_train_files, my_y_values, my_paths.human_labels)
y_test = image_handler.get_y_value(x_test_files, my_y_values, my_paths.human_labels)

# ## Transfer Learning

# `input_shape` is required to be squared and must have any of these sizes: [128, 160, 192, 224]
# We need to reshape the images.
# https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
# 
# 
input_shape = (WINDOW_Y, WINDOW_X, 3)


#Import the with Imagenet's weights and without the output layer
base_model=MobileNet(weights='imagenet',include_top=False, input_shape= input_shape) 


x=base_model.output


#Include a few dense layers and an output layer with softmax and two outputs (human head or no human head)
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
x=Dense(1024,activation='relu')(x)
preds=Dense(2,activation='softmax')(x) 


model=Model(inputs=base_model.input,outputs=preds)
model.summary()

len(model.layers)

#Make the first 85 layers non-trainable and the rest trainable
for layer in model.layers[:85]:
    layer.trainable=False
for layer in model.layers[85:]:
    layer.trainable=True

# ImageDataGenerators are inbuilt in keras and help us to train our model. We just have to specify the path to our training data and it automatically sends the data for training, in batches. It makes the code much simpler.
'''train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('path-to-the-main-data-folder',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)'''


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy' ])


model.fit(x=x_train, y=y_train, batch_size = 100, epochs = 3, validation_split= 0.2)


print(model.evaluate(x_test, y_test))
print(model.metrics_names)

print("Saving model")
model.save('stream2_lb_classifier_Y1.h5')


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


