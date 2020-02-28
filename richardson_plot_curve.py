# credit :https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/plot_precision_recall.html
# using this person code to plot precision recall curve

import richardson_file_handlers as file_handler
import richardson_image_handlers as image_handler

import richardson_path as my_paths
import cv2
import os
import json as json
from Richardson_Logger import r_logger


import random
import pylab as pl
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


#dimensions of input image
WINDOW_X = 80
WINDOW_Y = 80

# load test data
print("Loading images for test...")
x_test, x_test_files = file_handler.load_images_for_keras(my_paths.VALIDATION_PATH, "png", 100000, WINDOW_X, WINDOW_Y, num_channels=3)
x_test = x_test / 255

print("Loading dictionary for y_train...")
my_logger = r_logger.R_logger(os.path.join(my_paths.INFO_PATH, "data.csv"))
my_y_values = my_logger.load_dictionary()

print("Get y values...")
y_test = image_handler.get_y_value(x_test_files, my_y_values, my_paths.human_labels)

#create y_train and y_test values
print('Number of images in x_test', x_test.shape[0])
print('Number of images in y_test', y_test.shape[0])

# load the trained neural network
from keras.models import load_model

model_name = "path_1_richardson_V3.h5"
my_model = load_model(model_name) 
print("Completed loading model")

# Run classifier
results = my_model.predict(x_test)
print(model.evaluate(x_test, y_test))
print(model.metrics_names)

probas_ = [ 0 ]

# Compute Precision-Recall and plot curve
precision, recall, thresholds = precision_recall_curve(y[half:], probas_[:, 1])
area = auc(recall, precision)
print ("Area Under Curve: %0.2f" % area)


pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall example: AUC=%0.2f' % area)
pl.legend(loc="lower left")
pl.show()




