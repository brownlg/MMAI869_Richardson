# use https://storage.googleapis.com/openimages/web/download.html 
# process the images to create a classifier
# classifier will use 200 x 200 pixel input

#Setup global options

import pandas as pd
import numpy as np
from random import random

import richardson_image_handlers as image_handler
from richardson_file_handlers import load_data, save_image, get_file_list, load_image
import Richardson_Logger

import richardson_path
import os


import json

TRAIN_TEST_VALIDATION_DISTRIBUTION = (0.8, 0.0, 0.2)

IMG_WINDOW_X = 128
IMG_WINDOW_Y = 128

COLLECT_MAX = 1000   #select how many images you want total

DEFN_FILE = "mySettings.csv"

# create & clean-up output directories
print("deleting old json annotation file")
if os.path.exists(os.path.join(richardson_path.ATT_PATH , richardson_path.ATT_ANNOTATION_PATH, richardson_path.ATT_TRAIN_FILE)):
	os.remove(os.path.join(richardson_path.ATT_PATH , richardson_path.ATT_ANNOTATION_PATH, richardson_path.ATT_TRAIN_FILE))

# get list of images to load, based on jpg in file directory
# get list of images with human labels first
# get the bounding boxes
boxes = load_data(richardson_path.META_FILE, richardson_path.META_PATH)
class_definitions = load_data(richardson_path.CLASS_FILE, richardson_path.META_PATH)

print(boxes.head())

img_list = []
for label_name in richardson_path.human_labels:
	rows_human = boxes.loc[boxes.LabelName == label_name]
	
	#create a list
	for row in rows_human.ImageID:
		if row not in img_list:  # make sure you only have unique entries!
			img_list.append(row)

unique_number = -1 
for image in img_list:
    #load the image to get dimensions
    img_nparray = load_image(image + ".jpg", richardson_path.DATA_PATH, False)
    width = img_nparray.shape[1]
    height = img_nparray.shape[2]

    unique_number = unique_number + 1
    # get the boxes for this image    
    caption = image_handler.create_caption(image, boxes, class_definitions)

    # save the caption in the format required

    # save as JSON:
    x1 =  '{ "image_id": '+ image + ', "id": '+ str(unique_number) +', "caption": "' + caption + '" }'
    x2 = """{ "license": 1, "file_name": " """ + image + """.jpg", "height": """ + str(height) + """, "width": """ + str(width) + """, "date_captured": "2020-01-31 01:02:03", "id": """ + image +  """ }"""

    # append to a file    
    with open(os.path.join(richardson_path.ATT_PATH , richardson_path.ATT_ANNOTATION_PATH) + richardson_path.ATT_TRAIN_FILE, 'a') as outfile:
        json.dump(x1 + x2, outfile)
    