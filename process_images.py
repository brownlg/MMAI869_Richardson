# use https://storage.googleapis.com/openimages/web/download.html 
# process the images to create a classifier
# classifier will use 200 x 200 pixel input

#Setup global options

import pandas as pd
import numpy as np
from random import random

import richardson_image_handlers
from richardson_file_handlers import load_data, save_image, get_file_list

DATA_PATH = "[target_dir\\validation]\\" 
META_PATH = "[target_dir\\validation]\\Validation Meta data"
META_FILE = "validation-annotations-bbox.csv"

TRAIN_PATH = 'richardson_images_train_set'
TEST_PATH = 'richardson_images_test_set'
VALIDATION_PATH = 'richardson_images_validation_set'

TRAIN_TEST_VALIDATION_DISTRIBUTION = (70, 10, 20)

IMG_WINDOW_X = 100
IMG_WINDOW_Y = 300

DEFN_FILE = "mySettings.csv"

# get the human labels
human_labels = {
#		"/m/02p0tk3" : "Human body"
   #     "/m/01g317" : "Person"
		"/m/04yx4" : "Man",
		"/m/03bt1vf": "Woman"
	}

#get list of images to load, based on jpg in file directory
#img_list = get_file_list(DATA_PATH)

#get list of images with human labels first
# get the bounding boxes
boxes = load_data(META_FILE, META_PATH)

img_list = []
for label_name in human_labels:
	rows_human = boxes.loc[boxes.LabelName == label_name]
	
	#create a list
	for row in rows_human.ImageID:
		img_list.append(row)


# select rows with ImageID
#img_id = "00a159a661a2f5aa"

for img_id in img_list:
	#img_id = img_list[2]

	#remove extension
	img_id = img_id.split('.')[0]

	select_rows = boxes.loc[boxes.ImageID == img_id]
	if (len(select_rows.index)== 0):
		continue
	#print(select_rows)

	target_rows = richardson_image_handlers.cut_out_target(human_labels, select_rows)

	#print(target_rows)
	img_clipped, flag_success = richardson_image_handlers.create_clipped_images(img_id, DATA_PATH, target_rows, IMG_WINDOW_X, IMG_WINDOW_Y)

	if (flag_success):
		#save to file
		clip_index = 1
		print("Saving image!")
		r = random()

		# split data into train, test, validation
		if (r < TRAIN_TEST_VALIDATION_DISTRIBUTION[0]):
			save_image(str(clip_index) + '_' + str(img_id) + '.jpg', TRAIN_PATH, img_clipped)
		elif (r < (TRAIN_TEST_VALIDATION_DISTRIBUTION[2]+TRAIN_TEST_VALIDATION_DISTRIBUTION[0])):
			save_image(str(clip_index) + '_' + str(img_id) + '.jpg', VALIDATION_PATH, img_clipped)
		else:
			save_image(str(clip_index) + '_' + str(img_id) + '.jpg', TEST_PATH, img_clipped)


