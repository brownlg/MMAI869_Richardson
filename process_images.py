# use https://storage.googleapis.com/openimages/web/download.html 
# process the images to create a classifier
# classifier will use 200 x 200 pixel input

#Setup global options
import pandas as pd
import numpy as np
from random import random

import richardson_image_handlers
from richardson_file_handlers import load_data, save_image, get_file_list
import Richardson_Logger

import richardson_path
import os

TRAIN_TEST_VALIDATION_DISTRIBUTION = (0.8, 0.0, 0.2)

IMG_WINDOW_X = 80
IMG_WINDOW_Y = 80

COLLECT_MAX = 10000  #select how many images you want total

#get list of images to load, based on jpg in file directory
#img_list = get_file_list(DATA_PATH)

#get list of images with human labels first

DEFN_FILE = "mySettings.csv"

# delete old files
print("deleting csv file")
if os.path.exists(os.path.join("richardson_info_files", "data.csv")):
	os.remove(os.path.join("richardson_info_files", "data.csv"))

print("deleting old sample files 1/3")
for filename in os.listdir(os.path.join("richardson_images_train_set")):
	os.remove(os.path.join("richardson_images_train_set", filename))
print("deleting old sample files 2/3")
for filename in os.listdir(os.path.join("richardson_images_validation_set")):
	os.remove(os.path.join("richardson_images_validation_set", filename))
print("deleting old sample files 3/3")
for filename in os.listdir(os.path.join("richardson_images_test_set")):
	os.remove(os.path.join("richardson_images_test_set", filename))

# get list of images to load, based on jpg in file directory
# get list of images with human labels first
# get the bounding boxes
boxes = load_data(richardson_path.META_FILE, richardson_path.META_PATH)

img_list = []
for label_name in richardson_path.human_labels:
	rows_human = boxes.loc[boxes.LabelName == label_name]
	
	#create a list
	for row in rows_human.ImageID:
		if row not in img_list:  # make sure you only have unique entries!
			img_list.append(row)


# select rows with ImageID
#img_id = "00a159a661a2f5aa"

from Richardson_Logger import r_logger
my_logger = r_logger.R_logger(richardson_path.INFO_PATH + '/' + "data.csv")
my_logger.clear()
my_logger.write_line("tvt,flag_person,imgid\n")


total_collection = 0  # keep track of how many images are in your dataset
for img_id in img_list:
	#img_id = img_list[2]

	#remove extension
	img_id = img_id.split('.')[0]

	select_rows = boxes.loc[boxes.ImageID == img_id]
	if (len(select_rows.index)== 0):
		continue
	#print(select_rows)

	target_rows = richardson_image_handlers.cut_out_target(richardson_path.human_labels, select_rows)

	#print(target_rows)
	clipped_images = richardson_image_handlers.create_clipped_images(img_id, richardson_path.DATA_PATH, target_rows, IMG_WINDOW_X, IMG_WINDOW_Y)

	flag_TTC = False
	if flag_TTC is True:
		ttc_background_images = richardson_image_handlers.create_clipped_images("Photo from Luke(2) - use for TTC background", "", None, IMG_WINDOW_X, IMG_WINDOW_Y)

	if (clipped_images == None):
		continue

	clip_index = 0
	r = random()
	# split data into train, test, validation
	if (r < TRAIN_TEST_VALIDATION_DISTRIBUTION[0]):
		flag_data_for = richardson_path.TRAIN_PATH
	elif (r < (TRAIN_TEST_VALIDATION_DISTRIBUTION[2]+TRAIN_TEST_VALIDATION_DISTRIBUTION[0])):
		flag_data_for =  richardson_path.VALIDATION_PATH
	else:
		flag_data_for = richardson_path.TEST_PATH

	if "use for TTC background" in img_id:  # put the TTC background into training
		flag_data_for = richardson_path.TRAIN_PATH

	for clip_dict in clipped_images:	
		key = list(clip_dict)[0]
		img_clipped = clip_dict[key]

		#save to file
		clip_index = clip_index + 1
		#print("Saving image!")
		
		if (key == 'non-target'):
			identify_non_target = "F"
		else:
			identify_non_target = ""
				
		clipfilename = str(img_id) + '_' +  str(clip_index) + identify_non_target + '.png'
		
		# store data
		save_image(clipfilename, flag_data_for, img_clipped, True, False)  # needs to be png? or B&W 

		my_logger.write_line(flag_data_for + "," + key + "," + clipfilename + "\n")	
		
	# track how many images you have
	total_collection = total_collection + 1

	if total_collection % 1000 == 0:
		print("processed another 1000 files...")

	if (total_collection > COLLECT_MAX):
		break

