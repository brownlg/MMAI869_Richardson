# use https://storage.googleapis.com/openimages/web/download.html 
# process the images to create a classifier
# classifier will use 200 x 200 pixel input

#Setup global options

import pandas as pd
import numpy as np

import richardson_image_handlers
from richardson_file_handlers import load_data, save_image

DATA_PATH = "[target_dir\\validation]\\" # data file will be in current director for this assignment
META_PATH = "[target_dir\\validation]\\Validation Meta data"
META_FILE = "validation-annotations-bbox.csv"

TRAIN_PATH = 'richardson_images_train_set'
TEST_PATH = ''

IMG_WINDOW_X = 100
IMG_WINDOW_Y = 300

DEFN_FILE = "mySettings.csv"

# get the human labels
human_labels = {
		"/m/02p0tk3" : "human",
        "/m/01g317" : "human"
	}

#get list of images to load


# select rows with ImageID
img_id = "00a159a661a2f5aa"

# get the bounding boxs for human label
boxes = load_data(META_FILE, META_PATH)

print(boxes.head())
select_rows = boxes.loc[boxes.ImageID == img_id]

target_rows = richardson_image_handlers.cut_out_target(human_labels, select_rows)
img_clipped = richardson_image_handlers.create_clipped_images(img_id, DATA_PATH, target_rows, IMG_WINDOW_X, IMG_WINDOW_Y)

#save to file
clip_index = 1
save_image(str(clip_index) + '_' + str(img_id) + '.jpg', TRAIN_PATH, img_clipped)

