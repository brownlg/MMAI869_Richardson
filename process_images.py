# use https://storage.googleapis.com/openimages/web/download.html 
# process the images to create a classifier
# classifier will use 200 x 200 pixel input

#Setup global options
import os
import pandas as pd
import numpy as np


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img

DATA_PATH = "[target_dir\\validation]" # data file will be in current director for this assignment
META_PATH = "[target_dir\\validation]\\Validation Meta data"
META_FILE = "validation-annotations-bbox.csv"

DEFN_FILE = "mySettings.csv"

def load_image(datafile, path = DATA_PATH):
    file_path = os.path.join(path, datafile)
    print("loading file " + file_path)

    img = load_img(file_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    return x

 

def load_data(datafile, path = DATA_PATH):
    csv_path = os.path.join(path, datafile)
    print("loading file " + csv_path)
    return pd.read_csv(csv_path, encoding='latin-1')


# load the image
img_id = "00a159a661a2f5aa"
my_img = load_image(img_id + ".jpg")

# get the human labels
human_labels = {
		"/m/02p0tk3" : "human",
        "/m/01g317" : "human"
	}

# get the bounding boxs for human label
boxes = load_data(META_FILE, META_PATH)

# select rows with ImageID
print(boxes.head())
select_rows = boxes.loc[boxes.ImageID == img_id]

my_rows = pd.DataFrame()
print("\n")

# find the rows with human labels
for it in human_labels:
    #print(it + " >> " + select_rows.LabelName)
    human_row = select_rows.loc[select_rows.LabelName == it]
    #print(human_row.head())
    if (len(my_rows.index) == 0):
        my_rows = human_row
    else:
        my_rows.append(human_row)

print("\n")
print(my_rows.head())

# copy the bounding box to new image

index, height, width, color = my_img.shape
print("Image height: " + str(height))
print("Image width: "+  str(width))

for index, row in my_rows.iterrows():
    #img_clipped = pd.DataFrame()
    img_clipped = my_img[0, int(height * row['YMin']) : int(height * row['YMax']), 
                            int(width * row['XMin']) : int(width * row['XMax']), :]

    #save to file
    save_img('test.jpg', img_clipped)
    
