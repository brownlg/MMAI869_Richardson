import os
import pandas as pd
import numpy as np

import os
import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img


#load image
def load_image(datafile, path = ""):
    file_path = os.path.join(path, datafile)
    print("loading file " + file_path)

    img = load_img(file_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    print("image shape returned " + str(x.shape))

    return x

#save image
def save_image(filename, path = "", image_data = None, flag_png = False):
    if (flag_png == False):
        save_img(os.path.join(path, filename), image_data)
    else:
        myfile = filename.split('.')[0]
        myfile = myfile + ".png"        
        save_img(os.path.join(path, myfile), image_data)
    return True


#load CSV files
def load_data(datafile, path = ''):
    csv_path = os.path.join(path, datafile)
    print("loading file " + csv_path)
    return pd.read_csv(csv_path, encoding='latin-1')

#get list of files in a directory
def get_file_list(path = "", ext = "jpg"):

    file_list = []
    for root, dirs, files in os.walk(path):
        file_list = file_list + files

    return file_list

def load_images_for_keras(file_path="", ext = "jpg"):
    file_list = []
    
    for root, dirs, files in os.walk(file_path):
        file_list = file_list + files

    flag_first = True
    for file_name in file_list:
        # open the file add to array
        img = load_image(file_name, file_path)  # this is a PIL image
        if (flag_first):
            img_arr = img
            flag_first = False
        else:
            img_arr = np.append(img_arr, img, axis=0)

    return img_arr


