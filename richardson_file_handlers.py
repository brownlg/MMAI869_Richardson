import os
import pandas as pd
import numpy as np
import cv2
import scipy as scipy

import os
import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img


#load image
def load_image(datafile, path = "", flag_bw_load = False):    
    file_path = os.path.join(path, datafile)
    
    #print("loading file " + file_path)

    if (flag_bw_load):        
        try:
            img = load_img(file_path, True)  # this is a PIL image
        except:
            print("error trying to load")
            return None
    else:
        
        try:
            img = load_img(file_path)  # this is a PIL image
          
        except:
            print("error trying to load")
            return None

    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    #print("image shape returned " + str(x.shape))
    return x

#save image
def save_image(filename, path = "", image_data = None, flag_png = False, remove_color = False):
    if (flag_png == False):
        if (remove_color):
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)

        myfile = filename.split('.')[0]
        myfile = myfile + ".jpg"   
        #save_img(os.path.join(path, myfile), image_data)
        cv2.imwrite(os.path.join(path, myfile), image_data)

    else:
        myfile = filename.split('.')[0]
        myfile = myfile + ".png"        
        save_img(os.path.join(path, myfile), image_data)
    return True


#load CSV files
def load_data(datafile, path = ''):
    csv_path = os.path.join(path, datafile)
    #print("loading file " + csv_path)
    return pd.read_csv(csv_path, encoding='latin-1')

#get list of files in a directory
def get_file_list(path = "", ext = "jpg"):

    file_list = []
    for root, dirs, files in os.walk(path):
        file_list = file_list + files

    return file_list

def load_images_for_keras(file_path="", ext = "jpg", max_limit = 15000000, window_x=10, window_y=10, num_channels = 1):
    file_list = []
    
    for root, dirs, files in os.walk(file_path):
        file_list = file_list + files

    number_of_files = len(file_list) 

    # using naive method  
    # to remove None values in list 
    res = [] 
    for val in file_list: 
        if val is not None : 
            res.append(val) 

    file_list = res

    number_of_files = len(file_list) 

    number_of_files = min(number_of_files, max_limit)

    #create numpy array for files
    filenames = np.empty((number_of_files,), dtype = object)
    img_arr = np.empty((number_of_files, window_y, window_x, num_channels), dtype = object)

    cc=0
    for file_name in file_list:
        # open the file add to array
        img = load_image(file_name, file_path, False)  # this is a PIL image
      
        img_arr[cc] = img[0]

        cc = cc + 1
        if (cc >= number_of_files):
            break

    return img_arr, file_list[:number_of_files]

