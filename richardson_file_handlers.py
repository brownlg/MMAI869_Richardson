import os
import pandas as pd

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
def save_image(filename, path = "", image_data = None):
    save_img(os.path.join(path, filename), image_data)
    return True


#load CSV files
def load_data(datafile, path = ''):
    csv_path = os.path.join(path, datafile)
    print("loading file " + csv_path)
    return pd.read_csv(csv_path, encoding='latin-1')


