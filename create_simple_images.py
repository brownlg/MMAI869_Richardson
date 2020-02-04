# Create simple image for training
import cv2
import random
import numpy as np
import pandas as pd
import richardson_file_handlers as file_handler
import richardson_path
import os

PATH_IMG = "simple_gen_images"

def create_point():
    return 0

def create_line():
    return 0

# draw on blank
height = 500
width = 600
img = np.ones((height, width, 3), np.uint8)*255

image_max = 500

df = pd.DataFrame(data=np.zeros((image_max, 2)), columns = ['Image file', 'Label'])

for i in range(0, image_max):
    r = random.randint(1,3)
    img = np.ones((height, width, 3), np.uint8)*255

    if r == 1:
        img_label = "line"
        cv2.line(img,(0,0),(200,300),(0,0,255),50)
    elif r == 2:
        img_label = "rectangle"
        cv2.rectangle(img,(500,250),(250,100),(0,0,255),15)
    elif r == 3:
        img_label = "circle"
        cv2.circle(img,(200,63), 63, (0,255,0), -1)
    
    img_file_name = "my_test_" + str(i) + "_" + img_label + ".jpg"
    df.loc[i, "Image file"] = img_file_name
    df.loc[i, "Label"] = img_label
    file_handler.save_image(img_file_name, PATH_IMG, image_data=img)

print(df)

df.to_csv(os.path.join(richardson_path.INFO_PATH, "shapes_info.csv"),header=False)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

