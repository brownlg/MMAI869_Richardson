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


print("deleting old sample files...")
for filename in os.listdir(os.path.join("simple_gen_images")):
	os.remove(os.path.join("simple_gen_images", filename))


# draw on blank
height = 500
width = 600
img = np.ones((height, width, 3), np.uint8)*255

image_max = 5000

df = pd.DataFrame(data=np.zeros((image_max, 2)), columns = ['Image file', 'Label'])

for i in range(0, image_max):
    obj_type = random.randint(1,3)
    img = np.ones((height, width, 3), np.uint8)*255

    x_size = random.randint(10, int(width * .5))
    y_size = random.randint(10, int(height * .5))
    x = random.randint(0, width - x_size)
    y = random.randint(0, height - y_size)

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    if obj_type == 1:
        img_label = "line"
        cv2.line(img,(0,0),(150, 200),(255,0,0),50)

        #cv2.line(img,(x,y),(x+x_size, y+y_size),(r,g,b),50)
    elif obj_type == 2:
        img_label = "rectangle"
        #cv2.rectangle(img,(x,y),(x+x_size,y+y_size),(r,g,b),15)
        cv2.rectangle(img,(150,60),(200,350),(0,255,0),15)


    elif obj_type == 3:
        img_label = "circle"
        #cv2.circle(img,(x,y), int(x_size/2), (r,g,b), -1)
        cv2.circle(img,(150,120), int(25), (255,255,255), -1)
    
    img_file_name = "my_test_" + str(i) + "_" + img_label + ".jpg"
    df.loc[i, "Image file"] = img_file_name
    df.loc[i, "Label"] = img_label
    file_handler.save_image(img_file_name, PATH_IMG, image_data=img)

print(df)

df.to_csv(os.path.join(richardson_path.INFO_PATH, "shapes_info.csv"),header=False)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

