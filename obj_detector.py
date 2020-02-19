import richardson_file_handlers as file_handler
import richardson_image_handlers as img_handler
import cv2
import os

#dimensions of input image
WINDOW_X = 80
WINDOW_Y = 80

# load the trained neural network
from keras.models import load_model
#my_model = load_model("my_classifier_soft_max_2.h5") # works on faces, 1st one that actually seemed to work
#my_model = load_model("my_classifier_soft_max_2.h5") 
my_model = load_model("my_classifier_soft_max_2.h5") 

# open the image scene
#file_to_scan = "0000ec18c34241ad.jpg"
#file_to_scan = "00a06e610f2d6fc2.jpg"
#file_to_scan = "00cdf56c63191fd3.jpg"  # beach 
#file_to_scan = "0a1aee5d7701ce5c_1.jpg"  # 
#file_to_scan = "0a1aee5d7701ce5c_2F.jpg"  #
file_to_scan = "078ef1cbf61fa9fa.jpg"

my_image = file_handler.load_image(file_to_scan, "",True)[0]

#reate grid z-level 0
img_arr, list_of_boxes = img_handler.get_grid(my_image, 1, WINDOW_X, WINDOW_Y, 1)
img_arr = img_arr / 255

results = my_model.predict(img_arr)
#probability_true  = my_model.prediction[:,1]
#print(probability_true)
print("done")

# draw the boxes on the image
cc=0
for result in results:    
    if (result[0] < 0.5):
        #person found, draw the box
        box = list_of_boxes[cc]
        #draw rectangle
        cv2.rectangle(my_image, (box[0], box[1]), 
                                (box[2], box[3]), (255, 255, 255), 2)
    cc = cc + 1

cv2.imwrite(os.path.join("", "obj_detected2.jpg"), my_image)








