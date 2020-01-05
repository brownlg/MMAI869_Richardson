import pandas as pd
import numpy as np
import cv2

height = 500
width = 200

import richardson_file_handlers as file_handler

my_iamge1 = np.zeros((height,width,4), np.uint8)
my_iamge2 = np.zeros((height,width,4), np.uint8)

cv2.rectangle(my_iamge1,(int(width * .4) ,0),(int(width*.6), height), (255, 0, 0, 255), -1)
cv2.rectangle(my_iamge2,(0,int(height * .4)),(width, int(height*.6)), (0, 0, 255, 255), -1)

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(my_iamge2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask) 

# Now black-out the area of logo in ROI
my_iamge1 = cv2.bitwise_and(my_iamge1, my_iamge1,mask = mask_inv)

# Take only region of logo from logo image.
my_iamge2 = cv2.bitwise_and(my_iamge2, my_iamge2,mask = mask)

file_handler.save_image("background-1.png", "", my_iamge1, True)
file_handler.save_image("background-2.png", "", my_iamge2, True)

dst = cv2.add(my_iamge1,my_iamge2)
file_handler.save_image("background-3.png", "", dst, True)