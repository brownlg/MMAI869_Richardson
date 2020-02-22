import richardson_file_handlers as file_handler
import richardson_image_handlers as img_handler
import richardson_path as my_paths
import cv2
import os
import json as json
import bbox
from bbox.metrics import jaccard_index_2d

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




#my_model = load_model("my_classifier_soft_max_2.h5") 
my_model = load_model("my_classifier_soft_max_2.h5")

with open(os.path.join('path_1_models', 'annotations', 'via_region_data.json')) as json_file:
    data = json.load(json_file)


def get_bounding_box(meta_data, filename):
    boxes = []

    for key in meta_data.keys():   
        my_row = meta_data.get(key)

        if my_row.get('filename') == filename:            
            for region in my_row.get('regions'):                
                width =  region['shape_attributes']['width']
                height = region['shape_attributes']['height']
                x = region['shape_attributes']['x']
                y = region['shape_attributes']['y']
                boxes.append([x, y, x + width, y + height])            
    return boxes


my_model.summary()

meta_data = data.get('_via_img_metadata')

files = file_handler.get_file_list(my_paths.TTC_PATH)

img_index=0
for file_to_scan in files:
    #get ground truth bounding box from JSON
    true_bounding_boxes = get_bounding_box(meta_data, file_to_scan)

    my_image = file_handler.load_image(file_to_scan, path=my_paths.TTC_PATH, flag_bw_load=False)[0]

    #reate grid z-level 0
    img_arr, list_of_boxes = img_handler.get_grid(my_image, 1, WINDOW_X, WINDOW_Y, 3)
    img_arr = img_arr / 255

    results = my_model.predict(img_arr)
    #probability_true  = my_model.prediction[:,1]
    #print(probability_true)
    print("done")

    # draw the boxes on the image    
    cc = 0 
    for result in results:    
        if (result[1] > 0.50):
            #person found, draw the box
            box = list_of_boxes[cc]
            #draw rectangle
            cv2.rectangle(my_image, (box[0], box[1]), 
                                    (box[2], box[3]), (255, 255, 255), 3)

            # calculate area of overlap
            ious = []
            for true_bounding_box in true_bounding_boxes:
                predict_box = bbox.BBox2D([box[0], box[1], box[2], box[3]])                
                true_box = bbox.BBox2D(true_bounding_box)
                iou = jaccard_index_2d(predict_box, true_box)
                ious.append(iou)
            
        cc = cc + 1
    
    # calculate the average iou
    sum_iou = 0
    for iou in ious:
        sum_iou = sum_iou + iou
    
    if len(ious) > 0:
        iou = sum_iou / len(ious)
    else:
        iou = 0.0

    # draw the true boxes
    for box in true_bounding_boxes:
        cv2.rectangle(my_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    ## Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(my_image, 'Average IoU: ' + str(iou), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
        
    #save image with bounding boxes to output folder       
    file_handler.save_image('obj_detected' + str(img_index) +'.png', path = my_paths.OBJ_TEST_RESULTS, image_data = my_image, flag_png = True, remove_color = False)
    img_index = img_index + 1




