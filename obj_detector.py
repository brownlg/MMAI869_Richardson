import richardson_file_handlers as file_handler
import richardson_image_handlers as img_handler
import richardson_path as my_paths
import cv2
import os
import json as json
import bbox
from bbox.metrics import jaccard_index_2d
import numpy as np

from Richardson_Logger import r_logger

#dimensions of input image
WINDOW_X = 80
WINDOW_Y = 80

detector_threshold = 0.9999

# load the trained neural network
from keras.models import load_model

model_name = "path_1_richardson_V2.h5"

my_model = load_model(model_name) 
#my_model = load_model("stream2_lb_classifier.h5") 
print("Completed loading model")

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

def add_arr(target, source, index):
    for it in source:
        target[index] = it
        index = index + 1    
    return target, index

def add_list(target, source):
    # using naive method to concat 
    for it in source : 
        target.append(it) 
    return target

def process_image(my_image, img_index, my_logger, true_bounding_boxes):
    #reate grid z-level 0
    #img_arr1, list_of_boxes1 = img_handler.get_grid(my_image, 0.4, WINDOW_X, WINDOW_Y, 3)     # 51 pixel
    #img_arr2, list_of_boxes2 = img_handler.get_grid(my_image, 1.0, WINDOW_X, WINDOW_Y, 3)     # 128 px
    #img_arr3, list_of_boxes3 = img_handler.get_grid(my_image, 0.75, WINDOW_X, WINDOW_Y, 3)     # 96 pixel

    img_arr1, list_of_boxes1 = img_handler.get_grid(my_image, 0.6375 , WINDOW_X, WINDOW_Y, 3)     # 51 pixel
    img_arr2, list_of_boxes2 = img_handler.get_grid(my_image, 1.6, WINDOW_X, WINDOW_Y, 3)     # 128 px
    img_arr3, list_of_boxes3 = img_handler.get_grid(my_image, 0.9, WINDOW_X, WINDOW_Y, 3)     # 96 pixel
    
    number_of_clips = img_arr1.shape[0] + img_arr2.shape[0] + img_arr3.shape[0]
    num_channels = 3

    img_arr = np.empty((number_of_clips, WINDOW_Y, WINDOW_X, num_channels), dtype=int)
    img_arr, index = add_arr(img_arr, img_arr1, 0)
    img_arr, index = add_arr(img_arr, img_arr2, index)
    img_arr, index = add_arr(img_arr, img_arr3, index)
    img_arr = img_arr / 255

    list_of_boxes = []

    # using naive method to concat 
    list_of_boxes = add_list(list_of_boxes, list_of_boxes1)
    list_of_boxes = add_list(list_of_boxes, list_of_boxes2)
    list_of_boxes = add_list(list_of_boxes, list_of_boxes3)

    results = my_model.predict(img_arr)
    print("Prediction completed for image using grid clips")

    #filter for results that meet threshold for the object detector
    
    results_true = []
    for index in range(0, len(results)):
        result = results[index]
        if (result[1] > detector_threshold):
            results_true.append([index, result[1], True])
   
    # (!) you need to check to see if this result overlaps with previous one, if so does it have higher prediction on the class, if so then you should remove the other prediction and replace with this one
    for index_1 in range(0, len(results_true)):
        box_1 = list_of_boxes[results_true[index_1][0]]
        predict_box_1 = bbox.BBox2D([box_1[0], box_1[1], box_1[2], box_1[3]], mode=1)

        for index_2 in range(0, len(results_true)):
            if (index_1 == index_2): # dont compare itself
                continue
            # get the bounding box corresponding to this one
            box_2 = list_of_boxes[results_true[index_2][0]]            
            predict_box_2 = bbox.BBox2D([box_2[0], box_2[1], box_2[2], box_2[3]], mode=1)

            # do these boxes overlap?
            iou = jaccard_index_2d(predict_box_1, predict_box_2)            
            if (iou > 0.1):
                # is Box 1 beat?
                box_1_value = results_true[index_1][1]
                box_2_value = results_true[index_2][1]
                
                if (box_2_value >= box_1_value):
                    #flag box one as no good
                    results_true[index_1][2] = False
                    #bump up box_2 value a little bit
                    results_true[index_2][1] = results_true[index_2][1] * 1.00001
              
    #now remove all entries where = False
    results_final = []
    for index in range(0, len(results_true)):
        if results_true[index][2] == True:
            results_final.append(results_true[index])

    # draw the boxes on the image    
    ious = []   
    true_box_iou = [-1.0 for i in range(len(true_bounding_boxes))]
    for index in range(0, len(results_final)):          
        #person found, draw the box
        box = list_of_boxes[results_final[index][0]]  # 0 = index of the box
        #draw rectangle
        cv2.rectangle(my_image, (box[0], box[1]), 
                                (box[2], box[3]), (255, 255, 255), 3)

        # calculate area of overlap
        ious = []
        max_iou = 0.0
        for q in range(0, len(true_bounding_boxes)):
            true_bounding_box = true_bounding_boxes[q]
            predict_box = bbox.BBox2D([box[0], box[1], box[2], box[3]], mode=1)                
            true_box = bbox.BBox2D(true_bounding_box,mode=1)
            iou = jaccard_index_2d(predict_box, true_box)
            ious.append(iou)
            max_iou = max(iou, max_iou)
            true_box_iou[q] = max(true_box_iou[q], iou)

        # record the iou
        my_logger.write_line(str(img_index) + "," + str(results_final[index][0])+ "," + str(max_iou) + "," + file_to_scan + "\n")

    # record any 0 iou left over on true_boxes, i.e. false negative
    for q in range(0, len(true_bounding_boxes)):
        if true_box_iou[q] <= 0:
            my_logger.write_line(str(img_index) + "," + str(-1)+ "," + str(true_box_iou[q]) + "," + file_to_scan + "\n")

    # calculate false negatives
    # that each bounding box to see if it has been identified as a head
     
    
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
   
    return



my_model.summary()
meta_data = data.get('_via_img_metadata')
files = file_handler.get_file_list(my_paths.TTC_PATH)

my_logger = r_logger.R_logger(os.path.join("richardson_info_files", "obj_mAP_values.csv"))
my_logger.clear()
my_logger.write_line("Detector threshold: " + str(detector_threshold) + "\n")
my_logger.write_line("Model Name: " + model_name + "\n")
my_logger.write_line("imageid, boundingbox_id, IOU, sourceimage\n")

img_index=0
for file_to_scan in files:
    #get ground truth bounding box from JSON
    true_bounding_boxes = get_bounding_box(meta_data, file_to_scan)
    my_image = file_handler.load_image(file_to_scan, path=my_paths.TTC_PATH, flag_bw_load=False)[0]
    process_image(my_image, img_index, my_logger, true_bounding_boxes)
    img_index = img_index + 1


  
