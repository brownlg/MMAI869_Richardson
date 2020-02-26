import richardson_file_handlers as file_handler
import richardson_image_handlers as img_handler
import richardson_path as my_paths
import cv2
import os
import json as json
import bbox
from bbox.metrics import jaccard_index_2d

from Richardson_Logger import r_logger

#dimensions of input image
WINDOW_X = 80
WINDOW_Y = 80

# load the trained neural network
from keras.models import load_model
#my_model = load_model("my_classifier_soft_max_2.h5") # works on faces, 1st one that actually seemed to work
#my_model = load_model("my_classifier_soft_max_2.h5") 
my_model = load_model("path_1_richardson_V1.h5") 
print("done")

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

my_logger = r_logger.R_logger(os.path.join("richardson_info_files", "obj_mAP_values.csv"))
my_logger.clear()
my_logger.write_line("imageid, boundingbox_id, IOU, sourceimage\n")

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

    #filter for results that meet threshold for the object detector
    detector_threshold = 0.9999
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
                
                if (box_2_value > box_1_value):
                    #flag box one as no good
                    results_true[index_1][2] = False

    #now remove all entries where = False
    results_final = []
    for index in range(0, len(results_true)):
        if results_true[index][2] == True:
            results_final.append(results_true[index])

    # draw the boxes on the image    
    ious = []   
    for index in range(0, len(results_final)):          
        #person found, draw the box
        box = list_of_boxes[results_final[index][0]]  # 0 = index of the box
        #draw rectangle
        cv2.rectangle(my_image, (box[0], box[1]), 
                                (box[2], box[3]), (255, 255, 255), 3)

        # calculate area of overlap
        ious = []
        max_iou = 0.0
        for true_bounding_box in true_bounding_boxes:
            predict_box = bbox.BBox2D([box[0], box[1], box[2], box[3]], mode=1)                
            true_box = bbox.BBox2D(true_bounding_box,mode=1)
            iou = jaccard_index_2d(predict_box, true_box)
            ious.append(iou)
            max_iou = max(iou, max_iou)

        # record the iou
        my_logger.write_line(str(img_index) + "," + str(results_final[index][0])+ "," + str(max_iou) + "," + file_to_scan + "\n")
       
    
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
