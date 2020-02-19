import richardson_file_handlers as file_handler
import pandas as pd
import numpy as np
import cv2
import random as random
from bbox.metrics import jaccard_index_2d, BBox2D

BLUR_SIZE = 21


print(cv2.__version__)


DEBUG_MODE = False

# for object detection, return an array of images
def get_grid(my_image, zoom_level, window_x, window_y, num_channels):
    img_height, img_width, img_color = my_image.shape
    print (my_image.shape)

    #create image array
    number_of_clips = 1
    img_arr = np.empty((number_of_clips, window_y, window_x, num_channels), dtype=int)

    list_of_boxes = []

    cc = 0

    #create list of boxes check
    x_min = 0
    y_min = 0
    x_max = x_min + window_x
    y_max = y_min + window_y
    step_x = 30
    step_y = 30
    steps_x =  int((img_width - window_x) /step_x)
    steps_y = int((img_height - window_y) / step_y)

    if (steps_x == 0) and (steps_y == 0):
        img_clipped = my_image[y_min : y_max, 
                               x_min : x_max, :]
        img_arr[0] = img_clipped
        list_of_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))     
        return img_arr, list_of_boxes

    #create image array
    number_of_clips = steps_x * steps_y
    img_arr = np.empty((number_of_clips, window_y, window_x, num_channels), dtype=int)

    for y in range(0, steps_y):
        for x in range(0, steps_x):      
            list_of_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))        
            x_min = x_min + step_x       
            x_max = x_min + window_x
        
        x_min = 0
        x_max = x_min + window_x       
        y_min = y_min + step_y
        y_max = y_min + window_y
   
    for box in list_of_boxes:
        img_clipped = my_image[box[1] : box[3], 
                               box[0] : box[2], :]

        clip_height, clip_width, img_color = img_clipped.shape

        #zoom to fit box
        zoom_x = float(window_x) / float(clip_width)
        zoom_y = float(window_y) / float(clip_height)                
      #  img_clipped = cv2.resize(img_clipped, dsize=(int(zoom_x * img_width), int(zoom_y * img_height)), interpolation= cv.INTER_CUBIC)

        #save image for processing
        img_arr[cc] = img_clipped
        cc = cc + 1
       
    return img_arr, list_of_boxes

def create_list_of_false_clips(max_try, number_of_clips, my_img, target_rows, window_x, window_y):
    index, height, width, color = my_img.shape   
    # create false targets
    success = 0
    list_of_boxes = []
    for i in range(0, max_try): # randomly select 100 clips to try and get MAX_TRY images
        if (success >= number_of_clips):
            break

        #make sure the it is valid range
        if ( (height-window_y) <= 2 ):
            return list_of_boxes

        if ( (width-window_x) <= 2 ):
            return list_of_boxes

        #select a clip that is not in target
        y_min = int (random.randint(0, height-window_y))
        x_min = int (random.randint(0, width-window_x))

        y_max = random.gauss(window_y, window_y / 6.0) + y_min
        x_max = random.gauss(window_x, window_x / 3.0) + x_min

        #exception handling, should not happen
        if (y_max > height): 
            continue
        if (x_max > width):
            continue

        #does this clip fall inside target clips?
        flag_inside = False
        for index, row in target_rows.iterrows():  
            box1 = BBox2D([x_min, y_min, x_max, y_max])
            box2 = BBox2D([int(width * row['XMin']), int (height * row['YMin']), int(width * row['XMax']), int(height * row['YMax'] )])

            iou = jaccard_index_2d(box1, box2)

            if (iou > 0):
                flag_inside = True
                break
        
        # this clip is inside the target area so you cannot use it!
        if (flag_inside):
            continue

        #clip this image because it is not in the target box        
        success = success + 1
        list_of_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))        
          
    return list_of_boxes

def create_clipped_images(img_id, filepath, target_rows, window_x, window_y):        
    # copy the bounding box to new image
    
    # load the image
    my_img = file_handler.load_image(filepath + img_id + ".jpg")

    if (my_img is None):
        return None

    index, height, width, color = my_img.shape   
    if (DEBUG_MODE):
        print("Image height: " + str(height))
        print("Image width: "+  str(width))

    # first add the target images
    clipped_images = [] #list    
    for index, row in target_rows.iterrows():  
        img_clipped = my_img[0, int(height * row['YMin']) : int(height * row['YMax']), 
                                int(width * row['XMin']) : int(width * row['XMax']), :]

        result, target_img = process_image_for_window(img_clipped, window_x, window_y)
        if (result == True):
            clipped_images.append({ row['LabelName']: target_img })
            #  file_handler.save_image("boutput.png", "", target2, True)           
    
    # second add the false targets
    list_of_boxes = create_list_of_false_clips(200, 80, my_img, target_rows, window_x, window_y)

    for box in list_of_boxes:
        img_clipped = my_img[0, box[1] : box[3], 
                                box[0] : box[2], :]

        result, target_img = process_image_for_window(img_clipped, window_x, window_y)

        if (result == True):
            clipped_images.append({ 'non-target': target_img })

    return clipped_images


def create_blurred_background(my_img,window_x, window_y):
    return 0

def add_alpha(my_img):
    # add alpha channel to both images
    r_channel, g_channel, b_channel = cv2.split(my_img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
    return cv2.merge((r_channel, g_channel, b_channel, alpha_channel))

def add_border(my_img, window_x, window_y):
    img_height, img_width, img__color = my_img.shape

    top = int( (window_y - img_height) / 2 )
    bottom = int( (window_y - img_height) / 2 )
    left = int( (window_x - img_width) / 2 )
    right = int( (window_x - img_width) / 2 )

    # (!) error in code sometime i get negative borders!
    flag_error = False
    if ((left < 0) or (right<0) or (top<0) or (bottom<0)):
        return False, my_img  # (!) something wrong
    # -------------- fix later ----------------------------

    value = [0, 0, 0, 0] # transparent border
   
    return True, cv2.copyMakeBorder(my_img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value)

def process_image_for_window(my_img, window_x, window_y):    
    img_clipped = zoom_to_fit_box(window_x, window_y, my_img)
    img_background = np.copy(img_clipped)

    img_height, img_width, img_color = img_clipped.shape

    # validate dimensions
    if ((img_width == 0) or (img_height ==0)):
        return False, my_img
            
    if ((img_height*.75) < img_width): # if the image not meets aspect ratio requirements then        
        return False, my_img

        # add alpha channel to both images
    img_clipped_RGBA = add_alpha(img_clipped)
    img_background_RGBA = add_alpha(img_background)

    # Process the background image ----------------
    # stretch it to fit the window
    img_background_RGBA = stretch_to_fit_box(img_background_RGBA, window_x, window_y)        
    #blurr it severely        
    for i in range(0,10):   
        img_background_RGBA = cv2.GaussianBlur(img_background_RGBA,(BLUR_SIZE,BLUR_SIZE),cv2.BORDER_DEFAULT)
    # -----------------------------------------------
    
    # define the border on the foreground image       
    result, target = add_border(img_clipped_RGBA, window_x, window_y)
    if (result == False):
        #error occured -- need to fix
        return False, my_img
    target = stretch_to_fit_box(target, window_x, window_y)
    target = stamp_image(target, img_background_RGBA, window_x, window_y)
    blurred_transition = np.copy(target)

    #now blur the edges of the photo for nice transition to background
    for i in range(0,2):                
        blurred_transition = cv2.GaussianBlur(blurred_transition,(5,5),cv2.BORDER_DEFAULT)

    #del out the middle
    EDGE_MIX = 1.1
    top = int( (window_y - img_height) / 2 )
    bottom = int( (window_y - img_height) / 2 )
    left = int( (window_x - img_width) / 2 )
    right = int( (window_x - img_width) / 2 )
    #erase                
    cv2.rectangle(blurred_transition,(int(left * EDGE_MIX), int(top * EDGE_MIX)), (int(window_x - right * EDGE_MIX), int(window_y - bottom * EDGE_MIX)), (0, 0, 0, 255), -1)
    #make transparent
    cv2.rectangle(blurred_transition,(int(left * EDGE_MIX), int(top * EDGE_MIX)), (int(window_x - right * EDGE_MIX), int(window_y - bottom * EDGE_MIX)), (0, 0, 0, 0), -1)

    #add the images
    target_with_blurry_edge = stamp_image(blurred_transition, target, window_x, window_y)

    #resize just incase due to rounding errors
    target_with_blurry_edge = stretch_to_fit_box(target_with_blurry_edge, window_x, window_y)
    
    # first image is on top            
    target2 = stamp_image(target_with_blurry_edge, img_background_RGBA, window_x, window_y)

    return True, target2

def stamp_image(img_fg, img_bg, window_x, window_y):
    #stretch images to meet window size
    img_fg = stretch_to_fit_box(img_fg, window_x, window_y)
    img_bg = stretch_to_fit_box(img_bg, window_x, window_y)

    # Now create a mask of foreground
    img2gray = cv2.cvtColor(img_fg.astype('uint8'), cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask) 

    # Now black-out the area of logo in ROI
    img_bg = cv2.bitwise_and(img_bg.astype('uint8'), img_bg.astype('uint8'), mask = mask_inv)

    # mask select only relevant position
    img_fg = cv2.bitwise_and(img_fg.astype('uint8'), img_fg.astype('uint8'), mask = mask)

    #add the images       
    return cv2.add(img_fg.astype('uint8'), img_bg.astype('uint8'))

def stretch_to_fit_box(my_image, box_width, box_height):
    img_height, img_width, img__color = my_image.shape

    #calculate zoom up X & Y
    zoom_x = float(box_width) / float(img_width)
    zoom_y = float(box_height) / float(img_height)
    
    return cv2.resize(my_image, dsize=(int(zoom_x * img_width), int(zoom_y * img_height)), interpolation=cv2.INTER_CUBIC)

def zoom_to_fit_box(box_width, box_height, my_image):    
    img_height, img_width, img_color = my_image.shape

    if ((img_width <= 1) or (img_height <= 1)):
        return my_image

    #too big 
    flag_too_big_horizontal = (img_width > box_width)
    flag_too_big_vertical = (img_height > box_height)
   
    if (flag_too_big_horizontal or flag_too_big_vertical):
        zoom_x = float(box_width) / float(img_width)
        zoom_y = float(box_height) / float(img_height)
       
        #scale by smaller zoom factor - done
        zoom = min(zoom_x, zoom_y)
       
        return cv2.resize(my_image, dsize=(int(zoom * img_width), int(zoom * img_height)), interpolation=cv2.INTER_CUBIC)

    flag_too_small_horizontal = (img_width <= box_width)
    flag_too_small_vertical = (img_height <= box_height)

    #too small in both X & Y ?
    if (flag_too_small_horizontal and flag_too_small_vertical):
        #calculate zoom up X & Y
        zoom_x = float(box_width) / float(img_width)
        zoom_y = float(box_height) / float(img_height)
        
        #calculate window zoom up with X
        new_width = int(zoom_x * img_width)
        
        #calcaulte window zoom up with Y
        new_height = int(zoom_y * img_height)

        #decide which window fits
        if (new_width > img_width):
            zoom = zoom_y
        else:
            zoom = zoom_x

        return cv2.resize(my_image, dsize=(int(zoom * img_width), int(zoom * img_height)), interpolation=cv2.INTER_CUBIC)

    #no change to image required
    return my_image

def cut_out_target(target_label, selected_rows):
    my_rows = pd.DataFrame()
    
    # find the rows with human labels
    for it in target_label:
        #print(it + " >> " + select_rows.LabelName)
        human_row = selected_rows.loc[selected_rows.LabelName == it]
        #print(human_row.head())
        if (len(my_rows.index) == 0):
            my_rows = human_row
        else:
            my_rows.append(human_row)

    if (DEBUG_MODE):
        print("\n")
        print(my_rows.head())

    return my_rows

def get_y_value(file_list, y_values_dict, true_positive_labels):
    
    number_rows = len(file_list)
    y_values = np.empty((number_rows,))        
    
    i = 0
    for img_name in file_list:
        if img_name not in y_values_dict:
            print("missing y value set to false")
            print(img_name)
            y_value = 'non-target'
        else:
            y_value = y_values_dict[img_name]

        if y_value in true_positive_labels:
            y_values[i] = 1.0
        else:
            y_values[i] = 0.0
        
        i = i + 1
    return y_values

