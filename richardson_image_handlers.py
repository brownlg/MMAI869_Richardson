import richardson_file_handlers as file_handler
import pandas as pd
import numpy as np
import cv2
import random as random
from bbox.metrics import jaccard_index_2d, BBox2D

BLUR_SIZE = 21

def create_clipped_images(img_id, filepath, target_rows, IMG_WINDOW_X, IMG_WINDOW_Y):        
    # copy the bounding box to new image
    
    # load the image
    my_img = file_handler.load_image(filepath + img_id + ".jpg")
    index, height, width, color = my_img.shape        
    print("Image height: " + str(height))
    print("Image width: "+  str(width))

    # first add the target images
    clipped_images = [] #list    
    for index, row in target_rows.iterrows():  
        img_clipped = my_img[0, int(height * row['YMin']) : int(height * row['YMax']), 
                                int(width * row['XMin']) : int(width * row['XMax']), :]

        img_clipped = zoom_to_fit_box(IMG_WINDOW_X, IMG_WINDOW_Y, img_clipped)

        img_height, img_width, img__color = img_clipped.shape

        if ((img_width == 0) or (img_height ==0)):
            continue
       
        # check to make sure image taller than wide
        img_height, img_width, img__color = img_clipped.shape
        if ((img_height*.75) > img_width):
            # if the image meets aspect ratio requirements than add it
            #now you need to process the image so that it fits window size
            img_background = np.copy(img_clipped)

            #stretch it to fit the window
            img_background = stretch_to_fit_box(img_background, IMG_WINDOW_X, IMG_WINDOW_Y)

            #blurr it severely
            #src = cv2.imread('flower.jpg', cv2.IMREAD_UNCHANGED)
            for i in range(0,10):                
                img_background = cv2.GaussianBlur(img_background,(BLUR_SIZE,BLUR_SIZE),cv2.BORDER_DEFAULT)

            # add alpha channel to both images
            r_channel, g_channel, b_channel = cv2.split(img_clipped)
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
            img_clipped_RGBA = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))

            r_channel, g_channel, b_channel = cv2.split(img_background)
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
            img_background_RGBA = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))


            #now paste the img on the background
            alpha = 1.0
            beta = 1.0
            #beta = (1.0 - alpha)
            top = int( (IMG_WINDOW_Y - img_height) / 2 )
            bottom = int( (IMG_WINDOW_Y - img_height) / 2 )
            left = int( (IMG_WINDOW_X - img_width) / 2 )
            right = int( (IMG_WINDOW_X - img_width) / 2 )
            borderType = cv2.BORDER_CONSTANT

            # (!) error in code sometime i get negative borders!
            flag_error = False
            if ((left < 0) or (right<0) or (top<0) or (bottom<0)):
                flag_error = True

            if (flag_error == False):
                value = [0, 0, 0, 0]
                target = cv2.copyMakeBorder(img_clipped_RGBA, top, bottom, left, right, borderType, None, value)

                #now blur the edges of the photo for nice transition to background
                for i in range(0,5):                
                    blurred_transition = cv2.GaussianBlur(target,(BLUR_SIZE,BLUR_SIZE),cv2.BORDER_DEFAULT)
                                
                #mask out the middle
                EDGE_MIX = 1.2
                #erase
                cv2.rectangle(blurred_transition,(int(left * EDGE_MIX), int(top * EDGE_MIX)), (int(IMG_WINDOW_X - right * EDGE_MIX), int(IMG_WINDOW_Y - bottom * EDGE_MIX)), (0, 0, 0, 255), -1)

                #make transparent
                cv2.rectangle(blurred_transition,(int(left * EDGE_MIX), int(top * EDGE_MIX)), (int(IMG_WINDOW_X - right * EDGE_MIX), int(IMG_WINDOW_Y - bottom * EDGE_MIX)), (0, 0, 0, 0), -1)

                file_handler.save_image("blurred_transition.png", "", blurred_transition, True)
                file_handler.save_image("target.png", "", target, True)
                #target_with_blurry_edge = cv2.addWeighted(blurred_transition, alpha, target, beta, 0)    

                # Now create a mask of logo and create its inverse mask also
                img2gray = cv2.cvtColor(blurred_transition, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask) 

                # Now black-out the area of logo in ROI
               # new_back = cv2.bitwise_and(target, target, mask = mask_inv)
                
                target_with_blurry_edge = cv2.bitwise_or(blurred_transition, target)    

                file_handler.save_image("new_back.png", "", img2gray, True)             
                file_handler.save_image("blurred_transition_edge.png", "", target_with_blurry_edge, True)             

            #resize just incase due to rounding errors
            target_with_blurry_edge = stretch_to_fit_box(target_with_blurry_edge, IMG_WINDOW_X, IMG_WINDOW_Y)
            # first image is on top
            target2 = cv2.addWeighted( target_with_blurry_edge, alpha, img_background_RGBA, beta, 0)          

            clipped_images.append({ row['LabelName']: target2 })

    # create false targets
    MAX_TRY = len(clipped_images) * 3
    success = 0
    for i in range(0, 100): # randomly select 100 clips to try and get MAX_TRY images
        if (success >= MAX_TRY):
            break

        #select a clip that is not in target
        y_min = int (random.randint(0, height-IMG_WINDOW_Y))
        x_min = int (random.randint(0, width-IMG_WINDOW_X))

        y_max = IMG_WINDOW_Y + y_min
        x_max = IMG_WINDOW_X + x_min

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
        img_clipped = my_img[0, y_min : y_max, 
                                x_min : x_max, :]


        success = success + 1
    #    clipped_images.append({ 'non-target' : img_clipped })
          

    return clipped_images

def stretch_to_fit_box(my_image, box_width, box_height):
    img_height, img_width, img__color = my_image.shape

    #calculate zoom up X & Y
    zoom_x = float(box_width) / float(img_width)
    zoom_y = float(box_height) / float(img_height)
    
    return cv2.resize(my_image, dsize=(int(zoom_x * img_width), int(zoom_y * img_height)), interpolation=cv2.INTER_CUBIC)

def zoom_to_fit_box(box_width, box_height, my_image):    
    img_height, img_width, img_color = my_image.shape

    if ((img_width == 0) or (img_height ==0)):
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

    print("\n")
    print(my_rows.head())

    return my_rows

