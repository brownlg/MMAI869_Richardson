import richardson_file_handlers as file_handler
import pandas as pd
import cv2
import random as random


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
            clipped_images.append({ row['LabelName']: img_clipped })

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
            if (    (x_min > int(width * row['XMin'])) 
                and (x_min < int(width * row['XMin'])) 
                and (y_min > int(width * row['XMin']))                
                and (y_max < int(width * row['XMin']))):
                flag_inside = True
                break

        if (flag_inside):
            continue

        #clip this image because it is not in the target box
        img_clipped = my_img[0, y_min : y_max, 
                                    x_min : x_max, :]

        success = success + 1
        clipped_images.append({ 'non-target' : img_clipped })

        
                   

    return clipped_images

def zoom_to_fit_box(box_width, box_height, my_image):    
    img_height, img_width, img__color = my_image.shape

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
