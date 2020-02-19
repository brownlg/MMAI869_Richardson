from glob import glob
import os
from random import random
import json

print(os.getcwd())
#os.chdir('Desktop')
# Path to your images
"""
image_paths = glob('TTC_pics/*')
#Path to your annotations from VIA tool
annotation_file = 'via_region_data.json'
#clean up the annotations a little
annotations = json.load(open(annotation_file))
cleaned_annotations = {}
for k,v in annotations['_via_img_metadata'].items():
    cleaned_annotations[v['filename']] = v
# create train and validation directories
os.mkdir('ttcdata')
os.mkdir('ttcdata/val')
os.mkdir('ttcdata/train')
train_annotations = {}
valid_annotations = {}
# 20% of images in validation folder
for img in image_paths:
    # Image goes to Validation folder
    if random()<0.2:
        os.system("cp "+ img + " ttcdata/val/")
        img = img.split("/")[-1]
        valid_annotations[img] = cleaned_annotations[img]
    else:
        os.system("cp "+ img + " ttcdata/train/")
        img = img.split("/")[-1]
        train_annotations[img] = cleaned_annotations[img]
# put different annotations in different folders
with open('ttcdata/val/via_region_data.json', 'w') as fp:
    json.dump(valid_annotations, fp)
with open('ttcdata/train/via_region_data.json', 'w') as fp:
    json.dump(train_annotations, fp)
"""
#colne Mask_RCNN
#use command + shift + P Git:Clone
#https://github.com/matterport/Mask_RCNN.git

#os.chdir('Mask_RCNN')
pip install -r requirements.txt

#make a copy of the balloon dataset to make the required changes
cp -r samples/balloon samples/person_head

import sys
import itertools
import math
import logging
import random
import re
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

#own dataset
from samples.person_head import person_head

%matplotlib inline

config = person_head.ttcConfig()

#change directory to your own
ttc_DIR = "/Users/emoney/Desktop/ttcdata/"

# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = person_head.ttcDataset()
dataset.load_ttc(ttc_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    class_ids = [int(x) for x in class_ids]
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names,limit=1)

# Load random image and mask.
image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
class_ids = np.array([int(x) for x in class_ids])
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

## Create Model
dataset

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

## Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

#load the weights for COCO
#model.load_weights('/Users/emoney/Desktop/Mask_RCNN/mask_rcnn_coco.h5', 
                   #by_name=True, 
                   #exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 100
    #FPN_CLASSIF_FC_LAYERS_SIZE = 1028
    #IMAGE_MAX_DIM = 1028
    #IMAGE_SHAPE = [1028, 1028, 3]

config = InferenceConfig()
config.display()

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    #Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Training dataset
dataset_train = person_head.ttcDataset()
dataset_train.load_ttc(ttc_DIR, "train")
dataset_train.prepare()

# validation dataset
dataset_val = person_head.ttcDataset()
dataset_val.load_ttc(ttc_DIR, "val")
dataset_val.prepare()

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=30, 
            layers='heads')

#Detection

from mrcnn.config import Config

class ttcConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "person_head"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + person_head

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(ttcConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()


# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

#Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())

#EVALUATION

#Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))