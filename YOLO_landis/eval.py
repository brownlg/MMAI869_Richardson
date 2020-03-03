"""
Evaluation script for PyTorch Darknet model.

Evaluates the overlap between ground truth test labels and predicted labels
on the same images.  If there is one box from the predictions of an image
that overlaps by a user specifified threshhold, the true positive count is
increased and Average Precision will be higher.

If running on CPU-only, this script may be slow.
"""

import os
import argparse
import random
import copy
import glob

import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageOps

from darknet import Darknet, get_test_input
from customloader import CustomDataset, transform_annotation
from torch.utils.data import DataLoader
from data_aug.data_aug import Sequence, Equalize, Normalize, YoloResizeTransform
from util import process_output, de_letter_box, load_classes
from live import prep_image
from bbox import center_to_corner, center_to_corner_2d

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle as pkl

DEBUG = True

if DEBUG:
    logwriter = open('debug_model.log', 'w')

random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def arg_parse():
    """
    Parse arguments to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Evaluation Module')

    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="epoch199-final-bs2-loss236.8104.pth", type=str)
    parser.add_argument("--overlap", dest="overlap_thresh", 
                        help="Overlap threshhold", default=0.5)
    parser.add_argument("--plot-conf", dest="plot_conf", type=float,
                        help="Bounding box plotting confidence", default=0.8)
    return parser.parse_args()

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.

    Arguments
    ---------
    boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
    scores: (tensor) The class predscores for the img, Shape:[num_priors].
    overlap: (float) The overlap thresh for suppressing unnecessary boxes.
    top_k: (int) The Maximum number of box preds to consider.

    Returns
    -------
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 

    Input boxes are expected to be in x1y1x2y2 format.
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def average_precision(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def custom_eval(predictions_all,
             ground_truths_all,
             num_gts,
             ovthresh=0.5):
    """
    Arguments
    ---------
    predictions_all: list of list of tensors 
        with predicted bounding boxes
    ground_truths_all : list of list of arrays 
        with ground truth bounding boxes
    num_gts : int
        number of ground truths
    ovthresh : float
        overlap threshold (default = 0.5)

    Returns
    -------
    rec, prec, ap :  tuple
                     tuple of list, list, float
    """

    image_num = len(ground_truths_all)
    tp = np.zeros(image_num)
    fp = np.zeros(image_num)
    print('Calculating overlaps.')
    # For each image look for overlaps between ground truth and predictions
    for i in range(image_num):
        print('Image ', i)
        predictions = predictions_all[i]
        ground_truths = ground_truths_all[i]
        if len(predictions) == 0:
            print('Prediction for image is emtpy.')
            fp[i] = 1.
            continue
        confidence = predictions[:, 4]
        BB = predictions[:, :4]

        # Sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]

        # Get just bounding boxes for all gt
        BBGT = ground_truths[0][:, :4]

        # Number detections and number gt
        nd = BB.shape[0]
        ngt = BBGT.shape[0]
        
        # Go down detections and ground truths and calc overlaps (IOUs)
        overlaps = []
        for d in range(nd):
            bb = BB[d]
            for gt in range(ngt):
                bbox1 = torch.tensor(BBGT[np.newaxis, gt, :], dtype=torch.float)
                bbox2 = torch.tensor(bb[np.newaxis, :].clone().detach(), dtype=torch.float)
                bboxes_ious = bbox_iou(bbox1, bbox2)
                overlaps.append(bboxes_ious)
        # Get best IOU prediction/gt match
        ovmax = np.max(np.array(overlaps))

        # Mark TPs and FPs
        if ovmax > ovthresh:
            tp[i] = 1.
            print('tp!')
        else:
            fp[i] = 1.

    # Compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_gts)
    # Avoid divide by zero
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = average_precision(rec, prec, use_07_metric=False)

    return rec, prec, ap

def write(x, img):
    """
    Arguments
    ---------
    x : array of float
        [batch_index, x1, y1, x2, y2, objectness, label, probability]
        where x1, y1 is left, bottom corner
    img : numpy array
        original image

    Returns
    -------
    img : numpy array
        Image with bounding box drawn


    Note
    ----
    OpenCV draws and calls x1, y1, x2, y2 differently:

    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

    x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2

    Such that new_y1 = y2, new_y2 = y1

    """

    # Scale up thickness of lines to match size of original image
    scale_up = int(img.shape[0]/416)

    if x[-1] is not None:

        x = [int(n) for n in x]
        x_copy = copy.deepcopy(x)
        ## new_y1 = y2, new_y2 = y1
        # print('old x ', x)
        x[1] = x_copy[3]
        x[3] = x_copy[1]
        # print('new x', x)
        ## top, left corner of rectangle
        c1 = (x[0],x[1])
        # c1 = (x[3], x[0])
        # c1 = (x[1], x[0])
        ## bottom, right corner of rectangle 
        c2 = (x[2],x[3])
        # c2 = (x[1], x[2])
        # c2 = (x[3], x[2])
        label = int(x[-2])
        label = "{0}".format(classes[label])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, thickness=1*scale_up)
        t_size = cv2.getTextSize(text=label,
                                 fontFace=cv2.FONT_HERSHEY_PLAIN, 
                                 fontScale=1*scale_up//2, 
                                 thickness=1*scale_up)[0]
        c3 = (x[0], x[3])
        c4 = c3[0] + t_size[0] + 3, c3[1] + t_size[1] + 4
        cv2.rectangle(img, c3, c4, color, thickness=-1)
        cv2.putText(img, label, (x[0], x[3] + t_size[1] + 4), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1*scale_up//2,
                    color=[225,255,255],
                    thickness=1*scale_up)
    return img

if __name__ == "__main__":
    args = arg_parse()
    overlap_thresh = float(args.overlap_thresh)

    # Instantiate a model
    model = Darknet(args.cfgfile, train=False)

    # Get model specs
    model_dim = int(model.net_info["height"])
    assert model_dim % 32 == 0 
    assert model_dim > 32
    num_classes = int(model.net_info["classes"])
    bbox_attrs = 5 + num_classes

    # Load weights PyTorch style
    model.load_state_dict(torch.load(args.weightsfile))
    model = model.to(device)  ## Really? You're gonna eval on the CPU? :)

    # Set to evaluation (don't accumulate gradients)
    # Make sure to call eval() method after loading weights
    model.eval()

    # Define tranforms for images and bboxes
    transforms = Sequence([YoloResizeTransform(model_dim), Normalize()])

    # Load test data
    test_data = CustomDataset(root="data", ann_file="data/test.txt", 
                              det_transforms=transforms,
                              cfg_file=args.cfgfile,
                              num_classes=num_classes)
    test_loader = DataLoader(test_data, 
                             batch_size=1,
                             shuffle=False)

    ground_truths_all = []
    predictions_all = []
    num_gts = 0

    # Make a directory for the image files with their bboxes
    eval_output_dir = os.path.split(test_data.examples[0].rstrip())[0].replace('obj', 'eval_output')
    os.makedirs(eval_output_dir, exist_ok=True)

    for i, (image, ground_truth, filepath) in enumerate(test_loader):
        
        img_file = filepath[0].rstrip()
        img_ = plt.imread(img_file)
        orig_h, orig_w = img_.shape[0], img_.shape[1]
        orig_dim = torch.FloatTensor([orig_w, orig_h]).repeat(1,2)
        print(img_file)
        print(i)

        # Deal with ground truth: read, convert to corners and scale
        ground_truths = []
        gt_file = '.'.join(img_file.split('.')[:-1]) + '.txt'
        gt_df = np.array(pd.read_csv(gt_file, sep=' ', header=None))
        gt_df = center_to_corner_2d(gt_df[:, 1:])
        orig_dim_np = np.array([orig_w, orig_h, orig_w, orig_h])
        gt_df *= orig_dim_np
        ground_truths.append(gt_df)
        num_gts = gt_df.shape[0]

        # ground_truths = []
        # if len(ground_truth) == 0:
        #     continue
        # else:
        #     ground_truths.append(ground_truth)

        # Predict on input test image
        image = image.to(device)
        with torch.no_grad():        
            output = model(image)

        # NB, output is:
        # [batch, image_id, [x_center, y_center, width, height, objectness_score, class_score1, class_score2, ...]]
        
        if output.shape[0] > 0:
            # To later plot bboxes
            img_ = plt.imread(img_file)

            output = output.squeeze(0)

            for i in range(output.shape[0]):
                if output[i, -1] >= float(args.plot_conf):
                    logwriter.write('output before center2corner,' + ','.join([str(x.item()) for x in output[i]]) + '\n')

            output = process_output(output, num_classes)

            # Center to corner
            output_ = copy.deepcopy(output)
            output[:,0] = output_[:,0] - output_[:,2]/2
            output[:,1] = output_[:,1] - output_[:,3]/2
            output[:,2] = output_[:,0] + output_[:,2]/2
            output[:,3] = output_[:,1] + output_[:,3]/2

            # NMS
            keep = nms(output[:,:4], output[:,-1])
            output = torch.index_select(output, 0, keep[0])

            # Scale
            output = output[output[:,-1] > float(args.plot_conf), :]
            outputs = []
            if output.size(0) > 0:
                scale = float(model_dim)/orig_dim
                orig_dim = orig_dim.repeat(output.size(0), 1)
                output[:,:4] /= scale
                output[:, [0,2]] = torch.clamp(output[:,[0,2]], 0.0, orig_dim[0,0])
                output[:, [1,3]] = torch.clamp(output[:,[1,3]], 0.0, orig_dim[0,1])
                outputs = list(np.asarray(output[:,:8]))

            classes = load_classes('data/obj.names')
            colors = pkl.load(open("pallete", "rb"))
            
            # # Test resizing to model dim
            # img_ = Image.fromarray(np.uint8(img_))
            # img_ = F.resize(img_, (model_dim, model_dim))
            # img_ = np.asarray(img_)
            list(map(lambda x: write(x, img_), outputs))
            
            # Write image with bboxes to a new folder
            plt.imsave(img_file.replace('.'+img_file.split('.')[-1], '_out.jpg').replace(
                img_file.split(os.sep)[-2], 'eval_output'), img_)

            outputs = torch.tensor(outputs)

            ground_truths_all.append(ground_truths)
            predictions_all.append(outputs)

    ## WIP, use with caution
    # prec, rec, aps = custom_eval(predictions_all, ground_truths_all, num_gts=num_gts, ovthresh=overlap_thresh)
    # print('Precision ', prec, 'Recall ', rec, 'Average precision ', aps, sep='\n')