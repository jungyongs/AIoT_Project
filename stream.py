import argparse
import os
import sys
from pathlib import Path
import cv2
import tensorflow as tf

from utils.utils import read_label_file, letterbox, inference_int8, non_max_suppression, postprocessing, append_objs_to_img
from utils.sort import Sort

def getCameraStream(im0, interpreter, inference_size, labels, mot_tracker, conf_thres, iou_nms_thres, max_det, notrack):
    # Image Preprocessing        
    im = letterbox(im0, inference_size, stride=32, auto=False)[0] # resize + padding + add border. stride can be excludede if auto is always False
    im = im.transpose((2, 0, 1))[::-1]  # bgr to rgb, hwc to chw
    
    # Inference
    pred = inference_int8(interpreter, im)
    
    # NMS
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_nms_thres, max_det=max_det)
    
    # Postprocessing
    objs = postprocessing(pred)

    # Add tracker ID
    if not notrack:
        trks = mot_tracker.update(objs)
    else:
        trks = objs
    
    # Draw boxes to image
    im0 = append_objs_to_img(im0, inference_size, trks, labels, notrack)
    
    people = []
    for trk in trks:
        people.append({
            'id': trk[6],
            'state': trk[5]
        })
    
    return im0