# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
"""
import argparse
import cv2
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.utils import read_label_file, letterbox, inference_int8, non_max_suppression, postprocessing, append_objs_to_img, Timer
from utils.sort import Sort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def run(model=ROOT / 'models/best-int8.tflite',  # model path
        source=ROOT / 'data/sample.mp4',  # input video path
        labels=ROOT / 'models/labels.txt',  # label txt path
        conf_thres=0.25,  # confidence threshold
        iou_nms_thres=0.7,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_dir=ROOT / 'out',
        notrack=False,
        length=0,
        stream=False,
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        vid_stride=1  # video frame-rate stride
):
    print('Loading {} with {} labels.'.format(model, labels))
    
    # Load the tflite model
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inference_size = interpreter.get_input_details()[0]['shape']
    inference_size = (inference_size[1], inference_size[2])
    
    # Load the labels text file into dict
    labels = read_label_file(labels)

    # Check the video source (num for webcam)
    source = str(source)
    webcam = source.isnumeric()
    if webcam:
        source = int(source)
    cap = cv2.VideoCapture(source)

    if cap.isOpened():
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        if webcam:
            print(f"--source {source} camera unsupported")
            return 1
        else:
            print(f"Cannot open file {source}")
            return 1

    # Set the videowriter with mp4 codec
    if webcam:
        save_path = str(save_dir / 'out.mp4')
    else:
        p = Path(source)
        save_path = str((save_dir / p.name).with_suffix('.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    # Set the total frame number (same as input if length == 0)
    if length == 0:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    elif length < 0:
        print("Invalid output video length.")
        return 1
    else:
        frames = int(fps * length)
    frames0 = frames # copy to calculate proportion

    # Initialize SORT isntance    
    mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    
    # Set timer
    dt = [Timer(), Timer(), Timer(), Timer()]
    
    # Run the loop by each frame
    while cap.isOpened() and frames > 0:
        ret, frame = cap.read()
        if not ret:
            break
        im0 = frame
        
        with dt[0]:
            # Image Preprocessing        
            im = letterbox(im0, inference_size, stride=32, auto=False)[0] # resize + padding + add border. stride can be excludede if auto is always False
            im = im.transpose((2, 0, 1))[::-1]  # bgr to rgb, hwc to chw

        with dt[1]:
            # Inference
            pred = inference_int8(interpreter, im)

        with dt[2]:
            # NMS
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_nms_thres, max_det=max_det)
            
            # Postprocessing
            objs = postprocessing(pred)
    
        with dt[3]:
            # Add tracker ID
            if not notrack:
                trks = mot_tracker.update(objs)
            else:
                trks = objs

        # Draw boxes to image
        im0 = append_objs_to_img(im0, inference_size, trks, labels, notrack)

        # Write to video
        out.write(im0)
        
        # Stream to window (or yield binary image)
        if stream:
            # ret, buffer = cv2.imencode('.jpg', im0)        
            # binary = bytearray(buffer.tobytes())
            # yield(b'--PNPframe\r\n' b'Content-Type: image/jpeg\r\n\r\n' + binary + b'\r\n')
            cv2.imshow('teststream', im0)
            
        frames -=1        
        print(f'frame #: {frames0 - frames}/{frames0}')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    t = tuple(x.t / frames0 * 1E3 for x in dt)  # speeds per frame
    print('Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, and %.1fms tracking per frame' % t)
    cap.release()
    cv2.destroyAllWindows()

def parse_opt():      
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=ROOT / 'models/best-int8.tflite', help='.tflite model path')
    parser.add_argument('--source', default=ROOT / 'data/sample.mp4', help='input video path (str) or index of source webcam (0 ~ n)')
    parser.add_argument('--labels', default=ROOT / 'models/labels.txt', help='label file path')
    parser.add_argument('--conf_thres', default=0.25, type=float, help='classifier score threshold')
    parser.add_argument('--iou_nms_thres', default=0.7, type=float, help='nms iou threshold')
    parser.add_argument('--max_det', default=1000, type=int, help='number of categories with highest score to display')
    parser.add_argument('--save_dir', default=ROOT / 'out', help='output video directory')
    parser.add_argument('--length', default=0, type=int, help='Specify the length of video')
    parser.add_argument('--notrack', action='store_true', help='detect without SORT tracker')
    parser.add_argument('--stream', action='store_true', help='stream to window')
    opt = parser.parse_args()
    return opt

def main(opt):
    # check requirements
    run(**vars(opt))
            
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
