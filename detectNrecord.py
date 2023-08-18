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

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os
import collections
import numpy as np
import tensorflow as tf

from utils.utils import read_label_file, letterbox, inference_int8, non_max_suppression, postprocessing, append_objs_to_img
from utils.sort import Sort

def main():
    default_model_dir = './models'
    default_model = 'best-int8.tflite'
    default_labels = 'labels.txt'
    default_input_dir = './data'
    default_input = 'sample.mp4'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=300,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, help='classifier score threshold', default=0.25)
    parser.add_argument('--iou_nms', type=float, help='nms iou threshold', default=0.45)
    parser.add_argument('--input', default=os.path.join(default_input_dir, default_input),
                        help='input video path')
    parser.add_argument('--output', default='./out.mp4',
                        help='output video path')
    parser.add_argument('--length', type=int, default=7,
                        help='Specify the length of video (camera only)')
    parser.add_argument('--track', type=int, default=1,
                        help='1 or 0: with or without SORT tracker')
    
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    inference_size = interpreter.get_input_details()[0]['shape']
    inference_size = (inference_size[1], inference_size[2])
    
    labels = read_label_file(args.labels)
    
    conf_thres=args.threshold # default 0.25
    iou_thres=args.iou_nms # default 0.45
    max_det=args.top_k # default 300
    
    track = args.track

    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera_idx)

    if cap.isOpened():
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        print("Cannot open file {}", args.input)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    frames = fps * args.length
    if args.input:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames0 = frames

    # Initialize SORT isntance    
    mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    
    while cap.isOpened() and frames > 0:
        ret, frame = cap.read()
        if not ret:
            break
        im0 = frame
        
        # Image Preprocessing        
        im = letterbox(im0, inference_size, stride=32, auto=False)[0] # resize + padding + add border. stride can be excludede if auto is always False
        im = im.transpose((2, 0, 1))[::-1]  # bgr to rgb, hwc to chw
        
        # Inference
        pred = inference_int8(interpreter, im)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
        
        # Postprocessing
        objs = postprocessing(pred)
    
        # Add tracker ID
        if track:
            trks = mot_tracker.update(objs)
        else:
            trks = objs
        
        # Draw boxes to image
        im0 = append_objs_to_img(im0, inference_size, trks, labels, track)

        out.write(im0)
        frames -=1
        
        print(f'frame #: {frames0 - frames}/{frames0}')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
            
if __name__ == '__main__':
    main()
