"""
Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# The Object detection model is adapted from and pretrained using ultralytics YOLOv5 (https://github.com/ultralytics/yolov5)
# The Pose Estimation model is adapted from Tensorflow Hub MoveNet (https://tfhub.dev/google/movenet/singlepose/lightning/4)
# Person check model is trained using MobileNetV2 (https://doi.org/10.48550/arXiv.1801.04381)
# Tracking mechanism is implemented by SORT (https://github.com/abewley/sort, https://doi.org/10.48550/arXiv.1602.00763)
# Online datasets used for traning: Fall Detection Dataset(https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset), UR Fall Detection Dataset(http://fenix.ur.edu.pl/~mkepski/ds/uf.html)

import argparse
import cv2
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.utils import read_label_file, letterbox, inference_int8, non_max_suppression, postprocessing, append_objs_to_img, Timer, scale_boxes, inference_person
from utils.sort import Sort
from utils.pose import append_poses_to_img, imgToPose, poseToAngle, RNN_model

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

FALL = 0

def run(model=ROOT / 'models/ObjectDetection/best-int8.tflite',  # object detection model path
        source=ROOT / 'data/sample.mp4',  # input video path
        labels=ROOT / 'models/ObjectDetection/labels.txt',  # object detection label txt path
        conf_thres=0.25,  # confidence threshold
        iou_nms_thres=0.7,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        save_dir=ROOT / 'out',
        length=0,
        stream=False,
        person_check=True,
        model_person=ROOT / 'models/PersonCheck/human_detect_quant.tflite',
        model_pose=ROOT / 'models/PoseEstimation/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite',
        model_LSTM=ROOT / 'models/LSTM/LSTM-int8.tflite'
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
    save_dir = Path(save_dir)
    if not save_dir.exists(): # make the output directory
        save_dir.mkdir(parents=True)
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

    # Initialize SORT instance    
    mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    
    # Set object dict (contains feature data for each object ID)
    objects = {}
    
    # Load pose estimation model (MoveNet)
    interpreter_pose = tf.lite.Interpreter(model_path=str(model_pose))
    interpreter_pose.allocate_tensors()
    inference_size_pose = interpreter_pose.get_input_details()[0]['shape']
    inference_size_pose = (inference_size_pose[1], inference_size_pose[2])  
    
    # Load LSTM model
    interpreter_LSTM = tf.lite.Interpreter(model_path=str(model_LSTM))
    interpreter_LSTM.allocate_tensors()
    falls = {} # dict for recording the event of FALL by each ID
    if fps >= 30: # Set the frame interval for uniform LSTM inference by varying fps
        interval_rev = False
        interval = int(fps/30)
    else:
        interval_rev = True
        interval = int(30/fps)
    counter = interval
    
    # Load Person check model
    if person_check:
        person_flag = False # flag for detected person
        person_absence_counter = 0 # counter for flag conversion
        interpreter_person = tf.lite.Interpreter(model_path=str(model_person))
        interpreter_person.allocate_tensors()
    
    # Set timer
    dt = [Timer(), Timer(), Timer(), Timer(), Timer(), Timer(), Timer()]
    
    # Run the loop by each frame
    while cap.isOpened() and frames > 0:
        ret, frame = cap.read()
        if not ret:
            break
        im0 = frame
        
        with dt[6]:
            # Binary img classification for person detection
            if person_check and not person_flag:
                person = inference_person(interpreter_person, im0)
                if not person:
                    break # If not detected, do not proceed
                else:
                    person_absence_counter = 0 # Reset the absence counter
                    person_flag = True
        
        with dt[0]:
            # Image Preprocessing        
            im = letterbox(im0, inference_size, stride=32, auto=False)[0] # resize + padding + add border. stride can be excluded if auto is always False
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # bgr to rgb

        with dt[1]:
            # Inference
            pred = inference_int8(interpreter, im)

        with dt[2]:
            # NMS
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_nms_thres, max_det=max_det)
            
            if person_check and person_flag: # If no BB detected at all for some amount of time, change person flag to False
                if len(pred) == 0:
                    if not interval_rev: # Check for approx two seconds
                        if person_absence_counter > 60*interval:
                            person_flag = False
                    else:
                        if person_absence_counter > 60/interval:
                            person_flag = False
                    person_absence_counter += 1 # Counter for time check
            
            # Postprocessing
            objs = postprocessing(pred)
    
        with dt[3]:
            # Add tracker ID
            trks = mot_tracker.update(objs) # output format: [[x1,y1,x2,y2,score,cls,id],[x1,y1,x2,y2,score,cls,id],...]
                
        poses = [] # list of poses in the frame (for drawing)
        with dt[4]:
            active_ids = trks[:, 6] # Active IDs that we're currently tracking
            fall_trks = trks[trks[:,5] == FALL] # Pose estimation only for objects with cls == FALL
            for trk in fall_trks:
                # Pose estimation for each bounding box
                x1, y1, x2, y2, id = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[6])
                cropped_img = im[y1:(y2), x1:(x2), :] # Crop the BB from the image
                pose = imgToPose(cropped_img, interpreter_pose, inference_size_pose) # Pose estimation
                angle = poseToAngle(pose) # Feature extraction (9 angles)
                
                # Calculate pose coords in the original img for future reference (drawing)
                x1, y1, x2, y2 = scale_boxes(inference_size, trk[:4][None], (frame_height, frame_width))[0].round()
                pose[0][0][:, 0] = (pose[0][0][:, 0] * (y2-y1) + y1) / frame_height
                pose[0][0][:, 1] = (pose[0][0][:, 1] * (x2-x1) + x1) / frame_width
                poses.append(pose)
                
                # Stack feature data by required number of frames
                if id in objects:
                    if not interval_rev and counter==interval: # if fps is n times higher than 30, append once every n frames.
                        objects[id].append(angle)
                        counter -= 1
                        if counter == 0:
                            counter = interval
                    elif interval_rev: # if fps is n times lower than 30, append n times every frame.
                        for _ in range(interval):
                            objects[id].append(angle)
                else:
                    objects[id] = [angle]
                    
            for id, obj in objects.items():
                # LSTM inference with 60 sequence of features (approx 2 secs)
                if len(obj) == 60:
                    obj = np.array(obj)
                    obj = obj[None]
                    ret = RNN_model(interpreter_LSTM, obj)
                    
                    # Organize the return value into falls dictionary
                    fall = ret[0][0] > 0.5
                    if fall:
                        fall_types = {'Trip': ret[0][1], 'Slip': ret[0][2], 'Intrinsic': ret[0][3]}
                        fall_type = max(fall_types, key=fall_types.get)
                    else:
                        fall_type = 'normal'
                    if fall:
                        head = ret[0][4] > 0.5
                    else:
                        head = 'normal'
                    if id in falls:
                        if not falls[id][0] and fall:
                            falls[id] = [fall, fall_type, head]
                    else:
                        falls[id] = [fall, fall_type, head]
                    print(f"{id}: {fall}, {fall_type}, {head}")
                    
                    objects[id] = [] # Reset feature sequence for ID

        with dt[5]:
            if person_check:
                # Draw person detection alarm to image
                if person_flag:
                    im0 = cv2.putText(im0, "Person Detected", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (124, 64, 230), 2)
                
            # Draw boxes to image
            im0 = append_objs_to_img(im0, inference_size, trks, labels)
            
            # Draw poses to image
            for pose in poses:
                im0 = append_poses_to_img(im0, pose)
            
            # Draw fall ID, type, and head trauma event to image
            i = 20
            for id, fall in falls.items():
                if fall[0]:
                    if fall[2]:
                        im0 = cv2.putText(im0, f"{id}: Fall, Fall Type: {fall[1]}, Head Trauma+", (30, 20 + i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        im0 = cv2.putText(im0, f"{id}: Fall, Fall Type: {fall[1]}, Head Trauma-", (30, 20 + i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    i += 30

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
    if not person_check:
        t = t[:-1]
        print('Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms tracking, %.1fms action rec, and %.1fms drawing per frame' % t)
    else:
        print('Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms tracking, %.1fms action rec, %.1fms drawing, and %.1fms person classification per frame' % t)
    cap.release()
    cv2.destroyAllWindows()

def parse_opt():      
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=ROOT / 'models/ObjectDetection/best-int8.tflite', help='object detection model path')
    parser.add_argument('--source', default=ROOT / 'data/sample.mp4', help='input video path (str) or index of source webcam (0 ~ n)')
    parser.add_argument('--labels', default=ROOT / 'models/ObjectDetection/labels.txt', help='label text file path for object detection')
    parser.add_argument('--conf_thres', default=0.25, type=float, help='classifier score threshold')
    parser.add_argument('--iou_nms_thres', default=0.7, type=float, help='nms iou threshold')
    parser.add_argument('--max_det', default=1000, type=int, help='number of categories with highest score to display')
    parser.add_argument('--save_dir', default=ROOT / 'out', help='output video directory')
    parser.add_argument('--length', default=0, type=int, help='specify the length of video')
    parser.add_argument('--stream', action='store_true', help='stream to window')
    parser.add_argument('--person_check', default=True, help='use pre-detection binary image classification for person')
    parser.add_argument('--model_person', default=ROOT / 'models/PersonCheck/human_detect_quant.tflite', help='person classification model path')
    parser.add_argument('--model_pose', default=ROOT / 'models/PoseEstimation/lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite', help='pose estimation model path')
    parser.add_argument('--model_LSTM', default=ROOT / 'models/LSTM/LSTM-int8.tflite', help='LSTM model path')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))
            
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
