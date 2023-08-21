# app.py
from flask import Flask, flash, redirect, render_template, request, session, Response
import cv2
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.utils import read_label_file
from utils.sort import Sort
from stream import getCameraStream

app = Flask(__name__)

# Set file paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

model=ROOT / 'models/best-int8.tflite'  # model path
labels=ROOT / 'models/labels.txt'  # label txt path
conf_thres=0.25  # confidence threshold
iou_nms_thres=0.7  # NMS IOU threshold
max_det=1000  # maximum detections per image
notrack=False

# Load the tflite model
interpreter = tf.lite.Interpreter(model_path=str(model))
interpreter.allocate_tensors()
inference_size = interpreter.get_input_details()[0]['shape']
inference_size = (inference_size[1], inference_size[2])

# Load the labels text file into dict
labels = read_label_file(labels)

# Initialize capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(f"webcam unsupported")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialize SORT instance    
mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

people = []

def gen_frames():
    while True:
        ret, im0 = cap.read()
        if not ret:
            break
        else:
            im0 = getCameraStream(im0, interpreter, inference_size, labels, mot_tracker, conf_thres, iou_nms_thres, max_det, notrack=False)
            
            ret, buffer = cv2.imencode('.jpg', im0)        
            binary = bytearray(buffer.tobytes())
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + binary + b'\r\n')

@app.route('/')
def index():
    return render_template('video_show.html', width=frame_width, height=frame_height)
    
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)