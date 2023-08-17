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

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from utils.utils import letterbox, inference_int8, non_max_suppression
from utils.sort import Sort

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
"""Represents a detected object.

.. py:attribute:: id

    The object's class id.

.. py:attribute:: score

    The object's prediction score.

.. py:attribute:: bbox

    A :obj:`BBox` object defining the object's location.
"""

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='./out.mp4')
    parser.add_argument('--length', type=int, default=7)

    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    
    conf_thres=args.threshold,
    iou_thres=0.45,
    max_det=args.top_k

    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera_idx)

    if cap.isOpened():
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    frames = fps * args.length
    if args.input:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("input: ")
    print(interpreter.get_input_details())
    print("output: ")
    print(interpreter.get_output_details()) 
    
    mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    
    while cap.isOpened() and frames > 0:
        ret, frame = cap.read()
        if not ret:
            break
        im0 = frame

        # cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        # cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        # run_inference(interpreter, im.tobytes())
        # objs = get_objects(interpreter, args.threshold)[:args.top_k]
        
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
        trks = mot_tracker.upadate(objs)
        # trackers = mot_tracker.update(objs)
        
        # Draw boxes to image
        im0 = append_objs_to_img(im0, inference_size, trks, labels)

        out.write(im0)
        frames -=1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

def append_objs_to_img(im, inference_size, trks, labels):
    h, w, ch = im.shape
    sx, sy = w / inference_size[0], h / inference_size[1]
    
    for trk in trks:
        x0, y0 = int(sx * trk[0]), int(sy * trk[1])
        x1, y1 = int(sx * trk[2]), int(sy * trk[3])
        percent = int(100 * trk[4])
        cls = labels.get(trk[5], trk[5])
        id = trk[6]
        
        im = cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        im = cv2.putText(im, cls, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        im = cv2.putText(im, id, (x0-50, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    return im
    # for obj in objs:
    #     bbox = obj.bbox.scale(scale_x, scale_y)
    #     x0, y0 = int(bbox.xmin), int(bbox.ymin)
    #     x1, y1 = int(bbox.xmax), int(bbox.ymax)

    #     percent = int(100 * obj.score)
    #     label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

    #     im = cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2)
    #     im = cv2.putText(im, label, (x0, y0+30),
    #                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    # return im

def postprocessing(prediction):
    # prediction shape: batch, num_boxes, xyxy+conf+cls
    output = prediction[0]
    
    # xyxy, conf, cls, count
    detection_box = output[..., :4]
    cls_conf = output[..., 4:5]
    cls_id = output[..., 5:]
    count = output.shape[0]
    
    # 
    output = []
    for i in range(count):   
        x1, y1, x2, y2 = detection_box[i]
        score=float(cls_conf[i])
        id=int(cls_id[i])
        output.append(np.array([x1, y1, x2, y2, score, id]))
        
    return np.concatenate(output)

    # def make(i):
    #     xmin, ymin, xmax, ymax = detection_box[i]
    #     return Object(
    #         id=int(cls_id[i]),
    #         score=float(cls_conf[i]),
    #         bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax,
    #                 ymax=ymax))
    
    # return np.concatenate([make(i) for i in range(count)])

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    def scale(self, sx, sy):
        """Scales the bounding box.

        Args:
            sx (float): Scale factor for the x-axis.
            sy (float): Scale factor for the y-axis.

        Returns:
            A :obj:`BBox` object with the rescaled dimensions.
        """
        return BBox(
            xmin=sx * self.xmin,
            ymin=sy * self.ymin,
            xmax=sx * self.xmax,
            ymax=sy * self.ymax)
    def map(self, f):
        """Maps all box coordinates to a new position using a given function.

        Args:
        f: A function that takes a single coordinate and returns a new one.

        Returns:
        A :obj:`BBox` with the new coordinates.
        """
        return BBox(
            xmin=f(self.xmin),
            ymin=f(self.ymin),
            xmax=f(self.xmax),
            ymax=f(self.ymax))
            
if __name__ == '__main__':
    main()
