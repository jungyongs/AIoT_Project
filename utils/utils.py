""" Utility functions for main.py

"""
import cv2
import numpy as np
import re
import tensorflow as tf
import time

def read_label_file(file_path):
    """Reads labels from a text file and returns it as a dictionary.

    This function supports label files with the following formats:

    + Each line contains id and description separated by colon or space.
        Example: ``0:cat`` or ``0 cat``.
    + Each line contains a description only. The returned label id's are based on
        the row number.

    Args:
        file_path (str): path to the label file.

    Returns:
        Dict of (int, string) which maps label id to description.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ret = {}
    for row_number, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
        if len(pair) == 2 and pair[0].strip().isdigit():
           ret[int(pair[0])] = pair[1].strip()
        else:
           ret[row_number] = content.strip()
    return ret

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios 
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, ratio, (dw, dh)


def inference_int8(interpreter, im):
    input = interpreter.get_input_details()[0]
    output = interpreter.get_output_details()[0]
    scale, zero_point = input['quantization']
    
    im = im.astype(np.float32)  # uint8 to fp16/32
    im /= 255
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    b, ch, h, w = im.shape
    
    if True:
        im = im.transpose(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)
        
    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
    interpreter.set_tensor(input['index'], im)
    interpreter.invoke()
    
    x = interpreter.get_tensor(output['index'])
    scale, zero_point = output['quantization']
    x = (x.astype(np.float32) - zero_point) * scale  # re-scale

    # y = [x if isinstance(x, np.ndarray) else x.numpy()]
    y = x[None]
    y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels
    
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    # time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    # redundant = True  # require redundant detections

    # t = time.time()
    output = [np.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)

        # Detections matrix nx6 (xyxy, conf, cls)
        # best class only
        conf = x[:, 5:].max(axis=1, keepdims=True)
        j = x[:, 5:].argmax(axis=1, keepdims=True)
        x = np.concatenate((box, conf, j), axis=1)[conf.flatten() > conf_thres]  # j.astpye(np.float32)?

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS - agnostic NMS!
        boxes, scores = x[:, :4], x[:, 4]  # boxes (agnostic), scores
        # i = _nms(boxes, scores, iou_thres)  # Manual implementation of NMS
        i = tf.image.non_max_suppression(boxes, scores, max_output_size=max_det, iou_threshold=iou_thres)
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
        #     break  # time limit exceeded

    return output

def _nms(dets, scores, iou_threshold):
   # dets: xyxy
   # scores: confidence score

   # 1. Get the index of bbox sorted by confidence score
   order = scores.squeeze().argsort()[::-1]

   # 2. Calculate the bbox overlap
   keep = []
   while order.size > 0:
       i = order[0]
       keep.append(i)

       if order.size == 1:
           break

       ovr = box_iou(dets[i:i+1], dets[order[1:]])
       inds = np.nonzero(ovr <= iou_threshold)[0] # squeeze
       if inds.size == 0:
           break
       order = order[inds + 1]
   return np.array(keep)

def box_iou(boxes1, boxes2):
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou

def _box_inter_union(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.maximum(rb - lt, 0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union

def box_area(boxes):
    # boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def postprocessing(prediction):
    # prediction shape: batch, num_boxes, xyxy+conf+cls
    output = prediction[0]
    
    # xyxy, conf, cls, count
    detection_box = output[..., :4]
    cls_conf = output[..., 4:5]
    cls_id = output[..., 5:]
    count = output.shape[0]
    
    if count == 0:
        return np.empty((0, 6))
    # 
    output = []
    for i in range(count):   
        x1, y1, x2, y2 = detection_box[i]
        score = float(cls_conf[i])
        id = float(cls_id[i])
        output.append(np.array([x1, y1, x2, y2, score, id]))
        
    return np.stack(output)

def append_objs_to_img(im, inference_size, trks, labels, notrack=False):
                    # Rescale boxes from img_size to im0 size
    trks[:, :4] = scale_boxes(inference_size, trks[:, :4], im.shape).round()
    # h, w, ch = im.shape
    # sx, sy = w / inference_size[0], h / inference_size[1]
    
    for trk in trks:
        # x0, y0 = int(sx * trk[0]), int(sy * trk[1])
        # x1, y1 = int(sx * trk[2]), int(sy * trk[3])
        x0, y0 = int(trk[0]), int(trk[1])
        x1, y1 = int(trk[2]), int(trk[3])
        percent = int(100 * trk[4])
        cls = labels.get(trk[5], trk[5])
        if not notrack:
            id = int(trk[6])
            label = f'{percent}% {cls} ID:{id}'
        else:
            label = f'{percent}% {cls}'
            
        if cls == 'fall':
            color = (0, 0, 255)
        else:
            color = (55, 255, 55)
        im = cv2.rectangle(im, (x0, y0), (x1, y1), color, 2)
        im = cv2.putText(im, label, (x0+2, y0+12),
                             cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 55, 55), 1)
            
    return im

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

# def _upcast(t):
#     # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
#     if t.is_floating_point():
#         return t if t.dtype in (torch.float32, torch.float64) else t.float()
#     else:
#         return t if t.dtype in (torch.int32, torch.int64) else t.int()
    
class Timer():
    def __init__(self, t=0.0):
        self.t = t

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        return time.time()