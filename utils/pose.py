import cv2
import numpy as np
import re
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
M = (255, 0, 255)
C = (0, 255, 255)
Y = (255, 255, 0)
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): M,
    (0, 2): C,
    (1, 3): M,
    (2, 4): C,
    (0, 5): M,
    (0, 6): C,
    (5, 7): M,
    (7, 9): M,
    (6, 8): C,
    (8, 10): C,
    (5, 6): Y,
    (5, 11): M,
    (6, 12): C,
    (11, 12): Y,
    (11, 13): M,
    (13, 15): M,
    (12, 14): C,
    (14, 16): C
}

def append_poses_to_img(
        im, keypoints_with_scores, crop_region=None, close_figure=False,
        output_image_height=None):
    """Draws the keypoint predictions on image.

    Args:
        image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
        A numpy array with shape [out_height, out_width, channel] representing the
        image overlaid with keypoint predictions.
    """
    height, width, ch = im.shape
    
    (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(keypoints_with_scores, height, width)

    for edge, color in zip(keypoint_edges, edge_colors):
        cv2.line(im, tuple(edge[0]), tuple(edge[1]), color, 2)
    for loc in keypoint_locs:
        cv2.circle(im, tuple(loc), 3, (255, 20, 147), -1)

    # im = np.ascontiguousarray(im, dtype=np.uint8)
    
    return im

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                        height,
                                        width,
                                        keypoint_threshold=0.11):
    """Returns high confidence keypoints and edges for visualization.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        height: height of the image in pixels.
        width: width of the image in pixels.
        keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
        A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy.astype(np.int32), edges_xy.astype(np.int32), edge_colors

def movenet(input_image, interpreter):
    """Runs detection on an input image.

    Args:
    input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
    A [1, 1, 17, 3] float numpy array representing the predicted keypoint
    coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def imgToPose(im, interpreter):
    input_size = 192 # (192, 192)
    im = im.transpose((1, 2, 0)) # chw to hwc
        
    # image resize to input_size
    im = tf.expand_dims(im, axis=0)
    im = tf.image.resize_with_pad(im, input_size, input_size)
    
    # Process the im to get pose coordinates using your model
    pose_coordinates = movenet(im, interpreter)  # (1, 1, 17, 3)
    
    # Write pose coordinates to the output file
    return pose_coordinates


def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    if np.all(v1==0) or np.all(v2==0):
        return 0.5
        
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    cosine_angle = dot_product / magnitude_product
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg / 180

def poseToAngle(coords):
    tmp = []
    for coord in coords[0][0]:
        tmp.append(coord[0])
        tmp.append(coord[1])
    
    coords = [float(x) for x in tmp]
    # 0: nose, (left-right) 1-2: eye, 3-4: ear, 5-6: shoulder, 7-8: elbow, 9-10: wrist, 11-12: hip, 13-14: knee, 15-16: ankle
    angles = []
    
    landmarks = {
        'nose': [coords[0], coords[1]],
        'left_shoulder': [coords[10], coords[11]],
        'right_shoulder': [coords[12], coords[13]],
        'left_elbow': [coords[14], coords[15]],
        'right_elbow': [coords[16], coords[17]],
        'left_wrist': [coords[18], coords[19]],
        'right_wrist': [coords[20], coords[21]],
        'left_hip': [coords[22], coords[23]],
        'right_hip': [coords[24], coords[25]],
        'left_knee': [coords[26], coords[27]],
        'right_knee': [coords[28], coords[29]],
        'left_ankle': [coords[30], coords[31]],
        'right_ankle': [coords[32], coords[33]]
    }
    
    # Calculate angles for each specified joint
    angles.append(calculate_angle(landmarks['left_elbow'], landmarks['left_shoulder'], landmarks['nose']))
    angles.append(calculate_angle(landmarks['right_elbow'], landmarks['right_shoulder'], landmarks['nose']))
    angles.append(calculate_angle(landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']))
    angles.append(calculate_angle(landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']))
    angles.append(calculate_angle(landmarks['left_shoulder'], landmarks['left_hip'], landmarks['left_knee']))
    angles.append(calculate_angle(landmarks['right_shoulder'], landmarks['right_hip'], landmarks['right_knee']))
    angles.append(calculate_angle(landmarks['left_hip'], landmarks['left_knee'], landmarks['left_ankle']))
    angles.append(calculate_angle(landmarks['right_hip'], landmarks['right_knee'], landmarks['right_ankle']))
    angles.append((calculate_angle(landmarks['nose'], landmarks['left_shoulder'], landmarks['left_hip']) +
                  calculate_angle(landmarks['nose'], landmarks['right_shoulder'], landmarks['right_hip'])) / 2)

    return angles


# def draw_prediction_on_image(
#         image, keypoints_with_scores, crop_region=None, close_figure=False,
#         output_image_height=None):
#     """Draws the keypoint predictions on image.

#     Args:
#         image: A numpy array with shape [height, width, channel] representing the
#         pixel values of the input image.
#         keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
#         the keypoint coordinates and scores returned from the MoveNet model.
#         crop_region: A dictionary that defines the coordinates of the bounding box
#         of the crop region in normalized coordinates (see the init_crop_region
#         function below for more detail). If provided, this function will also
#         draw the bounding box on the image.
#         output_image_height: An integer indicating the height of the output image.
#         Note that the image aspect ratio will be the same as the input image.

#     Returns:
#         A numpy array with shape [out_height, out_width, channel] representing the
#         image overlaid with keypoint predictions.
#     """
#     height, width, ch = image.shape
#     aspect_ratio = float(width) / height
#     fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
#     # To remove the huge white borders
#     fig.tight_layout(pad=0)
#     ax.margins(0)
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     plt.axis('off')

#     im = ax.imshow(image)
#     line_segments = LineCollection([], linewidths=(4), linestyle='solid')
#     ax.add_collection(line_segments)
#     # Turn off tick labels
#     scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

#     (keypoint_locs, keypoint_edges,
#     edge_colors) = _keypoints_and_edges_for_display(
#         keypoints_with_scores, height, width)

#     line_segments.set_segments(keypoint_edges)
#     line_segments.set_color(edge_colors)
#     if keypoint_edges.shape[0]:
#         line_segments.set_segments(keypoint_edges)
#         line_segments.set_color(edge_colors)
#     if keypoint_locs.shape[0]:
#         scat.set_offsets(keypoint_locs)

#     fig.canvas.draw()
#     image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     image_from_plot = image_from_plot.reshape(
#         fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     if output_image_height is not None:
#         output_image_width = int(output_image_height / height * width)
#         image_from_plot = cv2.resize(
#             image_from_plot, dsize=(output_image_width, output_image_height),
#                 interpolation=cv2.INTER_CUBIC)
#     return image_from_plot
