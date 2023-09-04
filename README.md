# Fall detection with fall type and and head trauma classification

The Object detection model is adapted from and pretrained using ultralytics YOLOv5 (https://github.com/ultralytics/yolov5)  
The Pose Estimation model is adapted from Tensorflow Hub MoveNet (https://tfhub.dev/google/movenet/singlepose/lightning/4)  
Person check model is trained using MobileNetV2 (https://doi.org/10.48550/arXiv.1801.04381)  
Tracking mechanism is implemented by SORT (https://github.com/abewley/sort, https://doi.org/10.48550/arXiv.1602.00763)  
Online datasets used for traning: Fall Detection Dataset(https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset), UR Fall Detection Dataset(http://fenix.ur.edu.pl/~mkepski/ds/uf.html), Human Detection Dataset(https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)

The program has been implemented and tested in Python 3.8.10.  
If installing the 'lap' package results in an error, you can either install except the package or manually install it by using the uploaded file.

#### Example to Run
```
python detect.py --source vid.mp4
```
