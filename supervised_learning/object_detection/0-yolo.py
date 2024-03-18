#!/usr/bin/env /python3
from tensorflow.keras.models import load_model
"""initialize yolo
   yolo v3 algorithm to perform object detection
"""

class Yolo:
    """class yolo"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        mode_path is were the darknet model is stored
        classes_path is the path where the list of class names used for the darknet model
        class_t is a float representing the box score threshold for the initial filtering step
        nms_t is a float representing the intersection over union
        anchors of output is the number of prediction made by dark net
        anchor_boxes,is the number of anchor boxes used for each prediction
        2 => [anchor_box_width, anchor_box_height]
        """
        self.model = load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        # self.class_names = classes_path
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors