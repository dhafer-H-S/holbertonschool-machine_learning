#!/usr/bin/env python3
"""a class that uses the Yolo v3 algorithm to perform object detection"""

import tensorflow.keras as Keras
import numpy as np


class Yolo():
    """in this task we used yolo.h5 file yolo.h5':
    This is the path to the Darknet Keras model. The model is stored in
    the H5 file format, which is a common format for storing large amounts
    of numerical data, and is particularly popular in the machine learning
    field for storing models.
    """
    """coco_classes.txt': This is the path to the file containing the list
    of class names used for the Darknet model.
    The classes are listed in order of index.
    """
    """
    0.6: This is the box score threshold for the initial filtering step.
    Any boxes with a score below this value will be discarded.
    0.5: This is the IOU (Intersection Over Union) threshold for
    non-max suppression. Non-max suppression is a technique used
    to ensure that when multiple bounding boxes are detected for
    the same object, only the one with the highest score is kept.
    anchors: This is a numpy.ndarray containing all of the anchor boxes.
    The shape of this array should be (outputs, anchor_boxes, 2),
    where outputs is the number of outputs (predictions) made by the
    Darknet model, anchor_boxes is the number of anchor boxes used
    for each prediction, and 2 corresponds to [anchor_box_width,
    anchor_box_height].
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ initialize class constructor """
        self.model = Keras.models.load_model(model_path)
        """the path where is the Darknet model is stored"""
        self.class_t = class_t
        """
        a float representing the box score threshold
        for the initial filtiring step
        """
        self.nms_t = nms_t
        """
        a float representing the IOU (Intersection over Union)
        threshold for non-max suppression
        """
        self.anchors = anchors
        """the anchors boxes"""
        with open(classes_path, 'r') as f:
            """class_path is the list of class names used for darknet model"""
            classes = f.read()
            classes = classes.split('\n')
            classes.pop()
            """the use of pop is : removing the last item from
            the classes list. If the last item in your list is
            an empty string or any unwanted value, this line of
            code will remove it."""
        self.class_names = classes

    def process_outputs(self, outputs, image_size):
        """Process and normalize the output of the YoloV3 model"""
        def sigmoid(x):
            """sigmoid function"""
            return 1 / (1 + np.exp(-x))

        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size
        i = 0
        for output in outputs:
            grid_h, grid_w, nb_box, _ = output.shape
            box_conf = sigmoid(output[:, :, :, 4])
            box_conf = box_conf.reshape(-1, 1)
            box_prob = sigmoid(output[:, :, :, 5:])
            box_confidences.append(box_conf)
            box_class_probs.append(box_prob)
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]
            c_x = np.arange(grid_w)
            c_x = np.tile(c_x, grid_h)
            c_x = c_x.reshape(grid_h, grid_w, 1)
            c_y = np.arange(grid_h)
            c_y = np.tile(c_y, grid_w)
            c_y = c_y.reshape(1, grid_h, grid_w).T
            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]
            b_x = (sigmoid(t_x) + c_x)
            b_y = (sigmoid(t_y) + c_y)
            b_w = (np.exp(t_w) * p_w)
            b_h = (np.exp(t_h) * p_h)
            b_x = b_x / grid_w
            b_y = b_y / grid_h
            b_w = b_w / self.model.input.shape[1].value
            b_h = b_h / self.model.input.shape[2].value
            x1 = (b_x - b_w / 2) * img_w
            y1 = (b_y - b_h / 2) * img_h
            x2 = (b_x + b_w / 2) * img_w
            y2 = (b_y + b_h / 2) * img_h
            box = np.zeros((grid_h, grid_w, nb_box, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)
            i += 1
        return boxes, box_confidences, box_class_probs
