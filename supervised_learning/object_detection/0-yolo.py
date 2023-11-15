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
