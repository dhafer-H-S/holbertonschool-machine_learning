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

    # def process_outputs(self, outputs, image_size):
    #     boxes = []
    #     box_confidences = []
    #     """
    #     box_confidences: a list of numpy.ndarrays of shape
    #     (grid_height, grid_width, anchor_boxes, 1) containing
    #     the box confidences for each output, respectively
    #     """
    #     box_class_probs = []
    #     """
    #     box_class_probs: a list of numpy.ndarrays of shape
    #     (grid_height, grid_width, anchor_boxes, classes) containing
    #     the boxs class probabilities for each output, respectively
    #     """
    #     for output in outputs:
    #         # Extract boxes
    #         boxes.append(output[..., :4])
    #         """
    #         This line is extracting the first 4 elements along
    #         the last axis of the output array. These represent
    #         the parameters of the bounding box (t_x, t_y, t_w, t_h),
    #         where t_x, t_y are the center coordinates of the box,
    #         and t_w, t_h are the width and height of the box.
    #         """
    #         # Extract box confidences
    #         box_confidences.append(output[..., 4:5])
    #         """
    #         This line is extracting the 5th element along the last
    #         axis of the output array. This represents the confidence
    #         score that the box contains an object.
    #         """
    #         # Extract box class probabilities
    #         box_class_probs.append(output[..., 5:])
    #         """
    #         This line is extracting all elements from the 6th onwards
    #         along the last axis of the output array. These represent
    #         the probabilities of the object belonging to each class.
    #         """

    #     for i, box in enumerate(boxes):
    #         """
    #         a built-in function that allows you to iterate through
    #         a sequence and keep track of the index of each element
    #         """
    #         grid_height, grid_width, anchor_boxes, _ = box.shape
    #         """
    #         this line extract the dimension of the grid and the number
    #         of anchor boxes from the shape of the box
    #         """
    #         # Update box dimensions
    #         box[..., :2] = 1 / (1 + np.exp(-(box[..., :2])))
    #         """
    #         This line applies the sigmoid function to the first two elements
    #         of the last axis of the box (t_x, t_y). This transforms these
    #         elements from the output scale to the range (0, 1), representing
    #         the center of the bounding box relative to the grid cell.
    #         """
    #         box[..., 2:] = np.exp(box[..., 2:])
    #         """
    #         This line applies the exponential function to the last two elements
    #         of the last axis of the box (t_w, t_h). This transforms these elements
    #         from the output scale to be the width and height of the bounding box
    #         relative to the anchor box.
    #         """
    #         box[..., :2] *= self.anchors[i]
    #         """
    #         (which represent the center coordinates of the box)
    #         """
    #         box[..., 2:] *= self.anchors[i]
    #         """
    #         (which represent the width and height of the box)
    #         """
    #         # Scale boxes back to original image size
    #         """
    #         image_size : containing the images original size
    #         [image_height, image_width]
    #         """
    #         box[..., 0] *= image_size[1] / grid_width
    #         """
    #         This line is selecting the first element of
    #         the last axis of the box array
    #         (which represents the x-coordinate of the center of the box)
    #         and scaling it by the ratio of the original image width to
    #         the grid width.
    #         """
    #         box[..., 1] *= image_size[0] / grid_height
    #         """
    #         This line is selecting the second element of the last axis of
    #         the box array
    #         (which represents the y-coordinate of the center of the box)
    #         and scaling it by the ratio of the original image height to
    #         the grid height.
    #         """
    #         box[..., 2] *= image_size[1] / grid_width
    #         """
    #         This line is selecting the third element of the last axis of
    #         the box array (which represents the width of the box) and scaling
    #         it by the ratio of the original image width to the grid width.
    #         """
    #         box[..., 3] *= image_size[0] / grid_height
    #         """
    #         This line is selecting the fourth element of the last axis of the
    #         box array (which represents the height of the box) and scaling it
    #         by the ratio of the original image height to the grid height.
    #         """
    #     return boxes, box_confidences, box_class_probs
    def process_outputs(self, outputs, image_size):
        """ Process Outputs """
        boxes = []
        box_confidences = []
        box_class_probs = []
        for output in outputs:
            boxes.append(output[..., :4])
            box_confidences.append(output[..., 4:5])
            box_class_probs.append(output[..., 5:])

        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            # Calculate coordinates (x, y) and width/height
            box[..., :2] = 1 / (1 + np.exp(-(box[..., :2])))
            box[..., 2:] = np.exp(box[..., 2:])
            box[..., :2] = box[..., :2] * self.anchors[i]
            box[..., 2:] = box[..., 2:] * self.anchors[i]

            # Calculate coordinates relative to the image size
            box[..., 0] = (box[..., 0] + np.arange(grid_width).reshape(1, grid_width, 1)) / grid_width
            box[..., 1] = (box[..., 1] + np.arange(grid_height).reshape(grid_height, 1, 1)) / grid_height
            box[..., 2] *= image_size[1]
            box[..., 3] *= image_size[0]

            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            box[..., 0] = box[..., 0] - box[..., 2] / 2
            box[..., 1] = box[..., 1] - box[..., 3] / 2
            box[..., 2] = box[..., 0] + box[..., 2]
            box[..., 3] = box[..., 1] + box[..., 3]

        return boxes, box_confidences, box_class_probs
    