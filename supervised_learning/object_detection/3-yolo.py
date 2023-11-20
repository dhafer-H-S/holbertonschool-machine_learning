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
            """
            sigmoid that applies the sigmoid activation function
            element-wise. It's later used to transform certain values
            in the YOLOv3 output
            """
            return 1 / (1 + np.exp(-x))

        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size
        """
        These lines initialize empty lists
        (boxes, box_confidences, box_class_probs)
        to store bounding box information, confidence scores,
        and class probabilities. img_h and img_w are unpacked
        from image_size.
        """
        i = 0
        for output in outputs:
            grid_h, grid_w, nb_box, _ = output.shape
            """
            This line unpacks the shape of the current output,
            representing the grid dimensions and the number of bounding boxes
            """
            box_conf = sigmoid(output[:, :, :, 4:5])
            box_prob = sigmoid(output[:, :, :, 5:])
            box_confidences.append(box_conf)
            box_class_probs.append(box_prob)
            """
            Here, confidence scores (box_conf) and class probabilities
            (box_prob)are extracted from the output and then appended
            to their respective lists.
            """
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]
            """
            These lines extract the raw predicted values for bounding box
            coordinates and dimensions.
            """
            c_x = np.arange(grid_w)
            c_x = np.tile(c_x, grid_h)
            c_x = c_x.reshape(grid_h, grid_w, 1)
            c_y = np.arange(grid_h)
            c_y = np.tile(c_y, grid_w)
            c_y = c_y.reshape(1, grid_h, grid_w).T
            """
            These lines create arrays c_x and c_y representing
            the coordinates of the grid cells.
            """
            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]
            """
            These lines extract the width (p_w) and height (p_h)
            of the anchor boxes for the current scale.
            """
            b_x = (sigmoid(t_x) + c_x)
            b_y = (sigmoid(t_y) + c_y)
            b_w = (np.exp(t_w) * p_w)
            b_h = (np.exp(t_h) * p_h)
            """
            These lines extract the width (p_w) and height (p_h)
            of the anchor boxes for the current scale.
            """
            b_x = b_x / grid_w
            b_y = b_y / grid_h
            b_w = b_w / self.model.input.shape[1].value
            b_h = b_h / self.model.input.shape[2].value
            """
            These lines normalize the bounding box coordinates and dimensions
            """
            x1 = (b_x - b_w / 2) * img_w
            y1 = (b_y - b_h / 2) * img_h
            x2 = (b_x + b_w / 2) * img_w
            y2 = (b_y + b_h / 2) * img_h
            """
            These lines compute the absolute coordinates of the bounding boxes
            in the original image space.
            """
            box = np.zeros((grid_h, grid_w, nb_box, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)
            i += 1
            """
            These lines organize the bounding box coordinates into a 4D NumPy
            array (box) and append it to the boxes list.
            """
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes based on class confidence score."""
        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []

        for i in range(len(boxes)):
            box_scores = box_confidences[i] * box_class_probs[i]
            box_classes = np.argmax(box_scores, axis=-1)
            box_class_scores = np.max(box_scores, axis=-1)
            filtering_mask = box_class_scores >= self.class_t

            filtered_boxes += boxes[i][filtering_mask].tolist()
            filtered_scores += box_class_scores[filtering_mask].tolist()
            filtered_classes += box_classes[filtering_mask].tolist()

        filtered_boxes = np.array(filtered_boxes)
        filtered_scores = np.array(filtered_scores)
        filtered_classes = np.array(filtered_classes)

        return filtered_boxes, filtered_classes, filtered_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non-max suppression"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for i in np.unique(box_classes):
            """This line iterates over the unique predicted classes"""
            indexes = np.where(box_classes == i)
            """This line finds the predicted boxes and their associated
            class confidence scores for the current class"""
            boxes_of_class = filtered_boxes[indexes]
            classes_of_class = box_classes[indexes]
            scores_of_class = box_scores[indexes]
            """This line computes the bounding box coordinates
            for the current class"""
            x1 = boxes_of_class[:, 0]
            y1 = boxes_of_class[:, 1]
            x2 = boxes_of_class[:, 2]
            y2 = boxes_of_class[:, 3]
            """This line computes the area of the bounding boxes"""
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            """This line sorts the bounding boxes by their bottom-right
            y-coordinate"""
            idxs = scores_of_class.argsort()[::-1]
            """This line initializes a list to store the final bounding
            boxes, class confidence scores, and predicted classes"""
            pick = []
            """This line loops over the indexes of the bounding boxes"""
            while len(idxs) > 0:
                """This line finds the bounding box with the largest
                confidence score, and appends it to the pick list"""
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)
                """This line finds the largest (x, y) coordinates for
                the start of the bounding box and the smallest (x, y)
                coordinates for the end of the bounding box"""
                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])
                """This line computes the width and height of the bounding
                box"""
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                """This line computes the ratio of overlap"""
                overlap = (w * h) / area[idxs[:last]]
                """This line deletes all indexes from the index list
                that have overlap greater than the provided overlap
                threshold"""
                idxs = np.delete(idxs, np.concatenate(
                    ([last], np.where(overlap > self.nms_t)[0])))
            """This line appends the bounding boxes, class confidence
            scores, and predicted classes for the current class to
            their respective lists"""
            box_predictions.append(filtered_boxes[pick])
            predicted_box_classes.append(box_classes[pick])
            predicted_box_scores.append(box_scores[pick])
        """This line returns the bounding boxes, class confidence scores,
        and predicted classes for all classes"""
        return box_predictions, predicted_box_classes, predicted_box_scores
