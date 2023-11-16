#!/usr/bin/env python3
"""this module contains the class Yolo"""
import tensorflow.keras as K
import numpy as np
import os


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor method
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line[:-1] for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process and normalize the output of the YoloV3 model"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        i = 0
        for output in outputs:
            grid_h, grid_w, nb_box, _ = output.shape

            # Calculate box confidence and box class probabilities
            box_conf = 1 / (1 + np.exp(-(output[:, :, :, 4:5])))
            box_prob = 1 / (1 + np.exp(-(output[:, :, :, 5:])))
            box_confidences.append(box_conf)
            box_class_probs.append(box_prob)

            # Calculate box dimensions
            box = output[:, :, :, :4]
            box[..., :2] = 1 / (1 + np.exp(-(box[..., :2])))
            box[..., 2:] = np.exp(box[..., 2:])
            box[..., :2] += np.indices((grid_h, grid_w)).transpose((1, 2, 0))
            box[..., :2] /= (grid_w, grid_h)
            box[..., 2:] *= self.anchors[i]

            # Scale box dimensions back to original image size
            box[..., 0] *= image_size[1] / grid_w
            box[..., 1] *= image_size[0] / grid_h
            box[..., 2] *= image_size[1] / grid_w
            box[..., 3] *= image_size[0] / grid_h

            boxes.append(box)
            i += 1

        return boxes, box_confidences, box_class_probs
