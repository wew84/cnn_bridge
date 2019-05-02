#!/usr/bin/python
# BSD 3-Clause License

# Copyright (c) 2019, Noam C. Golombek
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import logging
import colorsys
import numpy as np
import cv2

import rospy
from cnn_bridge.srv import *


class DrawingTools(object):
    """Class for image processing"""

    def __init__(self, mode, opacity=0.6):
        self.metadata = None
        self.opacity = opacity
        self.mode = mode

    def get_cnn_metadata(self, mode):
        rospy.wait_for_service(rospy.get_name() + '/get_cnn_metadata')
        try:
            get_metadata = None
            if mode == 'segmentation':
                get_metadata = rospy.ServiceProxy(
                    rospy.get_name() + '/get_cnn_metadata', getSegmentationMetadata)
            elif mode == 'detection':
                get_metadata = rospy.ServiceProxy(
                    rospy.get_name() + '/get_cnn_metadata', getDetectionMetadata)
            self.metadata = get_metadata()
        except rospy.ServiceException, e:
            rospy.logerr("Could not get metadata: %e", e)

    def overlay_segmentation(self, image_to_overlay, segmentation):
        """
        Overlay an image with a segmentation result for multiple classes.

        Parameters
        ----------
        image_to_show : numpy.array
            An image of shape [width, height, 3]
        segmentation : numpy.array
            Segmentation of shape [width, height]

        Returns
        -------
        cv2.image
            The image overlayed with the segmentation
        """

        # Check that the image has three channels
        assert image_to_overlay.shape[2] == 3
        # Check that the segmentation has a single channel
        assert segmentation.ndim == 2 or segmentation.shape[2] == 1

        # This does not happen in the __init__ because the class
        # is initialized before the service is published
        if self.metadata is None:
            self.get_cnn_metadata(self.mode)

        # Squeeze the segmentation down if it is too large
        if segmentation.ndim == 3:
            segmentation = segmentation.squeeze(axis=2)

        # Find the unique classes in the image
        classes = np.unique(segmentation)
        # Create an image to overlay on
        img = np.zeros(np.shape(image_to_overlay), dtype=np.uint8)

        # Loop the classes and create an image to overlay
        # TODO Improve efficiency
        for classification in self.metadata.class_metadata:
            id = classification.classification.data
            color = (classification.color.r,
                     classification.color.g, classification.color.b)
            if id in classes:
                img[segmentation == id] = color

        # Overlay the image
        image_to_show = cv2.addWeighted(
            image_to_overlay, self.opacity, img, 1 - self.opacity, 0)

        return image_to_show

    def draw_detection(self, image, boxes, scores, labels, detection_size,
                       font=cv2.FONT_HERSHEY_SIMPLEX):
        """
        :param boxes, shape of  [num, 4]
        :param scores, shape of [num, ]
        :param labels, shape of [num, ]
        :param image,
        :param classes, the return list from the function `read_coco_names`
        """
        # This does not happen in the __init__ because the class
        # is initialized before the service is published
        if self.metadata is None:
            self.get_cnn_metadata(self.mode)

        # If no boxes return original image
        if boxes is None:
            return image
        a = len(self.metadata.class_names)*1.0
        hsv_tuples = [(x / a, 0.9, 1.0)
                      for x in range(len(self.metadata.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        # Loop the detections and draw the bounding boxes
        for i in range(len(labels)):
            # Get current data
            bbox, score, label = boxes[i], scores[i], self.metadata.class_names[labels[i]]
            bbox_text = "%s %.2f" % (label.data, score)

            # Parameters setup
            text_size = cv2.getTextSize(bbox_text, font, 1, 2)
            # Ration between detection size and image size
            ratio = np.array((self.metadata.image_width, self.metadata.image_height),
                             dtype=float) / np.array(detection_size, dtype=float)
            bbox = list((bbox.reshape(2, 2) * ratio).reshape(-1))

            # Draw bounding box
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(
                bbox[2]), int(bbox[3])), colors[labels[i]], thickness=3)
            text_origin = bbox[:2]-np.array([0, text_size[0][1]])
            cv2.rectangle(image, (int(text_origin[0]), int(text_origin[1])), (int(
                text_origin[0]+text_size[0][0]), int(text_origin[1]+text_size[0][1])), colors[labels[i]], thickness=-1)
            cv2.putText(image, bbox_text, (int(bbox[0]), int(
                bbox[1])), font, 1, (0, 0, 0), thickness=2)

        return image


def read_coco_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
