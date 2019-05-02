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



import numpy as np
import cv2


def resize_image(height, width, image, interpolation=None):
    """A function that resizes a provided picture.
    Inputs: width and height to resize to
            image to resize
    Outputs: input_image_resized"""

    if interpolation is None:
        if str(image.dtype).startswith(("int", "bool")):
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR  # default

    image_resized = cv2.resize(image, dsize=(width, height), interpolation=interpolation)
    return image_resized


class ResizeAndCrop(object):
    """Resize And Crop to process and back"""

    def __init__(self, hypes, original_image_size):
        """A function that provides the indices to start and stop cropping the picture at.
            Inputs: hypes file to get crop parameters,
                    original_image_size
            Define: crop_y_from, crop_y_to, crop_x_from, crop_x_to, processing_image_size"""

        def _get(h, field):
            """ Get field from h if such present, else return False"""
            if field in h.keys():
                return h[field]
            return False

        if 'jitter' in hypes.keys():
            h_ = hypes['jitter']
        else:
            h_ = hypes

        # ------------- resize_image -----------------------
        self.resize_image = _get(h_, 'reseize_image') or _get(h_, 'resize_image')
        if  self.resize_image:
            inter_image_size = (h_['image_width'], h_['image_height'])
        else:
            inter_image_size = original_image_size[:2]  # float

        self.inter_image_size = inter_image_size

        # ------------- crop_for_processing -----------------------
        self.crop_for_processing = _get(h_, 'crop_for_processing')
        if self.crop_for_processing:
            if 'crop_x_from' in h_.keys():
                self.crop_x_from = int(inter_image_size[0] * h_['crop_x_from'])
            else:
                self.crop_x_from = int(0)

            if 'crop_x_to' in hypes['jitter'].keys():
                self.crop_x_to = int(inter_image_size[0] * h_['crop_x_to'])
            else:
                self.crop_x_to = int(inter_image_size[0])

            if 'crop_y_from' in h_.keys():
                self.crop_y_from = int(inter_image_size[1] * h_['crop_y_from'])
            else:
                self.crop_y_from = int(0)

            if 'crop_y_to' in h_.keys():
                self.crop_y_to = int(inter_image_size[1] * h_['crop_y_to'])
            else:
                self.crop_y_to = int(inter_image_size[1])

            self.processing_image_size = (
                self.crop_x_to - self.crop_x_from, self.crop_y_to - self.crop_y_from)

        else:
            self.processing_image_size = inter_image_size

    def preprocess_image(self, image, image_uncropped=None):
        """A function that does all of the image preprocessing
        Inputs: image to process
                image_uncropped empty image for postprocessing (allocated if is None)                
        Outputs: preprocessed image, image_uncropped"""

        preprocessed_image = image

        # Resize the image
        if self.resize_image:
            #self.inter_image_size = (h_['image_width'], h_['image_height'])
            preprocessed_image = resize_image(self.inter_image_size[1], # -> image_height
                                              self.inter_image_size[0], # -> image_width
                                              image)

        # Crop the image
        if self.crop_for_processing:
            if image_uncropped is None:
                image_uncropped = np.zeros(
                    (preprocessed_image.shape[0], preprocessed_image.shape[1]))

            preprocessed_image = preprocessed_image[self.crop_y_from:self.crop_y_to, self.crop_x_from:self.crop_x_to]
        
        return preprocessed_image, image_uncropped

    def postprocess_image(self, image,
                          output_image_uncropped,
                          resulting_image_for_shape, # image shape to resize back, only shape is used
                          filter_data=None):
        """A function that does all of the image preprocessing for KittiSeg
        Inputs: image to process
                output_image_uncropped empty image for postprocessing                
        Outputs: way_prediction"""
        
        #Insert the cropped image into the full sized image
        if self.crop_for_processing:
            output_image_uncropped[self.crop_y_from:self.crop_y_to, self.crop_x_from:self.crop_x_to] = image
            image = output_image_uncropped

        #Resize the image to its original size
        if self.resize_image:           
            image = resize_image(resulting_image_for_shape.shape[0], resulting_image_for_shape.shape[1], image)

        # Accept all pixel with conf >= threshold as positive prediction
        # This creates a `hard` prediction result for class street
        if str(image.dtype).startswith("float"):
            if filter_data is None:
                filter_data = 0.5
            way_prediction = image > filter_data
        elif str(image.dtype).startswith("int"):
            way_prediction = image.copy()
        elif str(image.dtype).startswith("bool"):
            way_prediction = image.copy()
        else:
            print(image.dtype)
            assert str(image.dtype).startswith(("float", "int", "bool"))
        return way_prediction
