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
import tensorflow as tf
import timeit
import rospy


from tools import ResizeAndCrop


def load_hypes(model_dir):
    import os
    import json
    if os.path.isdir(model_dir):
        hypes_name = os.path.join(model_dir, "deeplab.json")
    else:
        hypes_name = model_dir

    with open(hypes_name, 'r') as f:
        return json.load(f)


class DeepLabSegmenter(object):
    """Class to load deeplab model and run inference."""

    def __init__(self, model_dir, original_image_size, tensor_io, runCPU, gpu_percent=1):
        self.hypes = load_hypes(model_dir)
        self.input_tensor = tensor_io["input_tensor"]
        self.output_tensor = tensor_io["output_tensor"]
        frozen_graph_path = self.hypes['frozen_graph_path']
        rospy.logwarn("Deeplab to load: " + frozen_graph_path)
        # ---------------------------------------------------------------------
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from given path.
        with open(frozen_graph_path, 'rb') as file_handle:
            graph_def = tf.GraphDef.FromString(file_handle.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in given path.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_percent
        self.sess = tf.Session(graph=self.graph, config=config)

        # ---------------------------------------------------------------------
        if "input_image_size" in self.hypes.keys():
            self.input_image_size = self.hypes["input_image_size"]
        else:
            self.input_image_size = (641, 361)

        self.tools = ResizeAndCrop(self.hypes, original_image_size)
        self.output_image_uncropped = None

    def run_model_on_image(self, image):
        """A function that sets up and runs an image through KittiSeg
        Input: Image to process
        Output: way_prediction, time_tf"""

        image_for_proc, self.output_image_uncropped = self.tools.preprocess_image(
            image, self.output_image_uncropped)
        
        # height, width, channels = image.shape
        # resize_ratio = 1.0 * self.input_image_size / max(width, height)
        # target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # resized_image = image.convert('RGB').resize(
        #     target_size, Image.ANTIALIAS)

        output_image, time_tf = self.run_processed_image(image_for_proc)

        # -----------------------------------------------------------------
        # Plot confidences as red-blue overlay
        # rb_image = seg.make_overlay(image, output_image)

        return self.tools.postprocess_image(
            output_image, self.output_image_uncropped, image, self.hypes["selected_classes"]), time_tf

    def run_processed_image(self, image):
        """Runs inference on a single image.

        Args:
        image: A PIL.Image object, raw input image.

        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """
        time__tf_start = timeit.default_timer()
        # ---------------------------------
        batch_seg_map = self.sess.run(
            self.output_tensor,
            feed_dict={self.input_tensor: [np.asarray(image)]})
        # ---------------------------------
        time__tf = timeit.default_timer() - time__tf_start

        seg_map = batch_seg_map[0]
        return seg_map, time__tf

# def create_pascal_label_colormap():
#     """Creates a label colormap used in PASCAL VOC segmentation benchmark.

#     Returns:
#         A Colormap for visualizing segmentation results.
#     """
#     colormap = np.zeros((256, 3), dtype=int)
#     ind = np.arange(256, dtype=int)

#     for shift in reversed(range(8)):
#         for channel in range(3):
#             colormap[:, channel] |= ((ind >> channel) & 1) << shift
#         ind >>= 3

#     return colormap

# def label_to_color_image(label):
#     """Adds color defined by the dataset colormap to the label.

#     Args:
#         label: A 2D array with integer type, storing the segmentation label.

#     Returns:
#         result: A 2D array with floating type. The element of the array
#         is the color indexed by the corresponding element in the input label
#         to the PASCAL color map.

#     Raises:
#         ValueError: If label is not of rank 2 or its value is larger than color
#         map maximum entry.
#     """
#     if label.ndim != 2:
#         raise ValueError('Expect 2-D input label')

#     colormap = create_pascal_label_colormap()

#     if np.max(label) >= len(colormap):
#         raise ValueError('label value too large.')

#     return colormap[label]
