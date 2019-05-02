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


class YoloDetector(object):
    """Class to load deeplab model and run inference."""

    def __init__(self, model_dir, original_image_size, tensor_io, runCPU, gpu_percent=1):
        self.hypes = load_hypes(model_dir)
        self.input_tensor = tensor_io['input_tensor']
        self.output_tensor = tensor_io['output_tensor']
        frozen_graph_path = self.hypes['frozen_graph_path']
        rospy.logwarn("Weights to load: " + frozen_graph_path)
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

        self.output_tensor = [self.graph.get_tensor_by_name(tensor) for tensor in self.output_tensor ]
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

        return self.run_processed_image(image_for_proc)

    def run_processed_image(self, image):
        """Runs inference on a single image.

        Args:
        image: A PIL.Image object, raw input image.

        Returns:
        resized_image: RGB image resized from original input image.
        detected_classes: Segmentation map of `resized_image`.
        """
        time__tf_start = timeit.default_timer()
        # ---------------------------------
        boxes, scores, classes = self.sess.run(
            self.output_tensor,
            feed_dict={self.input_tensor: [np.asarray(image)]})
        # ---------------------------------
        time__tf = timeit.default_timer() - time__tf_start

        detected_classes = {}
        detected_classes['boxes'] = boxes
        detected_classes['scores'] = scores
        detected_classes['classes'] = classes
        return detected_classes, time__tf
