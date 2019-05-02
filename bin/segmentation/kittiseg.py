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


sys
import logging
import timeit
import tensorflow as tf
import rospy
from tools import ResizeAndCrop

KITTISEG_ROOT_FOLDER = '/home/perfetto/KittiSeg_source' + '/incl/'

sys.path.insert(1, KITTISEG_ROOT_FOLDER)

try:
    # Check whether setup was done correctly
    import tensorvision.utils as tv_utils
    import tensorvision.core as tv_core

except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")

    import os
    if not os.path.exists(KITTISEG_ROOT_FOLDER):
        logging.error("KittiSeg folder " + KITTISEG_ROOT_FOLDER + ' in not found')
    else:
        logging.error("You use " + KITTISEG_ROOT_FOLDER + ' as KittSeg folder')

    import traceback
    EX_TYPE, EX, TB = sys.exc_info()
    traceback.print_tb(TB)

    exit(1)

class KittiSegmenter(object):
    """Class to initialize and run KittiSeg"""
    def __init__(self, logdir, original_image_size, runCPU=False):
        """Initialize KittiSeg:
            Inputs: logdir (Directory to KittiSeg model),
                    runCPU (Optional, if set to True TensorFlow runs on CPU instead of GPU"""

        rospy.loginfo("Using weights found in {}".format(logdir))

        # Loading hyper-parameters from logdir
        self.hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

        rospy.loginfo("self.hypes loaded successfully.")

        # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
        modules = tv_utils.load_modules_from_logdir(logdir)
        rospy.loginfo("Modules loaded successfully. Starting to build tf graph.")

        # Create tf graph and build module.
        with tf.Graph().as_default():
            tf_device = None
            if runCPU:
                tf_device = '/cpu:0'

            with tf.device(tf_device):
                # Create placeholder for input
                self.tf_image_pl = tf.placeholder(tf.float32)
                tf_image = tf.expand_dims(self.tf_image_pl, 0)
                tf_image.set_shape([1, None, None, 3])

                # build Tensorflow graph using the model from logdir
                self.tf_prediction = tv_core.build_inference_graph(
                    self.hypes, modules, image=tf_image)

                rospy.loginfo("Graph build successfully.")

                # Create a session for running Ops on the Graph.

                # Config to turn on JIT compilation - 3 rows instead
                # of simple tf_session = tf.Session()
                #
                # config = tf.ConfigProto()
                # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
                # tf_session = tf.Session(config=config)

                # tf_session=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

                # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
                # tf_session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

                self.tf_session = tf.Session()
                tf_saver = tf.train.Saver()

                # Load weights from logdir
                tv_core.load_weights(logdir, self.tf_session, tf_saver)

                rospy.loginfo("Weights loaded successfully.")

        self.tools = ResizeAndCrop(self.hypes, original_image_size)
        self.output_image_uncropped = None

    def run_processed_image(self, image):
        """A function that runs an image through KittiSeg
        Input: Image to process
        Output: tagged_image, time_tf"""
        feed = {self.tf_image_pl: image}
        softmax = self.tf_prediction['softmax']

        time__tf_start = timeit.default_timer()
        # ---------------------------------
        output = self.tf_session.run([softmax], feed_dict=feed)
        # ---------------------------------
        time__tf = timeit.default_timer() - time__tf_start

        # Reshape output from flat vector to 2D Image
        shape = image.shape
        return output[0][:, 1].reshape(shape[0], shape[1]), time__tf

    def run_model_on_image(self, image):
        """A function that sets up and runs an image through KittiSeg
        Input: Image to process
        Output: way_prediction, time_tf"""

        image_for_proc, self.output_image_uncropped = self.tools.preprocess_image(
            image, self.output_image_uncropped)

        output_image, time_tf = self.run_processed_image(image_for_proc)

        # -----------------------------------------------------------------
        # Plot confidences as red-blue overlay
        # rb_image = seg.make_overlay(image, output_image)

        return self.tools.postprocess_image(
            output_image, self.output_image_uncropped, image), time_tf
