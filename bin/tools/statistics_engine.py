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



import timeit
import numpy as np


class StatisticsEngine(object):
    """A class to calculate run statistics"""

    def __init__(self):
        self.frame_count = 0
        self.processing_image_size = 0
        self.original_image_size = 0
        self.time_end_of_start = 0
        self.frame_times_tf = []
        self.frame_times_no_drawing = []
        self.frame_times_inner = []
        self.frame_times_outer = []
        self.inner_time_spent = 0
        self.pure_tf_spent = 0
        self.overhead_spent = 0

        self.time_start = timeit.default_timer()

    def append_to_frame_times_tf(self, value):
        """Self descriptive"""
        self.frame_times_tf.append(value)

    def append_to_frame_times_no_drawing(self, value):
        """Self descriptive"""
        self.frame_times_no_drawing.append(value)

    def append_to_frame_times_inner(self, value):
        """Self descriptive"""
        self.frame_times_inner.append(value)

    def append_to_frame_times_outer(self, value):
        """Self descriptive"""
        self.frame_times_outer.append(value)

    def add_to_inner_time_spent(self, value):
        """Self descriptive"""
        self.inner_time_spent = value

    def add_to_pure_tf_spent(self, value):
        """Self descriptive"""
        self.pure_tf_spent = value

    def add_to_overhead_spent(self, value):
        """Self descriptive"""
        self.overhead_spent = value

    def set_time_end_of_start(self, value):
        """Self descriptive"""
        self.time_end_of_start = value

    def set_frame_count(self, value):
        """Self descriptive"""
        self.frame_count = value

    def set_images_sizes(self, original_image_size, processing_image_size):
        """Self descriptive"""
        self.original_image_size = original_image_size
        self.processing_image_size = processing_image_size

    def process_statistics(self):
        """Process the statistics of data received up to this point
        Input: None
        Output: (str)
            Statistics Summary"""

        # Simple stats
        time_end = timeit.default_timer()  # record end time of capture

        nCount = self.frame_count + 1
        stat_string = ""
        stat_string += "\n" + '\n' + 8 * (6 * '-' + '\t')

        stat_string += "\n" + \
            'Preparation spent:\t\t%.4g sec' % (
                self.time_end_of_start - self.time_start)
        stat_string += "\n" + \
            "Runs spent:\t\t\t%.4g sec" % (time_end - self.time_end_of_start)
        stat_string += "\n" + \
            "Netto time per iteration:\t%.4g sec" % (
                self.inner_time_spent / nCount)
        if self.pure_tf_spent > 0:
            stat_string += "\n" + \
                "Time in tf per iteration:\t%.4g sec" % (
                    self.pure_tf_spent / nCount)

        stat_string += "\n" + \
            "Overhead per iteration:\t\t%.4g sec" % (
                self.overhead_spent/nCount)

        stat_string += "\n"
        str_to_logging = "Processing Image Size:\t\t" + \
            np.round(self.processing_image_size).astype(int).__str__()
        if 'original_image_size' in globals():
            str_to_logging += "\tfrom\t" + \
                np.round(self.original_image_size).astype(int).__str__()

        stat_string += "\n" + str_to_logging
        stat_string += "\n" + "Number of frames:\t\t%d" % nCount

        stat_string += "\n" + 8 * (6 * '-' + '\t') + '\n\n'

        # Inner Stats:
        def get_stat_array(a):
            return (np.around(np.min(a[1:]), decimals=4),
                    np.around(np.mean(a[1:]), decimals=4),
                    np.around(np.max(a[1:]), decimals=4),
                    np.around(np.std(a[1:]), decimals=4))

        stat_string += "\n" + \
            '\tStatistics:\t| \t   Min  \t|\t   Mean \t|\t   Max  \t|\t   Std   \t|'
        if len(self.frame_times_tf) > 0:
            stat_string += "\n" + '\tTf Frames:\t|' + (4 * '\t %06.4f \t|') % \
                (get_stat_array(self.frame_times_tf))

        stat_string += "\n" + '\tProc w/o draw:\t|' + (4 * '\t %06.4f \t|') % \
            (get_stat_array(self.frame_times_no_drawing))

        stat_string += "\n" + '\tDrawing(only):\t|' + (4 * '\t %06.4f \t|') % \
            (get_stat_array(np.subtract(self.frame_times_inner, self.frame_times_no_drawing)))

        stat_string += "\n" + '\tInner Frames: \t|' + (4 * '\t %06.4f \t|') % \
            (get_stat_array(self.frame_times_inner))

        stat_string += "\n" + '\tOuter Frames: \t|' + (4 * '\t %06.4f \t|') % \
            (get_stat_array(self.frame_times_outer))

        stat_string += "\n" + 8 * (6 * '-' + '\t') + '\n\n'

        # calculate processed frames per second
        calc_fps = float(nCount /
                         (time_end - self.time_end_of_start))
        stat_string += "\n" + 'Estimated process per second rate: %.4g fps' % calc_fps

        return stat_string
