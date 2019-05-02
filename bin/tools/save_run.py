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



"""
The VideoSaver module recieves images and saves them to a video

Input: Image
Output: Video

Submodule of the cnn_publisher program

--------------------------------------------------------------------------------

Copyright (c) 2019 Perfetto Team, Technical Division, Israel Defence Force

"""
from __future__ import print_function

import logging
import datetime
import cv2
import rospy

class VideoSaver(object):
    """A class to save images into a video"""
    def __init__(self, image, video_name=""):
        self.h_video_out = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if video_name is "" or video_name is True:
            self.video_base_name = 'Camera__' + \
                datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
        else:
            self.video_base_name = video_name
        (height, width, _) = image.shape
        self.video_sz = (width, height)
        self.open_out_video()
        # self.print_info()

    def __del__(self):
        self.close_video()

    def close_video(self):
        """ Close video if opened
            Since, we need it also in reopen module put in in separate function
            Input:
                None
            Output:
                None
        """
        if self.h_video_out is not None and self.h_video_out.isOpened():
            # print('Close ' + self.video_base_name + ' video')
            self.h_video_out.release()
            self.h_video_out = None

    def open_out_video(self, count=None):
        """Open a video out
        Input:
            count (int) The frame number
        Output: None"""

        self.close_video()

        if count is not None:
            video_name = self.video_base_name + '._%d_' % count + '.mp4'
        else:
            video_name = self.video_base_name + '.mp4'

        self.h_video_out = cv2.VideoWriter(
            video_name, self.fourcc, 10.0, self.video_sz)

        if self.h_video_out.isOpened():
            rospy.loginfo("Open video stream {0} of size ({1}, {2})".format(video_name, self.video_sz[0],self.video_sz[1]))
        else:
            rospy.logfatal("Failed to open video {0}".format(video_name))

    def print_info(self):
        """Print info for debug """

        if self.h_video_out is not None:
            rospy.logdebug('Video_base_name: {0} is opened: {1}'.\
            format( self.video_base_name, self.h_video_out.isOpened()))
    