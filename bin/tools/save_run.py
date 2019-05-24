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

from __future__ import print_function

import logging
from os import mkdir, path
import datetime
import cv2
import rospy


def get_powers(mode):
    flags = []
    power = 5
    while mode > 0 and power >= 0:
        if mode - 2**power >= 0:
            flags.append(2**power)
            mode = mode - 2**power
        power -= 1
    return flags


class VideoSaver(object):
    """A class to save images into a video"""

    def __init__(self, image, mode, save_path="", name=""):

        rospy.logerr(mode)
        self.run_modes = get_powers(mode)
        if save_path != "" and save_path[len(save_path)-1] != '/':
            self.save_path = save_path + '/'
        else:
            self.save_path = save_path

        # Setup name
        if name is "" or name is True:
            self.base_name = 'Camera__' + \
                datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
        else:
            self.base_name = datetime.datetime.now().strftime(
                "%Y-%m-%d_%H:%M:%S") + "___" + name

        # Setup video
        if 2 in self.run_modes or 8 in self.run_modes or 32 in self.run_modes:
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            (height, width, _) = image.shape
            self.video_sz = (width, height)
            self.org_video_out = None
            self.overlay_video_out = None
            self.mask_video_out = None
            self.open_out_video()

        # Setup video
        if 1 in self.run_modes or 4 in self.run_modes or 16 in self.run_modes:
            self.setup_image_saving()

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
        if self.org_video_out is not None and self.org_video_out.isOpened():
            self.org_video_out.release()
            self.org_video_out = None

        if self.overlay_video_out is not None and self.overlay_video_out.isOpened():
            # print('Close ' + self.video_base_name + ' video')
            self.overlay_video_out.release()
            self.overlay_video_out = None

        if self.mask_video_out is not None and self.mask_video_out.isOpened():
            # print('Close ' + self.video_base_name + ' video')
            self.mask_video_out.release()
            self.mask_video_out = None

    def open_out_video(self, count=None):
        """Open a video out
        Input:
            count (int) The frame number
        Output: None"""

        self.close_video()
        if 2 in self.run_modes:
            if count is not None:
                name = self.save_path + self.base_name + '_%d_' % count + '.mp4'
            else:
                name = self.save_path + self.base_name + '.mp4'
            self.org_video_out = cv2.VideoWriter(
                name, self.fourcc, 10.0, self.video_sz)

        if 8 in self.run_modes:
            if count is not None:
                name = self.save_path + self.base_name + '_overlay_%d_' % count + '.mp4'
            else:
                name = self.save_path + self.base_name + '_overlay.mp4'
            self.overlay_video_out = cv2.VideoWriter(
                name, self.fourcc, 10.0, self.video_sz)

        if 32 in self.run_modes:
            if count is not None:
                name = self.save_path + self.base_name + '_mask_%d_' % count + '.mp4'
            else:
                name = self.save_path + self.base_name + '_mask.mp4'
            self.mask_video_out = cv2.VideoWriter(
                name, self.fourcc, 10.0, self.video_sz)

        if (self.org_video_out is not None and self.org_video_out.isOpened()) \
                or (self.overlay_video_out is not None and self.overlay_video_out.isOpened()) \
                or (self.mask_video_out is not None and self.mask_video_out.isOpened()):
            rospy.loginfo("Open video stream {0} of size ({1}, {2})".format(
                name, self.video_sz[0], self.video_sz[1]))
        else:
            rospy.logfatal("Failed to open video {0}".format(name))

    def print_info(self):
        """Print info for debug """

        if 2 in self.run_modes and self.org_video_out is not None:
            rospy.logdebug('Video_base_name: {0} is opened: {1}'.
                           format(self.base_name, self.org_video_out.isOpened()))

        if 8 in self.run_modes and self.overlay_video_out is not None:
            rospy.logdebug('Video_base_name: {0} is opened: {1}'.
                           format(self.base_name, self.org_video_out.isOpened()))

        if 32 in self.run_modes and self.mask_video_out is not None:
            rospy.logdebug('Video_base_name: {0} is opened: {1}'.
                           format(self.base_name, self.mask_video_out.isOpened()))

    def setup_image_saving(self):
        """Set up the ability to save all the capture images
        Input: None
        Output: None"""
        base_dir_name = self.save_path + self.base_name

        if 1 in self.run_modes:
            mkdir(base_dir_name + '_org')
        if 4 in self.run_modes:
            mkdir(base_dir_name + '_overlay')
        if 16 in self.run_modes:
            mkdir(base_dir_name + '_mask')
