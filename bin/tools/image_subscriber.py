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




# Use compressed image?
COMPRESSED = False

import rospy
import cv2

if not COMPRESSED:
    from cv_bridge import CvBridge, CvBridgeError
    from sensor_msgs.msg import Image
else:
    import numpy as np
    from sensor_msgs.msg import CompressedImage

import timeit

class ImageSubscriber(object):
    '''A class to subscribe to a ROS camera'''

    def __init__(self, cam_type="usb_cam"):
        '''Initialize ros publisher, ros subscriber'''

        self.currentImage = None
        self.currentHeader = None

        self.frame_count = 0
        self.start_time = timeit.default_timer()

        # subscribed Topic
        sub_string = ""

        if cam_type == "usb_cam":
            sub_string = "usb_cam/image_raw"
        elif cam_type == "cv_cam":
            sub_string = "cv_camera/image_raw"
        elif cam_type == "ueye":
            sub_string = "ueye_0/image_raw"
        else:
            raise NameError("Camera type not defined")

        if not COMPRESSED:
            self.bridge = CvBridge()
            self.subscriber = rospy.Subscriber(sub_string,
                                               Image, self.callback,  queue_size=1)
        else:
            sub_string += "/compressed"
            self.subscriber = rospy.Subscriber(
                sub_string, CompressedImage, self.callback, queue_size=1)

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted to cv2 format'''
        global count
        if not COMPRESSED:
            self.currentImage = cv2.resize(
                self.bridge.imgmsg_to_cv2(ros_data, "rgb8"),
                (640, 480))
        else:
            np_arr = np.fromstring(ros_data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, 1)

            self.currentImage = cv2.cvtColor(image_np, cv2.BGR2RGB)

        self.currentHeader = ros_data.header
        
        self.frame_count += 1
        # if self.frame_count % 100 == 0:
            # rospy.logwarn("Size:\t" + str(self.currentImage.shape))
            # rospy.logwarn( "ImgSub:\t" + str(self.frame_count) + ":\t" + 
                        #    str( float(self.frame_count) / (timeit.default_timer() - self.start_time)))
