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

import rospy 
import numpy as np
import timeit

COMPRESSED = False
if not COMPRESSED:
    from cv_bridge import CvBridge, CvBridgeError
    from sensor_msgs.msg import Image
else:
    import numpy as np
    import cv2
    from sensor_msgs.msg import CompressedImage


class testCam:
    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted to cv2 format'''
        currentImage = ""
        if not COMPRESSED:
            currentImage =  self.bridge.imgmsg_to_cv2(ros_data, "rgb8")
        else:
            np_arr = np.fromstring(ros_data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, 1)

            currentImage = cv2.cvtColor(image_np, cv2.BGR2RGB)

        np.mean(currentImage)
        self.count += 1
        # if  self.count % 100 == 0:
            # rospy.logwarn( "testCam\t" + str(self.count) + ":\t" + 
            #                 str(float(self.count / (timeit.default_timer() -  self.start_time))))

    def __init__(self):
        self.bridge = CvBridge()
        self.count = 0
        self.start_time = timeit.default_timer()
        if not COMPRESSED:
            self.subscriber = rospy.Subscriber("ueye_0/image_raw",
                                                Image,  self.callback,  queue_size=1)
        else:
            sub_string += "/compressed"
            self.subscriber = rospy.Subscriber(
                sub_string, CompressedImage, callback, queue_size=1)
