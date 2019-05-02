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




import socket
import numpy as np
import rospy
import rosgraph
from std_msgs.msg import Header, Int16, Float32
from geometry_msgs.msg import Quaternion
from cnn_bridge.msg import Detection

import timeit
class DetectionPublisher(object):
    '''A class to publish commands to ROS'''

    def __init__(self, topic_name="detection"):
        '''Initialize ros publisher, ros subscriber'''
        self.detect_pub = rospy.Publisher(
            rospy.get_name() + '/' + topic_name, Detection, queue_size=1)

    def send_mask(self, data, meta_header):
        """Set up a ROS camera
        Input: 
            data  (str) The data to send to ROS [throttle, steer]
        Output: None"""
        try:
            rosgraph.Master('/rostopic').getPid()
        except socket.error:
            raise rospy.ROSException("Unable to communicate with master!")
        
        detection = Detection()

        # detection.boxes = [float(x) for x in data['boxes']]
        # detection.scores = [float(x) for x in data['scores']]
        detection.classes = []
        for x in data['classes']:
            detection.classes.append(Int16(int(x)))
        detection.scores = []
        for x in data['scores']:
            detection.scores.append(Float32(float(x)))
        detection.boxes = []
        for box in data['boxes']:
            quat = Quaternion()
            quat.x = box[0]
            quat.y = box[1]
            quat.z = box[2]
            quat.w = box[3]
            detection.boxes.append(quat)

        h = Header()
        h.stamp = rospy.Time.now()

        detection.header = h
        detection.image_identifier = meta_header

        self.detect_pub.publish(detection)