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
from std_msgs.msg import Header, Int8MultiArray, MultiArrayDimension
from cnn_bridge.msg import Netmask

import timeit
class MaskPublisher(object):
    '''A class to publish commands to ROS'''

    def __init__(self, width, height, topic_name="mask"):
        '''Initialize ros publisher, ros subscriber'''
        self.mask_pub = rospy.Publisher(
            rospy.get_name() + '/' + topic_name, Netmask, queue_size=1)

        self.mask = Int8MultiArray()
        self.mask.layout.dim.append(MultiArrayDimension())
        self.mask.layout.dim.append(MultiArrayDimension())
        self.mask.layout.dim[0].label = "height"
        self.mask.layout.dim[1].label = "width"
        self.mask.layout.dim[0].size = height
        self.mask.layout.dim[1].size = width
        self.mask.layout.dim[0].stride = height * width
        self.mask.layout.dim[1].stride = width
        self.mask.layout.data_offset = 0

    def send_mask(self, data, meta_header):
        """Set up a ROS camera
        Input: 
            data  (str) The data to send to ROS [throttle, steer]
        Output: None"""
        try:
            rosgraph.Master('/rostopic').getPid()
        except socket.error:
            raise ROSTopicIOException("Unable to communicate with master!")
        
        start_time = timeit.default_timer()

        mask_data = np.array(data)        

        np_arr_time = timeit.default_timer() - start_time

        mask_data = np.array(data).flatten()
        flatten_time = timeit.default_timer() - start_time - np_arr_time

        mask_data = np.pad(mask_data, (self.mask.layout.data_offset, 0), 'constant')
        pad_time = timeit.default_timer() - start_time- np_arr_time - flatten_time

        self.mask.data = mask_data.tolist()
        list_time = timeit.default_timer() - start_time - np_arr_time - flatten_time - pad_time

        # rospy.logwarn((np_arr_time*1000, flatten_time*1000, pad_time*1000, list_time*1000))

        h = Header()
        h.stamp = rospy.Time.now()

        msg = Netmask()
        msg.header = h
        msg.image_identifier = meta_header
        msg.mask = self.mask

        self.mask_pub.publish(msg)
