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

import os
import json
import rospy
import unicodedata

from std_msgs.msg import String, Int8, ColorRGBA
from cnn_bridge.srv import *
from cnn_bridge.msg import SegmentationClass


class MetadataService(object):
    def __init__(self, shape, classPath, logdir, mode):
        try:
            classes_parsed = json.load(open(classPath, "r"))['classes']
            if mode == "segmentation":
                self.metadata = getSegmentationMetadataResponse()
                self.metadata.class_metadata = []
                
                for current_class in classes_parsed:
                    class_msg = SegmentationClass()
                    class_msg.color = ColorRGBA()
                    class_msg.color.r = classes_parsed[current_class]['color'][0]
                    class_msg.color.g = classes_parsed[current_class]['color'][1]
                    class_msg.color.b = classes_parsed[current_class]['color'][2]
                    class_msg.classification = Int8(classes_parsed[current_class]['id'])
                    class_msg.classification_name = String(classes_parsed[current_class]['name'])
                    class_msg.parent_class = Int8(classes_parsed[current_class]['id_category'])
                    class_msg.parent_name = String(classes_parsed[current_class]['category'])
                    self.metadata.class_metadata.append(class_msg)
                

                self.s = rospy.Service(rospy.get_name() + '/get_cnn_metadata',
                                    getSegmentationMetadata, self.get_cnn_metadata_callback)
            elif mode == "detection":
                self.metadata = getDetectionMetadataResponse()
                self.metadata.class_names = []
                for current_class in classes_parsed:
                    self.metadata.class_names.append(String(unicodedata.normalize('NFKD', current_class).encode('ascii','ignore')))
                self.s = rospy.Service(rospy.get_name() + '/get_cnn_metadata',
                    getDetectionMetadata, self.get_cnn_metadata_callback)

        except IOError as e:
            rospy.logerr("Could not read metadata file\nI/O error({0}): {1}".format(e.errno, e.strerror))
        self.metadata.image_height = shape[1]
        self.metadata.image_width = shape[0]



    def get_cnn_metadata_callback(self, req):
        rospy.logdebug("Sent metadata about node {0}".format(rospy.get_name()))
        return self.metadata
