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
from cnn_manager import CNNManager
from metadata_service import MetadataService
from tools import str2bool


def parse_args():
    """Parse the arguments provided when the module was initialized"""

    rospy.init_node('segmentation_publisher', log_level=rospy.DEBUG)
    params = {}
    #'The camera or video file to get the pictures from, '
    #          'options are: (String, Required) \n If video file, then '
    #          'path to the video file \n If ROS camera, then uEye, '
    #         'usb_cam \n If CV2 device, then device ID (0, 1, 3...)'
    if rospy.has_param('~source'):
        if isinstance(rospy.get_param('~source'), str):
            params['source'] = rospy.get_param('~source')
        else:
            raise rospy.ROSInitException(
                'The source display needs to be of type: String')
    else:
        raise rospy.ROSInitException('Param source is required (String)')

    # Path to KittiSeg Model Directory (String, Required)
    if rospy.has_param('~logdir'):
        if isinstance(rospy.get_param('~logdir'), str):
            params['logdir'] = rospy.get_param('~logdir')
        else:
            raise rospy.ROSInitException(
                'The logdir display needs to be of type: String')
    else:
        raise rospy.ROSInitException('Param logdir is required (String)')

    if rospy.has_param('~metadata_source'):
        if isinstance(rospy.get_param('~metadata_source'), str):
            params['metadata_source'] = rospy.get_param('~metadata_source')
        else:
            raise rospy.ROSInitException(
                'The metadata source needs to be of type: String')
    else:
        raise rospy.ROSInitException('Param source is required (String)')

    if rospy.has_param('~input_tensor'):
        if isinstance(rospy.get_param('~input_tensor'), str):
            params['input_tensor'] = rospy.get_param('~input_tensor')
        else:
            raise rospy.ROSInitException(
                'The input_tensor needs to be of type: String')
    else:
        raise rospy.ROSInitException('Param input_tensor is required (String)')

    if rospy.has_param('~output_tensor'):
        if isinstance(rospy.get_param('~output_tensor'), str):
            params['output_tensor'] = rospy.get_param('~output_tensor')
        else:
            raise rospy.ROSInitException(
                'The output_tensor needs to be of type: String')
    else:
        raise rospy.ROSInitException(
            'Param output_tensor is required (String)')

    # Whether to display the output or not (Boolean, Default True)
    if rospy.has_param('~display'):
        if str2bool(rospy.get_param('~display')) is not None:
            params['display'] = rospy.get_param('~display')
        else:
            raise rospy.ROSInitException(
                'The param display needs to be of type: Boolean')
    else:
        params['display'] = True

    # File to save the video to: (String) \n Use False to disable\n '
    #          'True saves to Camera__datetime \n If a string is provided it '
    #          'will be the video title
    if rospy.has_param('~video_save'):
        params['video_save'] = rospy.get_param('~video_save')
    else:
        params['video_save'] = ''

    # Sets whether to use an Nvidia GPU (bool, Default False)
    if rospy.has_param('~cpu'):
        if str2bool(rospy.get_param('~cpu')) is not None:
            params['cpu'] = rospy.get_param('~cpu')
        else:
            raise rospy.ROSInitException(
                'The param cpu needs to be of type: Boolean')
    else:
        params['cpu'] = False

    # Sets what mode the program is running in (String, Either detection or segmentation)
    if rospy.has_param('~mode'):
        if isinstance(rospy.get_param('~metadata_source'), str):
            if rospy.get_param('~mode') == 'detection' or rospy.get_param('~mode') == 'segmentation':
                params['mode'] = rospy.get_param('~mode')
            else:
                raise rospy.ROSInitException(
                    'Unknown mode')
        else:
            raise rospy.ROSInitException(
                'The param mode needs to be of type: String')
    else:
        raise rospy.ROSInitException('Param mode is required (String)')

    # Sets what mode the program is running in (String, Either detection or segmentation)
    if rospy.has_param('~gpu_percent'):
        if isinstance(rospy.get_param('~gpu_percent'), float):
            params['gpu_percent'] = rospy.get_param('~gpu_percent')
        else:
            raise rospy.ROSInitException(
                'The param gpu_percent needs to be of type: Float')
    else:
        params['gpu_percent'] = 1

    return params


if __name__ == "__main__":
    args = parse_args()
    tensor_io = {}
    tensor_io['input_tensor'] = args['input_tensor']
    tensor_io['output_tensor'] = args['output_tensor']
    if args['mode'] == 'segmentation':
        tensor_io['output_tensor'] = args['output_tensor']
    elif args['mode'] == 'detection':
        tensor_io['output_tensor'] = args['output_tensor'].split(',')

    manager = CNNManager(args['source'], args['video_save'], args['logdir'], args['cpu'],
                         args['display'], tensor_io, args['mode'], gpu_percent=args['gpu_percent'])
    metadata_srv = MetadataService(
        manager.camera.original_image_size, args['metadata_source'], args['logdir'], args['mode'])
    manager.run_loop()
