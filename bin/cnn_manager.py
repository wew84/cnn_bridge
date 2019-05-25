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
import logging
import timeit
import atexit
import numpy as np
import cv2
import rospy

# TODO:  Functions to support video save (get_from_video_out_video_name, video_out_grid )
# are not defined

try:

    from tools import ImageSubscriber, StatisticsEngine, DrawingTools, Camera, VideoSaver, exit_handler, str2bool

except ImportError:
    # You forgot to initialize submodules
    rospy.logfatal("Could not import the submodules.")

    import traceback
    import sys
    ex_type, ex, tb = sys.exc_info()
    traceback.print_tb(tb)

    sys.exit(1)


class CNNManager(object):
    def __init__(self, source, save_mode, logdir, cpu, display, tensor_io, mode, gpu_percent=1, save_path=""):
        self.statistics_engine = StatisticsEngine()

        self.mode = mode
        # setup video output
        self.DO_VIDEO_SAVE = save_mode != 0

        self.camera = Camera(source)
        tmp_image, tmp_frame = self.camera.get_image()

        if self.DO_VIDEO_SAVE:
            video_name = self.DO_VIDEO_SAVE
            if isinstance(video_name, bool):
                video_name = self.camera.get_from_camera_video_name()

                if not video_name:  # Means we get empty from camera, means we are in video
                    video_name = source[source.rfind(
                        '/')+1:-(len(source)-source.find('.'))]

            self.h_video_out = VideoSaver(
                tmp_image, save_mode, name=video_name, save_path=save_path)
            self.h_video_out.print_info()

            self.DO_VIDEO_SAVE = True
            rospy.loginfo(
                'Video results of network run will be saved as %s', video_name)

        else:
            rospy.loginfo('Video results of network run will not be saved')
            self.h_video_out = None
            self.DO_VIDEO_SAVE = False

        self.SHOW_DISPLAY = str2bool(display)

        if self.mode == 'segmentation':
            from segmentation import MaskPublisher
            from segmentation import DeepLabSegmenter as SegmentationProcessor
            self.cnn_processor = SegmentationProcessor(
                logdir, self.camera.original_image_size, tensor_io, runCPU=cpu, gpu_percent=gpu_percent)

        elif self.mode == 'detection':
            from detection import YoloDetector as YoloProcessor
            from detection import DetectionPublisher
            self.cnn_processor = YoloProcessor(
                logdir, self.camera.original_image_size, tensor_io, runCPU=cpu, gpu_percent=gpu_percent)

        self.statistics_engine.set_images_sizes(
            self.camera.original_image_size, self.cnn_processor.tools.processing_image_size)

        atexit.register(exit_handler, self.camera,
                        self.h_video_out, self.statistics_engine)

        self.statistics_engine.set_time_end_of_start(timeit.default_timer())

        if self.mode == 'segmentation':
            self.mask_publisher = MaskPublisher(
                self.camera.original_image_size[0], self.camera.original_image_size[1])
            self.drawing_tools = DrawingTools('segmentation')
        elif self.mode == 'detection':
            self.detection_publisher = DetectionPublisher()
            self.drawing_tools = DrawingTools('detection')

    def run_loop(self):
        count = 0

        while not rospy.is_shutdown():
            # capture frames
            time_iter_start = timeit.default_timer()

            image_frame, image_header = self.camera.get_image()  # RGB

            if image_frame is None:
                break

            # Open/Close video
            if self.DO_VIDEO_SAVE is not False and count % 1000 == 0 and count > 0:
                self.h_video_out.open_out_video(count=count)

            time_iter_start_inner = timeit.default_timer()

            # Run the network
            cnn_result, time_pure_tf = self.cnn_processor.run_model_on_image(
                image_frame)

            # send way_prediction mask
            if self.mode == 'segmentation':
                self.mask_publisher.send_mask(cnn_result, image_header)
            elif self.mode == 'detection':
                self.detection_publisher.send_mask(cnn_result, image_header)

            time_iter_no_drawing = timeit.default_timer() - time_iter_start_inner

            # Plot the hard prediction as overlay
            if True or self.SHOW_DISPLAY or self.h_video_out is not None:
                if self.mode == 'segmentation':
                    image_overlay = self.drawing_tools.overlay_segmentation(
                        image_frame.copy(), cnn_result)
                elif self.mode == 'detection':
                    image_overlay = self.drawing_tools.draw_detection(
                        image_frame.copy(), cnn_result['boxes'],  cnn_result['scores'],  cnn_result['classes'], [320, 320])

            time_iter_inner = timeit.default_timer() - time_iter_start_inner

            # Show images and save video
            if self.SHOW_DISPLAY or self.h_video_out is not None:
                frame_overlay = cv2.cvtColor(image_overlay, cv2.COLOR_RGB2BGR)
                if self.SHOW_DISPLAY:
                    cv2.imshow('cnn_bridge', cv2.resize(
                        frame_overlay, dsize=self.camera.image_size_to_show))

                if self.DO_VIDEO_SAVE:
                    if 1 in self.h_video_out.run_modes:
                        assert os.path.exists(
                            self.h_video_out.save_path + self.h_video_out.base_name + '_org')
                        cv2.imwrite(self.h_video_out.save_path + self.h_video_out.base_name +
                                    '_org/' + str(image_header.seq) + '.jpg', cv2.cvtColor(image_frame, cv2.COLOR_RGB2BGR))

                    if 4 in self.h_video_out.run_modes:
                        assert os.path.exists(
                            self.h_video_out.save_path + self.h_video_out.base_name + '_overlay')
                        cv2.imwrite(self.h_video_out.save_path + self.h_video_out.base_name +
                                    '_overlay/' + str(image_header.seq) + '.jpg', frame_overlay)

                    if 16 in self.h_video_out.run_modes:
                        assert os.path.exists(
                            self.h_video_out.save_path + self.h_video_out.base_name + '_overlay')
                        opacity = self.drawing_tools.opacity
                        self.drawing_tools.opacity = 0
                        cv2.imwrite(self.h_video_out.save_path + self.h_video_out.base_name + '_mask/' + str(
                            image_header.seq) + '.jpg',  cv2.cvtColor(self.drawing_tools.overlay_segmentation(image_frame.copy(), cnn_result), cv2.COLOR_RGB2BGR))
                        self.drawing_tools.opacity = opacity

                if self.h_video_out is not None:
                    if 2 in self.h_video_out.run_modes:
                        assert self.h_video_out.org_video_out.isOpened()
                        self.h_video_out.org_video_out.write(
                            cv2.cvtColor(image_frame, cv2.COLOR_RGB2BGR))

                    if 8 in self.h_video_out.run_modes:
                        assert self.h_video_out.overlay_video_out.isOpened()
                        self.h_video_out.overlay_video_out.write(frame_overlay)

                    if 32 in self.h_video_out.run_modes:
                        assert self.h_video_out.mask_video_out.isOpened()
                        opacity = self.drawing_tools.opacity
                        self.drawing_tools.opacity = 0
                        self.h_video_out.mask_video_out.write(
                            self.drawing_tools.overlay_segmentation(image_frame.copy(), cnn_result))
                        self.drawing_tools.opacity = opacity

            # record time
            time_iter = timeit.default_timer() - time_iter_start

            str_to_print = '%d: Iteration spent %.4g inner %.4g' % (
                count, time_iter, time_iter_inner)
            if time_pure_tf is not None:
                str_to_print += '\t(in tf was %.4g)' % time_pure_tf
                self.statistics_engine.append_to_frame_times_tf(
                    time_pure_tf)

            rospy.logdebug(str_to_print)

            self.statistics_engine.append_to_frame_times_no_drawing(
                time_iter_no_drawing)
            self.statistics_engine.append_to_frame_times_inner(
                time_iter_inner)
            self.statistics_engine.append_to_frame_times_outer(time_iter)

            if count > 0:
                self.statistics_engine.add_to_inner_time_spent(
                    time_iter_inner)
                self.statistics_engine.add_to_overhead_spent(
                    time_iter - time_iter_inner)

                if time_pure_tf is not None:
                    self.statistics_engine.add_to_pure_tf_spent(
                        time_pure_tf)

            count += 1
            self.statistics_engine.set_frame_count(count)

            # quit option - pres q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        rospy.signal_shutdown('CNN closed. Shutting down...')
