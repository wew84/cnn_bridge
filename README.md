# Overview
This package provides support for parsing convolution neural networks (CNN), and publishing them as ROS messages. Currently the package supports both detection an segmentation networks.

Input can be either from camera topics, an OpenCV camera, or a video.

# ROS Nodes
## cnn_publisher 
ROS node that opens a freeze graph and run it on images.
### Publishes
#### **detection**
type = cnn_bridge/Detection
Detection data. Published as boxes, scores, and classes. In addition, the header of the image that the network was run on (useful for statistics, and for hmi).
**OR**
#### segmentation
type = cnn_bridge/Netmask
Segmentation data. Published a 2-dimensional array of mask values. In addition, the header of the image that the network was run on (useful for statistics, and for hmi).

### Parameters
* *source*
type = string  
required = True  
The source of the images to be run through the network. There are three types of inputs allowed. The first, is a path to a video file (any OpenCV compatible files will work). The second, is a device number (0, 1, 2, 3,...) for an OpenCV device. The third is a ROS Image or CompressedImage topic. Currently, this option works for 'usb_cam' (subscribes to 'usb_cam/image_raw'), 'cv_cam' (subscribes to 'cv_camera/image_raw'), and 'ueye' subscribes to 'ueye_0/image_raw').  
* *logdir*
type = string  
required = True  
Path to the hypes file. See bellow for an example JSON file.
* *metadata_source*
     type = string  
     required = True  
     Path to the metadata file. See bellow for an example JSON file.
* *mode*
type = string  
required = True  
Either 'detection' or 'segmentation' depending on the mode.  
* *input_tensor*
type = string  
required = True  
Self explanatory.  
* *output_tensor*
type = string/[string]  
required = True  
If segmentation, self explanatory. If detection an array of three tensors that are [boxes,scores,classes]
* *display*
     type = Boolean  
     default = True  
     Whether to display the output or not
* *video_save*
type = Boolean/String  
default = True  
**Not currently implemented!** Whether to save the output or not. Use False to disable. True saves to '<current_dir>/Camera__datetime'. If a string is provided it will be the video title. The node saves both the raw video and the video with the mask / boxes overlay. If running on a video file, the raw is not saved.
* *cpu*
type = string
default = False
**Not currently implemented!** Sets whether to run the network on the CPU if an Nvidia GPU is present.
* *gpu_percent*
type = Float
default = 1.0
Sets the percentage of an Nvidia GPU to use. This is used generally for running simultaneous networks.

# Launch File Examples
Start a cnn_bridge in segmentation mode:
`$ roslaunch cnn_bridge segmentation_publisher.launch`
Start a cnn_bridge in detection mode:
`$ roslaunch cnn_bridge detection_publisher.launch`

# Hypes Example
`
{
    "frozen_graph_path": "<path_to_frozen_graph.pb>",
    "image_height": 361,
    "image_width": 641,
    "resize_image": true,
    //TODO Add additional fields
}
`

# Metadata JSON
If segmentation mode:
`
{
    "classes": ["CLASS_NAME_1", "CLASS_NAME_2", ..., "CLASS_NAME_N"]
}
`  

If detection mode:
`
{
    "classes": [{
        "color": (red, green, blue),
        "id": < The ID of the class as outputted from the network >,
        "name": < Name assigned to the class >
        "id_category": < The ID of a parent class (ie. If class dog parent Animal) >
        "category": < The name of a parent class (ie. If class dog parent Animal) >
    }]
}
`
