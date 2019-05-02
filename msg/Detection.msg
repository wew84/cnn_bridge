std_msgs/Header header   	            # Header timestamp should be the time the mask is published at
std_msgs/Header image_identifier        # the image that was processed
geometry_msgs/Quaternion[] boxes
std_msgs/Float32[] scores
std_msgs/Int16[] classes