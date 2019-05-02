std_msgs/Header header   	            # Header timestamp should be the time the mask is published at
std_msgs/Header image_identifier        # the image that was processed
std_msgs/Int8MultiArray mask           	# actual matrix data, size is (height * width)
