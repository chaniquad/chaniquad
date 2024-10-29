#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

class DetectionSubscriberNode:
    def __init__(self):
        rospy.init_node('detection_subscriber_node', anonymous=True)
        rospy.Subscriber('/yolo/detections', String, self.detection_callback)

    def detection_callback(self, msg):
        detection_info = msg.data
        rospy.loginfo(f"Received detection info: {detection_info}")

if __name__ == '__main__':
    try:
        detection_node = DetectionSubscriberNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
