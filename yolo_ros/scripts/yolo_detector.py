#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import cv2
import torch
import numpy as np

class YoloDetector:
    def __init__(self):
        rospy.init_node('yolo_detector', anonymous=True)
        self.publisher = rospy.Publisher('detection_results', String, queue_size=10)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                results = self.model(frame)
                detections = results.pandas().xyxy[0]  # Detected objects in a DataFrame
                detection_info = detections.to_string(index=False)
                self.publisher.publish(detection_info)  # Publish detection results
                cv2.imshow('YOLOv5 Detection', np.array(results.render()))  # Display results
                if cv2.waitKey(1) == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = YoloDetector()
    try:
        detector.run()
    except rospy.ROSInterruptException:
        pass

