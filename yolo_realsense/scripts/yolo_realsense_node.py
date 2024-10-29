#!/usr/bin/env python3
import sys
sys.path.append('/home/quad/yolov5-python3.6.9-jetson')  # YOLOv5가 설치된 경로

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String  # 새로운 토픽에 사용할 메시지 타입
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from pathlib import Path

class YoloRealsenseNode:
    def __init__(self):
        rospy.init_node('yolo_realsense_node', anonymous=True)
        self.bridge = CvBridge()
        self.model = self.load_yolo_model()
        
        # Image topic을 구독
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        # 새로운 토픽을 생성 (객체 타입과 개수를 보내기 위한 토픽)
        self.detection_pub = rospy.Publisher('/yolo/detections', String, queue_size=10)

    def load_yolo_model(self):
        model_path = Path("/home/quad/catkin_ws/src/yolo_realsense/scripts/yolov5s.pt")  # 모델 파일 경로
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        model = model['model'].float().eval()
        model.to(device)
        return model

    def image_callback(self, msg):
        try:
            # ROS 이미지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # YOLOv5로 이미지 처리
            results = self.process_image(cv_image)
            # 결과를 화면에 표시
            self.display_results(cv_image, results)
            # 객체 정보 게시
            self.publish_detections(results)
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")

    def process_image(self, cv_image):
        # YOLOv5 모델을 통해 이미지 처리
        results = self.model(cv_image)
        return results

    def display_results(self, cv_image, results):
        # YOLOv5 모델의 결과를 이미지에 표시
        detections = results.pandas().xyxy[0]
        for _, row in detections.iterrows():
            # 사각형 그리기
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 이미지 표시
        cv2.imshow('YOLO Detection', cv_image)
        cv2.waitKey(1)  # 화면 업데이트를 위해 잠시 대기

    def publish_detections(self, results):
        # 인식된 객체의 타입과 개수를 문자열로 변환하여 새로운 토픽에 게시
        detections = results.pandas().xyxy[0]
        labels = detections['name'].value_counts()  # 각 객체 타입별 개수 계산
        detection_info = []

        # 각 객체 타입과 개수 추출
        for label, count in labels.items():
            detection_info.append(f"{label}: {count}")

        # 최종적으로 객체 타입과 개수를 하나의 문자열로 병합
        detection_message = ', '.join(detection_info)

        # 메시지를 토픽에 게시
        rospy.loginfo(f"Publishing detection info: {detection_message}")
        self.detection_pub.publish(detection_message)

if __name__ == '__main__':
    try:
        yolo_node = YoloRealsenseNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

