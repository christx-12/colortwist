import cv2
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path='yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        self.confidence = 0.5
    
    def detect(self, frame):
        """YOLO Pose Detection → Keypoints"""
        results = self.model(frame, verbose=False)
        
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            if len(keypoints) == 17:
                self.confidence = results[0].keypoints.conf[0].max()
                return keypoints
        
        return None
    
    def preprocess_frame(self, frame):
        """Optional: Frame optimieren"""
        frame = cv2.resize(frame, (640, 480))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
