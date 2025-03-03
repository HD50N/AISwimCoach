import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

class SwimmingDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.results = None
        self.landmarks = None
        self.style = "Unknown"
        self.left_stroke = 0
        self.right_stroke = 0
        self.l_stage = 'up'
        self.r_stage = 'up'
        self.start_time = None
    
    def get_strokes(self):
        return self.left_stroke + self.right_stroke if self.style in ["Freestyle", "Backstroke"] else max(self.left_stroke, self.right_stroke)
    
    def get_style(self):
        return self.style
    
    def get_elapsed_time(self):
        return time.time() - self.start_time if self.start_time else 0
    
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle
    
    def get_landmark_value(self, part):
        return self.landmarks[self.mp_pose.PoseLandmark[part].value] if self.landmarks else None
    
    def get_orientation(self):
        left_shoulder, right_shoulder = self.get_landmark_value("LEFT_SHOULDER"), self.get_landmark_value("RIGHT_SHOULDER")
        return "Forward" if left_shoulder and right_shoulder and left_shoulder.x > right_shoulder.x else "Backward"
    
    def set_ready(self):
        self.start_time = time.time()
    
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            self.landmarks = self.results.pose_landmarks.landmark
            orientation = self.get_orientation()
            left_hip, left_shoulder, left_elbow, left_wrist = [self.get_landmark_value(p) for p in ["LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]]
            right_hip, right_shoulder, right_elbow, right_wrist = [self.get_landmark_value(p) for p in ["RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]]
            
            left_shoulder_angle = self.calculate_angle([left_hip.x, left_hip.y], [left_shoulder.x, left_shoulder.y], [left_elbow.x, left_elbow.y])
            right_shoulder_angle = self.calculate_angle([right_hip.x, right_hip.y], [right_shoulder.x, right_shoulder.y], [right_elbow.x, right_elbow.y])
            left_elbow_angle = self.calculate_angle([left_shoulder.x, left_shoulder.y], [left_elbow.x, left_elbow.y], [left_wrist.x, left_wrist.y])
            right_elbow_angle = self.calculate_angle([right_shoulder.x, right_shoulder.y], [right_elbow.x, right_elbow.y], [right_wrist.x, right_wrist.y])
            
            if left_shoulder_angle > 160 and right_shoulder_angle > 160 and not self.start_time:
                self.set_ready()
            
            if self.start_time:
                if left_shoulder_angle < 40 and self.l_stage == 'half-down':
                    if self.style == "Unknown":
                        self.style = "Butterfly" if right_shoulder_angle < 70 and left_elbow_angle > 160 else "Breaststroke" if right_shoulder_angle < 70 else "Freestyle" if orientation == "Backward" else "Backstroke"
                    self.l_stage = "down"
                elif 40 <= left_shoulder_angle <= 160:
                    self.l_stage = "half-up" if self.l_stage == 'down' else "half-down" if self.l_stage == 'up' else self.l_stage
                elif left_shoulder_angle > 160 and self.l_stage == 'half-up':
                    self.l_stage = "up"
                    self.left_stroke += 1
                
                if right_shoulder_angle < 40 and self.r_stage == 'half-down':
                    if self.style == "Unknown":
                        self.style = "Butterfly" if left_shoulder_angle < 70 and right_elbow_angle > 160 else "Breaststroke" if left_shoulder_angle < 70 else "Freestyle" if orientation == "Backward" else "Backstroke"
                    self.r_stage = "down"
                elif 40 <= right_shoulder_angle <= 160:
                    self.r_stage = "half-up" if self.r_stage == 'down' else "half-down" if self.r_stage == 'up' else self.r_stage
                elif right_shoulder_angle > 160 and self.r_stage == 'half-up':
                    self.r_stage = "up"
                    self.right_stroke += 1
            
            cv2.putText(image, f'Stroke: {self.get_strokes()}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(self.style), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'{self.get_elapsed_time():.2f}', (10, 460), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        except:
            pass
        
        self.mp_drawing.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(102, 255, 255), thickness=2, circle_radius=2),
                                       self.mp_drawing.DrawingSpec(color=(240, 207, 137), thickness=2, circle_radius=2))
        return image
    
    def count_strokes(self, src=0):
        cap = cv2.VideoCapture(src)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            cv2.imshow('Stroke Counter', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def reset(self):
        self.__init__()
        
if __name__ == "__main__":
    detector = SwimmingDetector()
    detector.count_strokes()
    #'/Users/hudsonch/Downloads/swim3.mp4'