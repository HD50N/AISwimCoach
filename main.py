import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose module
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def main():
    cap = cv2.VideoCapture('/Users/hudsonch/Downloads/SwimExample.mp4')  # Change to video file path or 0 for webcam
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            height, width, _ = frame.shape  # Get frame dimensions
            
            # Flip the frame horizontally to match the mirrored webcam display
            # frame = cv2.flip(frame, 1)  # Remove mirroring for video
            
            # Convert image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            # Convert image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Define specific joint triplets for angle calculation, swapping left and right labels
                joint_triplets = [
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW)
                ]
                
                for joint_a, joint_b, joint_c in joint_triplets:
                    a = [landmarks[joint_a.value].x * width, landmarks[joint_a.value].y * height]
                    b = [landmarks[joint_b.value].x * width, landmarks[joint_b.value].y * height]
                    c = [landmarks[joint_c.value].x * width, landmarks[joint_c.value].y * height]
                    
                    angle = calculate_angle(a, b, c)
                    
                    # Convert to integer pixel coordinates
                    b_pixel = (int(b[0]), int(b[1]))  # Correctly align coordinates with mirrored frame
                    
                    # Adjust text placement to be directly above the joint, ensuring it remains in frame
                    text_x = max(10, min(b_pixel[0], width - 50))  # Keep text within horizontal bounds
                    text_y = max(20, min(b_pixel[1] - 20, height - 10))  # Adjust vertical placement
                    
                    # Display swapped joint name and angle above the joint_b location
                    swapped_name = "RIGHT" if "LEFT" in joint_b.name else "LEFT"
                    joint_label = joint_b.name
                    
                    cv2.putText(image, f'{joint_label}: {int(angle)}°',
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                
            except:
                pass
            
            # Render pose detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
            cv2.imshow('Mediapipe Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
