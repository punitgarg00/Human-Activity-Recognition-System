


import cv2  
import mediapipe as mp
import numpy as np
import time
import joblib
from collections import deque
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import time
import joblib
from collections import deque
import pandas as pd
from scipy.signal import savgol_filter

#
class ActivityRecognitionSystem:
 
    def __init__(self, model_path='activity_stacking_classifier.joblib', sequence_length=20):
        self.model = joblib.load(model_path)  # Load your pre-trained model
        self.activity_names = {
            0: 'Fall Down', 
            1: 'Lying Down', 
            2: 'Sit Down', 
            3: 'Sitting', 
            4: 'Stand up', 
            5: 'Standing', 
            6: 'Walking'
        }
        self.mp_pose = mp.solutions.pose
       

        self.mp_drawing = mp.solutions.drawing_utils
        # self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # Lower for better detection rate
            min_tracking_confidence=0.7,   # Keep high for stable tracking
            model_complexity=1,            # Balance between performance and accuracy
            smooth_landmarks=True          # Enable built-in smoothing
                         )
 
        self.sequence_length = sequence_length
        self.landmarks_buffer = deque(maxlen=sequence_length)
        
        self.current_activity = None
        self.activity_confidence = 0.0
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.2 # seconds between predictions
        
    def calculate_angle(self, a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.arccos(cosine) * 180.0 / np.pi

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def extract_features_from_sequence(self, landmarks_sequence):
        features = {}
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
        num_frames = len(landmarks_sequence)
        
        angles = {
            "left_elbow": [], "right_elbow": [],
            "left_knee": [], "right_knee": [],
            "left_hip": [], "right_hip": [],
            "torso": [], "head_tilt": []
        }
        height_ratios = []
        hip_heights = []
        com_positions = []
        
        for frame in landmarks_sequence:
            s = np.array(frame[LEFT_SHOULDER][:2])  # Left shoulder
            rs = np.array(frame[RIGHT_SHOULDER][:2])  # Right shoulder
            le = np.array(frame[LEFT_ELBOW][:2])  # Left elbow
            re = np.array(frame[RIGHT_ELBOW][:2])  # Right elbow
            lw = np.array(frame[LEFT_WRIST][:2])  # Left wrist
            rw = np.array(frame[RIGHT_WRIST][:2])  # Right wrist
            lh = np.array(frame[LEFT_HIP][:2])  # Left hip
            rh = np.array(frame[RIGHT_HIP][:2])  # Right hip
            lk = np.array(frame[LEFT_KNEE][:2])  # Left knee
            rk = np.array(frame[RIGHT_KNEE][:2])  # Right knee
            la = np.array(frame[LEFT_ANKLE][:2])  # Left ankle
            ra = np.array(frame[RIGHT_ANKLE][:2])  # Right ankle
            nose = np.array(frame[NOSE][:2])  # Nose
            ls = np.array(frame[LEFT_SHOULDER][:2])  # Converting to NumPy array
            rs = np.array(frame[RIGHT_SHOULDER][:2])  # Converting to NumPy array
  
            # Joint angles
            angles["left_elbow"].append(self.calculate_angle(ls, le, lw))
            angles["right_elbow"].append(self.calculate_angle(rs, re, rw))
            angles["left_knee"].append(self.calculate_angle(lh, lk, la))
            angles["right_knee"].append(self.calculate_angle(rh, rk, ra))
            angles["left_hip"].append(self.calculate_angle(ls, lh, lk))
            angles["right_hip"].append(self.calculate_angle(rs, rh, rk))

            # Torso angle (shoulders to hips)
            mid_shoulder = (ls + rs) / 2
            mid_hip = (lh + rh) / 2
            torso_angle = np.arctan2(mid_shoulder[1] - mid_hip[1], mid_shoulder[0] - mid_hip[0])
            angles["torso"].append(np.degrees(torso_angle))
            
            # Head tilt (nose to mid shoulder line)
            shoulder_vector = rs - ls
            head_vector = nose - mid_shoulder
            head_tilt = np.arctan2(head_vector[1], head_vector[0]) - np.arctan2(shoulder_vector[1], shoulder_vector[0])
            angles["head_tilt"].append(np.degrees(head_tilt))
            
            # Height and hip metrics
            body_height = self.calculate_distance(nose, mid_hip)
            shoulder_width = self.calculate_distance(ls, rs)
            height_ratios.append(body_height / shoulder_width if shoulder_width > 0 else 0)
            hip_heights.append(mid_hip[1])
            
            # Center of mass (average of hips, shoulders, knees)
            keypoints = [ls, rs, lh, rh, lk, rk]
            com = np.mean(keypoints, axis=0)
            com_positions.append(com)
        
        # Smooth signals
        window_size = min(15, num_frames - (num_frames % 2) - 1)
        if window_size > 3:
            for key in angles:
                angles[key] = savgol_filter(angles[key], window_size, 2)
            height_ratios = savgol_filter(height_ratios, window_size, 2)
            hip_heights = savgol_filter(hip_heights, window_size, 2)
        
        # Aggregate stats
        for key, values in angles.items():
            features[f"{key}_mean"] = np.mean(values)
            features[f"{key}_std"] = np.std(values)
            features[f"{key}_min"] = np.min(values)
            features[f"{key}_max"] = np.max(values)
        
        # Symmetry
        features["elbow_symmetry"] = np.mean(np.abs(np.array(angles["left_elbow"]) - np.array(angles["right_elbow"])))
        features["knee_symmetry"] = np.mean(np.abs(np.array(angles["left_knee"]) - np.array(angles["right_knee"])))
        
        # Height and hips
        features["height_ratio_mean"] = np.mean(height_ratios)
        features["hip_height_mean"] = np.mean(hip_heights)
        features["hip_height_std"] = np.std(hip_heights)
        
        # Velocity and acceleration
        if num_frames > 0:
            hip_velocity = np.diff(hip_heights)
            features["hip_velocity_mean"] = np.mean(hip_velocity)
            features["hip_velocity_max_abs"] = np.max(np.abs(hip_velocity))
            
            com_positions = np.array(com_positions)
            com_velocity = np.linalg.norm(np.diff(com_positions, axis=0), axis=1)
            features["com_speed_mean"] = np.mean(com_velocity)
            features["com_speed_max"] = np.max(com_velocity)
            
            # Total joint energy
            left_knee_vel = np.diff(angles["left_knee"])
            right_knee_vel = np.diff(angles["right_knee"])
            joint_energy = np.sum(np.abs(left_knee_vel)) + np.sum(np.abs(right_knee_vel))
            features["total_joint_movement"] = joint_energy
        else:
            features["hip_velocity_mean"] = 0
            features["hip_velocity_max_abs"] = 0
            features["com_speed_mean"] = 0
            features["com_speed_max"] = 0
            features["total_joint_movement"] = 0
        
        return features
    
    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Extract landmarks
            frame_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            # Add landmarks to buffer
            self.landmarks_buffer.append(frame_landmarks)
            
            current_time = time.time()
            if len(self.landmarks_buffer) == self.sequence_length and (current_time - self.last_prediction_time) > self.prediction_interval:
                features = self.extract_features_from_sequence(self.landmarks_buffer)
                features_df = pd.DataFrame([features])
                
                prediction_proba = self.model.predict_proba(features_df)[0]
                predicted_class = np.argmax(prediction_proba)
                confidence = prediction_proba[predicted_class]
                
                self.current_activity = self.activity_names[predicted_class]
                self.activity_confidence = confidence
                self.last_prediction_time = current_time
        
        if self.current_activity:
            cv2.putText(frame, f"Activity: {self.current_activity}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {self.activity_confidence:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame

    def run(self, camera_id=0):
        """
        Run the real-time activity recognition system.
        
        Args:
            camera_id: ID of the webcam to use
        """
        # Initialize webcam
        video_path = "cam1.mp4"  # Change this to your video file path  
        cap = cv2.VideoCapture(video_path)

        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow('Activity Recognition', processed_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize and run the activity recognition system
    recognition_system = ActivityRecognitionSystem()
    recognition_system.run()
