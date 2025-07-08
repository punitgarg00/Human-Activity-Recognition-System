import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks_from_video(video_path, output_landmark_path, output_viz_path=None):
    """
    Extract pose landmarks from a video and save them to a file.
    
    Args:
        video_path: Path to the input video
        output_landmark_path: Path to save the landmarks
        output_viz_path: Optional path to save visualization video
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_landmark_path), exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer for visualization if needed
    if output_viz_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(output_viz_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize pose detector
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        landmarks_sequence = []
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect pose
            results = pose.process(image_rgb)
            
            # Store landmarks if detected
            if results.pose_landmarks:
                # Extract landmarks
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                landmarks_sequence.append(frame_landmarks)
                
                # Draw pose landmarks on the image for visualization
                if output_viz_path:
                    annotated_image = image.copy()
                    mp_drawing.draw_landmarks(
                        annotated_image, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS
                    )
                    # Add frame number
                    cv2.putText(
                        annotated_image, 
                        f"Frame: {frame_count}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    out_video.write(annotated_image)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames from {video_path}")
        
        # Save the landmarks
        if landmarks_sequence:
            np.save(output_landmark_path, np.array(landmarks_sequence))
            print(f"Saved landmarks to {output_landmark_path}")
        else:
            print(f"No pose landmarks detected in {video_path}")
    
    # Release resources
    cap.release()
    if output_viz_path:
        out_video.release()
    
    return True

def process_activity_videos(activity_dir, output_dir, visualize=True):
    """
    Process all .avi videos in an activity directory and extract landmarks.
    
    Args:
        activity_dir: Directory containing videos for a specific activity
        output_dir: Directory to save the extracted landmarks
        visualize: Whether to create visualization videos
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .avi videos in the directory
    video_files = [f for f in os.listdir(activity_dir) if f.endswith('.avi')]
    print(f"Found {len(video_files)} .avi videos in {activity_dir}")
    
    for video_file in video_files:
        video_path = os.path.join(activity_dir, video_file)
        
        # Define output paths
        video_name = os.path.splitext(video_file)[0]
        landmark_path = os.path.join(output_dir, f"{video_name}_landmarks.npy")
        
        if visualize:
            viz_path = os.path.join(output_dir, f"{video_name}_visualization.avi")
        else:
            viz_path = None
        
        print(f"Processing {video_path}...")
        extract_landmarks_from_video(video_path, landmark_path, viz_path)

def process_all_activities(base_data_dir, output_base_dir):
    """
    Process videos for all activities.
    
    Args:
        base_data_dir: Base directory containing activity subdirectories
        output_base_dir: Base directory to save extracted landmarks
    """
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # List of activities to process
    activities = ['Fall Down', 'Lying Down', 'Sit Down', 'Sitting', 'Stand up', 'Standing', 'Walking']
    
    for activity in activities:
        activity_dir = os.path.join(base_data_dir, activity)
        if not os.path.exists(activity_dir):
            print(f"Warning: Activity directory {activity_dir} not found. Skipping.")
            continue
        
        output_dir = os.path.join(output_base_dir, activity)
        print(f"\nProcessing activity: {activity}")
        process_activity_videos(activity_dir, output_dir)

# Example usage
if __name__ == "__main__":
    # Specify your paths
    train_dir = "train"  # Directory containing all activity subdirectories
    output_dir = "processed_data"  # Directory to save processed data
    
    process_all_activities(train_dir, output_dir)
