import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt

def load_activity_data(processed_data_dir):
    """
    Load all the processed landmark data for all activities.
    
    Args:
        processed_data_dir: Directory containing processed landmark data
        
    Returns:
        X: List of landmark sequences
        y: List of activity labels
    """
    X = []  # Store landmark sequences
    y = []  # Store activity labels
    
    # Define the list of activities
    activities = ['Fall Down', 'Lying Down', 'Sit Down', 'Sitting', 'Stand up', 'Standing', 'Walking']
    
    # Define a numeric mapping for activities
    activity_map = {activity: idx for idx, activity in enumerate(activities)}
    
    # Loop through each activity folder
    for activity in activities:
        activity_dir = os.path.join(processed_data_dir, activity)
        
        if not os.path.exists(activity_dir):
            print(f"Warning: Activity directory {activity_dir} not found. Skipping.")
            continue
        
        # Find all landmark files
        landmark_files = glob.glob(os.path.join(activity_dir, "*_landmarks.npy"))
        print(f"Found {len(landmark_files)} landmark files for activity {activity}")
        
        # Process each landmark file
        for landmark_file in landmark_files:
            try:
                # Load landmark data
                landmarks = np.load(landmark_file)
                
                # Add to dataset
                X.append(landmarks)
                y.append(activity_map[activity])
                
            except Exception as e:
                print(f"Error loading {landmark_file}: {e}")
    
    print(f"Total sequences loaded: {len(X)}")
    print(f"Activity distribution: {pd.Series(y).value_counts().sort_index()}")
    
    return X, y, activity_map

# Example usage
if __name__ == "__main__":
    processed_data_dir = "processed_data"  # Directory containing processed landmark data
    X, y, activity_map = load_activity_data(processed_data_dir)
    
    # Save the loaded data for future use
    data_file = "activity_data.npz"
    np.savez(data_file, X=np.array(X, dtype=object), y=np.array(y), 
             activity_map=np.array(list(activity_map.items()), dtype=object))
    print(f"Data saved to {data_file}")
