import numpy as np
from scipy.signal import savgol_filter
import pandas as pd

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine) * 180.0 / np.pi

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_features_from_sequence(landmarks_sequence):
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

    num_frames = landmarks_sequence.shape[0]

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
        ls = frame[LEFT_SHOULDER][:2]
        rs = frame[RIGHT_SHOULDER][:2]
        le = frame[LEFT_ELBOW][:2]
        re = frame[RIGHT_ELBOW][:2]
        lw = frame[LEFT_WRIST][:2]
        rw = frame[RIGHT_WRIST][:2]
        lh = frame[LEFT_HIP][:2]
        rh = frame[RIGHT_HIP][:2]
        lk = frame[LEFT_KNEE][:2]
        rk = frame[RIGHT_KNEE][:2]
        la = frame[LEFT_ANKLE][:2]
        ra = frame[RIGHT_ANKLE][:2]
        nose = frame[NOSE][:2]

        # Joint angles
        angles["left_elbow"].append(calculate_angle(ls, le, lw))
        angles["right_elbow"].append(calculate_angle(rs, re, rw))
        angles["left_knee"].append(calculate_angle(lh, lk, la))
        angles["right_knee"].append(calculate_angle(rh, rk, ra))
        angles["left_hip"].append(calculate_angle(ls, lh, lk))
        angles["right_hip"].append(calculate_angle(rs, rh, rk))

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
        body_height = calculate_distance(nose, mid_hip)
        shoulder_width = calculate_distance(ls, rs)
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
    if num_frames > 1:
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

def prepare_dataset(X, y):
    all_features = []
    valid_indices = []

    for i, landmarks in enumerate(X):
        try:
            features = extract_features_from_sequence(landmarks)
            all_features.append(features)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error in sequence {i}: {e}")

    X_features = pd.DataFrame(all_features)
    y_labels = np.array([y[i] for i in valid_indices])
    X_features.fillna(0, inplace=True)
    return X_features, y_labels

if __name__ == "__main__":
    data = np.load("activity_data.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]

    print("Extracting features...")
    X_features, y_labels = prepare_dataset(X, y)
    X_features.to_csv("activity_features.csv", index=False)
    np.save("activity_labels.npy", y_labels)

    print(f"Extracted {X_features.shape[1]} features for {X_features.shape[0]} samples.")
    print("Top features:", X_features.columns.tolist())
