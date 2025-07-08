**Project Overview**
This project implements a pipeline for human activity recognition using pose estimation and machine learning. It leverages the MediaPipe library to extract pose landmarks from images or videos, performs feature engineering on these landmarks, and trains machine learning models to classify different physical activities. The project also supports real-time activity classification using a webcam feed.

**Directory Structure**

 1install_requirements.py: Installs all necessary Python packages for the project

 2extract_landmarks.py: Extracts pose landmarks from video or image data using the MediaPipe library.

 3organise.py: Loads and organizes the processed landmark data, preparing it for feature engineering and model training.

 4feature_engineering.py: Calculates angles and other features from the pose landmarks to create a robust feature set for classification.

 5train_model.py: Trains machine learning models (Random Forest, SVM, Logistic Regression, XGBoost) using the engineered features. Also generates evaluation plots such as confusion matrices and feature importance.

 6realtime.py: Enables real-time activity recognition using a video path.

 activity_features.csv: Stores the engineered features extracted from pose landmarks for various activities.

 activity_labels.npy: Contains activity labels corresponding to the features.

 activity_randomforest_classifier.joblib, activity_stacking_classifier.joblib: Serialized trained model files.

 confusion_matrix.png, feature_importance.png: Visualizations for model evaluation.


 **Usage**
1. Extract Landmarks
     Use 2extract_landmarks.py to process your videos or images and extract pose landmarks.

2. Organize Data
    Run 3organise.py to structure the extracted data for further processing.

3. Feature Engineering
    Execute 4feature_engineering.py to compute angles and other features from the landmarks.

4. Train Models
    Use 5train_model.py to train activity classification models. This script will also generate evaluation plots.

5. Real-Time Classification
    Run 6realtime.py to perform live activity recognition providing your video path.