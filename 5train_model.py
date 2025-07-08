import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_activity_classifier():
    """
    Train a stacking ensemble classifier to classify human activities.
    """
    # Load data
    X_features = pd.read_csv("activity_features.csv")
    y_labels = np.load("activity_labels.npy")
    
    # Activity label mapping
    activity_names = {
        0: 'Fall Down', 
        1: 'Lying Down', 
        2: 'Sit Down', 
        3: 'Sitting', 
        4: 'Stand up', 
        5: 'Standing', 
        6: 'Walking'
    }
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.3, random_state=42, stratify=y_labels
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Define base models
    base_models = [
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('svc', SVC(probability=True, kernel='rbf', C=1.0, gamma='scale'))
    ]

    # Define meta-model
    meta_model = LogisticRegression(max_iter=3000)
    
    # Define StackingClassifier
    clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        passthrough=True,
        n_jobs=-1
    )

    # Fit model
    clf.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[activity_names[i] for i in range(len(activity_names))]))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[activity_names[i] for i in range(len(activity_names))],
                yticklabels=[activity_names[i] for i in range(len(activity_names))])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Feature importance (based on Random Forest in base models)
    rf_model = clf.named_estimators_['rf']
    feature_importance = pd.DataFrame({
        'Feature': X_features.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features (from RF)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Save the stacked model
    joblib.dump(clf, 'activity_stacking_classifier.joblib')
    print("Model saved as 'activity_stacking_classifier.joblib'")
    
    return clf, feature_importance

if __name__ == "__main__":
    print("For Stacking Model:")
    clf, feature_importance = train_activity_classifier()
    print("\nTop 10 most important features (from RF in stack):")
    print(feature_importance.head(10))

