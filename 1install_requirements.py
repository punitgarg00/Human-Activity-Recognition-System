import subprocess
import sys
def install_required_packages():
    required_packages = [
        'opencv-python',
        'mediapipe',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'matplotlib'
    ]
    
    for package in required_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All required packages installed successfully!")

if __name__ == "__main__":
    install_required_packages()
