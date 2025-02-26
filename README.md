# Face Recognition System

## Overview
This face recognition project was inspired by my biometry class and its based on concepts from the *Introduction to Biometry* book. It utilizes OpenCV, `face_recognition`, to identify and verify faces in real-time using a webcam.

The system allows users to set a custom matching threshold, balancing False Rejection Rate (FRR) and False Acceptance Rate (FAR), and provides a dynamic and interactive experience for real-time facial recognition.

![FAR and FRR](images/FARandFRR.ppm.png)

---

## Features
- **Face dataset loading**: Uses the `Labeled Faces in the Wild` (LFW) dataset to build a reference set of known faces.
- **Dynamic thresholding**: Users can customize the matching confidence threshold.
- **Camera detection and initialization**: Automatically detects available cameras and selects the appropriate one.
- **Live face recognition**: Identifies and labels faces in real time using a webcam feed.
- **Error handling**: It handles errors like camera failures or missing faces in the dataset.
- **Performance logging**: Logs processing updates and frame counts to indicate system activity.
- **Real-time visualization**: Displays detected faces with bounding boxes and confidence percentages.

---

## Installation
### Requirements
Ensure you have the following dependencies installed:

```bash
pip install opencv-python face-recognition numpy scikit-learn
```

The `face_recognition` library requires `dlib`, so ensure your system supports it. If needed, install additional dependencies:

```bash
pip install dlib
```

For macOS users, installing OpenCV via `brew` may resolve compatibility issues:

```bash
brew install opencv
```

---

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/gabrielniculaesei/FaceRecognition
   cd FaceRecognition
   ```

2. **Run the script:**
   ```bash
   python face_recognition.py
   ```

3. **Set the threshold:** Enter a percentage value to define the threshold value for face matching.

4. **Camera selection:** The program will attempt to find an available camera automatically. If multiple cameras are present, it selects an alternative if needed.

5. **Live recognition:** Detected faces will be labeled with names and confidence scores. Press `q` to exit.

---

## Code Breakdown
### 1. **Dataset Loading**
- Uses `fetch_lfw_people` to load face images and their corresponding labels.
- Converts grayscale images to RGB format for compatibility with `face_recognition`.
- Extracts and encodes known face embeddings.

### 2. **Camera Handling**
- Lists available cameras and selects one dynamically.
- Implements robust initialization with retries in case of failures.

### 3. **Face Recognition Logic**
- Captures frames from the webcam and converts them to RGB.
- Detects faces and computes embeddings.
- Compares detected faces against known encodings using `face_distance`.
- Displays results with bounding boxes and labels based on matching confidence.

### 4. **Error Handling & Cleanup**
- Ensures the camera is released properly upon exit.
- Handles missing dataset images gracefully.

---

## Customization
### Adjusting the Matching Threshold
Modify the threshold input at the start of the script to control match sensitivity.

```python
threshold = int(input("Enter matching threshold as % (e.g., 50): "))
```

Higher thresholds reduce false positives but may reject valid matches.

### Changing the Dataset
Replace `fetch_lfw_people` with a custom dataset for improved recognition accuracy.

---

## Potential Improvements
I plan to use **deep learning models** for higher accuracy and optimize performance with **GPU acceleration** using OpenCV's CUDA support.



