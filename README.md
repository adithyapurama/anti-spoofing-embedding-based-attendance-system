# Anti-Spoofing Embedding-Based Attendance System

## Real-Time Attendance Verification Demo

<p align="center">
  <video width="800" controls>
    <source src="assets/demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>



## 1. Overview

This project implements a secure, real-time biometric attendance system designed to mitigates common presentation attacks (print, replay, and screen-based spoofs). Unlike traditional face recognition systems that only match identity, this solution integrates a liveness detection layer using MiniFASNet to ensure the subject is a live human before processing identification.

The system utilizes Flask for the backend interface, ONNX Runtime for high-performance inference, and SQLite for lightweight user management. It captures video from a webcam, detects faces, verifies liveness, and matches the user against a database of high-dimensional embeddings using ArcFace.

## 2. Core Features

- **Liveness Detection:** MiniFASNet-based anti-spoofing to distinguish real faces from print/replay attacks  
- **Face Recognition:** ArcFace embedding-based identity verification  
- **Weighted Enrollment:** Multi-image registration with quality-based weighting  
- **Uncertainty Modeling:** Variance-penalized similarity scoring  
- **Suspicious Activity Logging:** Automatic spoof and mismatch capture  
- **Efficient Inference:** ONNX Runtime for detection and recognition  
- **Web Interface:** Flask-based admin and employee UI  


## 3. End-to-End Workflow

```text
[Webcam Input]  
↓  
[Face Detection (SCRFD)] → No Face → Ignore Frame  
↓  
[Face Alignment (112×112)]  
↓  
[Anti-Spoofing Check (MiniFASNet)]  
├── Spoof → Log Event + Save Frame  
└── Real  
  ↓  
[ArcFace Embedding Extraction]  
↓  
[Similarity + Variance Penalty]  
↓  
Match → Attendance IN/OUT  
No Match → Log Unmatched Event
```

## 4. Repository Structure

```text
anti-spoofing-embedding-based-attendance-system/
├── flask_app.py
├── Attendance.csv
├── suspicious_log.csv
├── models/
│   ├── anti_spoof/
│   ├── arcface/
│   ├── face_detection/
│   └── users.db
├── anti_spoof_predict.py
├── alignment.py
├── convert_to_onnx.py
├── camera_test.py
├── utility.py
├── detect.py
└── static/
    ├── enrollment_images/
    └── suspicious_frames/
```


## 5. Model Stack Explanation
Face Detection
The system uses SCRFD (Sample and Computation Redistribution for Face Detection) via ONNX. It provides high-speed face detection and 5-point landmark regression (eyes, nose, mouth corners) required for alignment.

    Input Size: 640x480

    Confidence Threshold: 0.45

Face Alignment
Using the landmarks provided by SCRFD, the system applies a Similarity Transformation (Affine) to crop and align the face to a standard 112x112 pixel image. This ensures eyes and facial features are in consistent positions for the recognition model.

Anti-Spoofing
The liveness detection relies on MiniFASNet (and optionally MultiFTNet), a lightweight CNN architecture designed for face anti-spoofing.

    Framework: PyTorch

    Architecture: MiniFASNet (V1/V2/SE variants)

    Input: Multi-scale crop of the aligned face.

    Output: Softmax probability (Class 0: Spoof, Class 1: Real).

Face Recognition
Identity verification is handled by ArcFace (ResNet-100 backbone) converted to ONNX.

    Input: 112x112 Aligned RGB Face (Normalized -1 to 1).

    Output: 512-dimensional normalized embedding vector.

## 6. Anti-Spoofing Decision Logic
The AntiSpoofPredict class aggregates predictions. The decision logic in flask_app.py is as follows:

 1. The detected face is resized and passed through the loaded anti-spoofing models.

 2. The maximum probability class is determined.

 3. Strict Thresholding: The system considers a face "Real" only if:

    The predicted class is 1 (Real).

    The confidence score is greater than 0.9 (SPOOF_CONF_THRESHOLD).

 4. If the score is below 0.9, the frame is flagged as a potential spoof, even if the argmax class is Real.

## 7. Face Verification & Thresholding Strategy
The system employs a sophisticated matching strategy that accounts for enrollment quality.
 1. Cosine Similarity: Calculates the dot product of the    input embedding and the stored user embedding (both normalized).
 2. Variance Penalty: The system subtracts the stored variance of the user's enrollment embeddings from the similarity score.
  Formula: Score = Cosine(Embedding_new, Embedding_stored) − Mean(Variance_stored)
  This penalizes users who had poor or inconsistent lighting/angles during enrollment, requiring a closer match for verification.
 3. Decision: If $Score > 0.55$ (MATCH_THRESHOLD),the      identity is verified.

## 8. User Enrollment Strategy
To ensure a high-quality reference database, enrollment is not a single-shot process:

 1. Multiple Captures: The admin captures up to 7 images of the user.

 2. Quality Metrics: Each image is evaluated for:

    Sharpness: Laplacian variance.

    Area Ratio: Size of the face relative to the frame.

 3. Weighted Aggregation: Valid embeddings are combined into a Weighted Mean Embedding and Variance Vector based on their quality scores.

 4. Database Storage: The system stores the mean embedding (identity) and the variance (uncertainty) in users.db.

## 9. Attendance Logging Mechanism
Attendance is recorded in Attendance.csv.

    Fields: User ID, Date, Time, Event (IN/OUT).

    Logic: The system prevents duplicate entries (e.g., prevents marking "IN" twice consecutively on the same day).

    Suspicious Cleanup: Upon a successful "OUT" event, temporary suspicious frames associated with that user session are cleaned up to manage storage.

## 10. Handling Spoofing & Suspicious Events
Security events are tracked in suspicious_tracker to prevent log flooding.

    Triggers:

        Spoof: Anti-spoof score < 0.9.

        Unmatched: Real face but similarity score < 0.55.

    Action:

        The frame is saved to static/suspicious_frames/<user_id>/.

        An entry is written to suspicious_log.csv.

        The UI displays a red warning box.

    Throttling: Suspicious events for the same user are logged at most once every 5 seconds.

## 11. Environment Setup
The project relies on Python 3.x and specific libraries for computer vision and deep learning.

Prerequisites:

    Python 3.8+

    CMake (required for building dlib or specific opencv extensions if needed)

    Visual C++ Redistributable (if on Windows)

## 12. Dependency Installation
Install the required packages using pip.

    pip install flask opencv-python numpy torch torchvision onnxruntime six scikit-image easydict

Note: For GPU acceleration, install onnxruntime-gpu and the appropriate PyTorch CUDA version.

## 13. How to Run the Project
    1. Clone the repository:
    git clone https://github.com/your-username/anti-spoofing-embedding-based-attendance-system.git
    cd anti-spoofing-embedding-based-attendance-system
    2. Verify Model Placement: Ensure the models/ directory contains the required ONNX and PTH files as described in the structure section.
    3. Start the Application:
    python flask_app.py
    4. Access the Interface: Open a web browser and navigate to http://localhost:2000.

## 14. Configuration Parameters
Key configurations are located in flask_app.py:

    TARGET_SIZE = (640, 480): Input resolution for the camera.

    MAX_ENROLL_IMAGES = 7: Number of images captured during registration.

    MIN_VALID_IMAGES = 2: Minimum acceptable images to form a profile.

    MATCH_THRESHOLD = 0.55: Cosine similarity cutoff for verification.

    SPOOF_CONF_THRESHOLD = 0.9: Minimum confidence to pass liveness check.

## 15. Known Constraints
    
    Lighting: Strong backlighting or very low light may affect Anti-Spoofing accuracy.

    Single Face: The logic currently processes faces[0]. If multiple people are in the frame, it will only process the most prominent face.

    Compute: While ONNX is optimized, the Anti-Spoofing step runs in PyTorch; running on a CPU-only machine may result in lower FPS.

## 16. Potential Enhancements
    HTTPS Support: Essential for browser webcam permissions in production.

    Database Migration: Move from SQLite/CSV to PostgreSQL for scalability.

    Asynchronous Processing: Offload model inference to a background thread or Celery worker to improve UI responsiveness.

    Dockerization: Containerize the application for easier deployment

## 17. License

This project is intended for academic and research purposes.
License to be defined.
