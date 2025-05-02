# GenTbD: Generalised Tracking-by-Detection

GenTbD is a Python-based framework for building **tracking-by-detection systems**. It provides a modular architecture for combining object detection and tracking, enabling users to create robust tracking pipelines tailored to their needs. This work is inspired by several prominent online MOT frameworks such as ([ByteTrack](https://github.com/ifzhang/ByteTrack) , [DeepSORT](https://github.com/nwojke/deep_sort), [StrongSORT](https://github.com/dyhBUPT/StrongSORT), [SMILETrack](https://github.com/WWangYuHsiang/SMILEtrack), [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)) While these projects have influenced the design and modular structure of GenTbD, all code and implementation herein are original. 

---

## Key Features

- **Abstracted Design**: Add custom video processors, detectors and trackers
- **Generalised Detection**: Enables integration of various detector types, including object detectors, keypoint detectors, and re-identification (Re-ID) models. The framework adopts a generalised detection approach, allowing users to define identity-preserving properties (e.g., bounding boxes, appearance features, keypoints) and apply custom similarity metrics for more flexibility.
- **Track Lifecycle**: Implements a clear track lifecycle (`NEW`, `MATCHED`, `LOST`, `RESERVED`) for better state management.
- **Cascaded Assignment**: Implements a cascaded assignment algorithm, which is a multi-stage assignment procedure based on some priority ordering and can be used throughout the tracking procedure. 
- **Generalised Tracking**:  Ability to modify tracking logic, association procedure, and track management.

---

## System Overview

The system consists of three core components:

1. **Video Processor**: Manages video input and processes each frame.
2. **Detector**: Performs object detection within frames.
3. **Tracker**: Updates and manages tracks based on detected objects.

### Architecture Diagram
![Architecture](diagrams/GenTbD_architecture.png)

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- OpenCV
- NumPy
- ONNX Runtime (for YOLOv7 detector)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/GenTbD.git
   cd GenTbD
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv7 ONNX model and place it in the `models/` directory:
   ```bash
   mkdir models
   # Add your model download instructions here
   ```

---

## Usage Guide

### Running the System

1. Place your video file in the `data/` directory.
2. Place your model files (e.g., YOLOv7 ONNX model) in the `models/` directory.
3. Update the `video_file` and `model_file` variables in `src/main.py`:
   ```python
   video_file = "data/your_video.mp4"
   model_file = "models/yolov7.onnx"
   ```
4. Run the main script:
   ```bash
   python src/main.py
   ```
5. Key events during video processing:
    - **`c`**: Toggle between continuous and step-by-step processing modes.
    - **`s`**: Save the current frame as an image.
    - **`q`**: Quit the video processing.

### Example Output
![Example](diagrams/example.gif)

---

## File Structure

Here’s how the project directory should be organized:

```
GenTbD/
├── data/                     # Directory for input video files
│   └── your_video.mp4
├── models/                   # Directory for model files
│   └── yolov7.onnx
├── diagrams/                 # Directory for diagrams and images
│   └── GenTbD_architecture.png
├── src/                      # Source code directory
│   ├── main.py               # Main script to run the system
│   ├── Tracking/             # Tracking-related modules
│   │   ├── Track.py
│   │   ├── Trackers/
│   │   │   └── Tracker.py
│   │   └── Partition.py
│   └── Detecting/            # Detection-related modules
│       └── Detections/
│           └── Detection.py
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # License file
```

---

## Features

### Track Lifecycle
A track is defined as a state machine which is
![lifecycle](diagrams/Track_lifecycle.png)

### Assignment
![assignment](diagrams/simple_assignment.png)

### Cascaded Assignment
The system uses a cascaded assignment algorithm to prioritise high-confidence detections and tracks during the matching process. This improves accuracy and reduces false positives.
![cascaded_assignment_algo](diagrams/Cascaded_assignment_algo.png)
![cascaded_assignment](diagrams/Cascaded_assignment.png)


---

## Future Work

- Fully define system parameters for base version and create simpler interface
- Add support for appearance-based tracking using re-identification models.
- Implement keypoint-based tracking for human pose estimation.
- Add weighted sum and gating thresholds for feature fusion.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.