# Drowsiness Detection System

This project implements a drowsiness detection system using OpenCV, dlib, and Simpleaudio. The system detects the user's eye blinks through facial landmarks captured from a webcam and plays a warning sound if signs of drowsiness are detected.

## Features

- **Real-time eye blink detection**: Monitors the user's eye activity through a webcam.
- **Drowsiness alert**: Plays an alert sound when signs of drowsiness are detected.
- **Facial landmark detection**: Uses dlib's 68-point facial landmark model to track the user's eye movements.

## Requirements

1. **Pre-trained Model Information**: Added detailed information about the `shape_predictor_68_face_landmarks.dat` file, explaining what it does and where to download it.
2. **Music File**: Explained the use of the `music.wav` file and how it can be customized with other `.wav` files.
3. **File Placement**: Instructions about where to place the required files and how to modify paths if needed.
Before running the project, ensure the following dependencies are installed:

```bash
pip install opencv-python
pip install dlib
pip install numpy
pip install imutils
pip install simpleaudio

