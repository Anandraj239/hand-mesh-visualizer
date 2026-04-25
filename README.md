# 🖐️ Hand Mesh Visualizer

An elegant, performant Python script that uses the latest **MediaPipe Tasks Vision API** to generate a dynamic, glowing skeleton and mesh tracking system for your hands in real-time.

## Features
- **Clean Skeletons**: Tracks standard skeletal joints to draw a flawless digital outline of your hands.
- **Dynamic Cross-Connections**: Dynamically connects corresponding keypoints (like thumb-tip to thumb-tip and wrist to wrist) between both hands.
- **Spectrum Coloring**: Auto-generates unique, vibrant HSV rainbow colors for each cross-hand connection to create a jaw-dropping visual bridge.
- **Auto-Model Downloading**: Automatically pulls down the required `hand_landmarker.task` AI model directly from Google on its first run. 

## Installation

Ensure you have Python installed, then run:

```bash
pip install opencv-python mediapipe
```

## Usage

Simply run the script:

```bash
python hand_mesh.py
```

- A webcam window will pop up. Bring one or two hands into the frame.
- **Press `Q` or `ESC`** to exit the camera window.
