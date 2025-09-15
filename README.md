# VideoTracker

**VideoTracker** is a Python script that adds tracking effects to videos to detect people, cars, animals, and more. It's easy to use, and you can customize the tracking visuals by changing the parameters.  

<img width="50%" alt="image" src="https://github.com/user-attachments/assets/bc3d46bd-47d5-4bf9-a617-a9126cb30b21" />

## Features

- Detect and track objects such as people, cars, animals, etc.
- Customize tracking visuals through adjustable parameters.
- Supports multiple Ultralytics YOLO models
- Simple input/output workflow.

## Customization

Change visual parameters (e.g., color, line thickness) in the script to customize how tracking effects appear.

Switch between different YOLO models depending on your need for speed or accuracy.

## Requirements

- **Python 3.8 – 3.12** (Some libraries may not support versions newer than 3.12)  
- Libraries:
  - [Ultralytics](https://github.com/ultralytics/ultralytics)
  - [Mediapipe](https://github.com/google/mediapipe)
  - [OpenCV](https://opencv.org/)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/karimhantaou/VideoTracker.git
    cd VideoTracker
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place the video you want to process in the input folder and name it vid.mp4.

2. Run the script

```bash
python main.py
```

3. The processed video with tracking effects will be saved as vid.mp4 in the output folder

## Licence

This project is open-source. Feel free to modify and use it for your personal or commercial projects.
