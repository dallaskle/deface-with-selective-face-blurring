# Selective Face Anonymization for Lecture Videos

A modified version of [deface](https://github.com/ORB-HD/deface) that intelligently preserves the lecturer's face while anonymizing all other faces in educational videos. This project combines multiple state-of-the-art models to achieve robust face tracking and selective anonymization.

## Key Features

- **Smart Face Detection**: Uses CenterFace (from deface) for efficient face detection and blurring
- **Person Detection**: Employs YOLOv8 to identify human figures in the frame
- **Person Re-identification**: Utilizes TorchReID to accurately identify and track the target person (lecturer)
- **Robust Face Tracking**: Implements OpenCV's CSRT tracker for consistent face tracking
- **Recovery Mechanisms**: Includes tracking recovery when the target face is temporarily lost
- **Batch Processing**: Supports processing multiple videos with their corresponding target person images

## Usage

1. **Setup Directory Structure**:
   - Create a folder for each video
   - Include the video file (default name: `video.mp4`)
   - Add a `target_person` folder with reference images of the person to preserve

2. **Run the Script**:

```
python deface/main.py /path/to/input_directory \
--video-filename video.mp4 \
--target-person-dirname target_person \
--debugging
```

## Command Line Arguments

- `input_dir`: Directory containing video folders
- `--video-filename`: Name of video files (default: `video.mp4`)
- `--target-person-dirname`: Name of target person directory (default: `target_person`)
- `--debugging`: Enable debug visualization and console output
- `--keep-audio`: Preserve original audio in output video

## Credits

- Project idea: [Divide-By-0](https://github.com/Divide-By-0/) and [MIT SOUL](http://soul.mit.edu/)
- Original deface project: [ORB-HD/deface](https://github.com/ORB-HD/deface)
- YOLOv8: [Ultralytics](https://github.com/ultralytics/yolov8)
- TorchReID: [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)