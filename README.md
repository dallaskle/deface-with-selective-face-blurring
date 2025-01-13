# Selective Face Anonymization for Lecture Videos

A modified version of [deface](https://github.com/ORB-HD/deface) that intelligently preserves the lecturer's face while anonymizing all other faces in educational videos. This project combines multiple state-of-the-art models to achieve robust face tracking and selective anonymization.

## Key Features

- **Smart Face Detection**: Uses CenterFace (from deface) for efficient face detection and blurring
- **Person Detection**: Employs YOLO11 to identify human figures in the frame
- **Person Re-identification**: Utilizes TorchReID to accurately identify and track the target person (lecturer)
- **Robust Face Tracking**: Implements OpenCV's CSRT tracker for consistent face tracking
- **Recovery Mechanisms**: Includes tracking recovery when the target face is temporarily lost
- **Batch Processing**: Supports processing multiple videos with their corresponding target person images

## Usage

1. **Setup Directory Structure**:
   - Create a folder for each video
   - Include the video file (default name: `video.mp4`)
   - Add a `target_person` folder with reference images of the person to preserve (for reference images it is recommended to include full body images of the target person in the video from different angles as the detection uses max similarity with target images for ReID)

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
- `--thresh`: Face detection threshold (tune this to trade off between false positive and false negative rate) (default: `0.4`)
- `--disable-tracker-reset`: Disable automatic face tracker reset. Use this if the target subject never leaves the frame.
- `--debugging`: Enable debug visualization and console output
- `--keep-audio`: Preserve original audio in output video

## Credits

- Project idea: [Divide-By-0](https://github.com/Divide-By-0/) and [MIT SOUL](http://soul.mit.edu/)
- Original deface project: [ORB-HD/deface](https://github.com/ORB-HD/deface)
- YOLO11: [Ultralytics](https://github.com/ultralytics/ultralytics)
- TorchReID: [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)