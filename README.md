# Selective Face Anonymization for Lecture Videos

A modified version of [deface](https://github.com/ORB-HD/deface) that intelligently preserves the lecturer's face while anonymizing all other faces in educational videos. This project combines multiple state-of-the-art models to achieve robust face tracking and selective anonymization.

## Key Features

- **Smart Face Detection**: Uses CenterFace (from deface) for efficient face detection and blurring
- **Person Detection**: Employs YOLO11 to identify human figures in the frame
- **Person Re-identification**: Utilizes TorchReID to accurately identify and track the target person (lecturer)
- **Robust Face Tracking**: Implements OpenCV's CSRT tracker for consistent face tracking
- **Recovery Mechanisms**: Includes tracking recovery when the target face is temporarily lost
- **Batch Processing**: Supports processing multiple videos with their corresponding target person images
- **Partial Video Processing**: Allows processing specific segments of videos with different parameters

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

### Basic Arguments
- `input_dir`: Directory containing video folders
- `--video-filename`: Name of video files (default: `video.mp4`)
- `--target-person-dirname`: Name of target person directory (default: `target_person`)
- `--debugging`: Enable debug visualization and console output
- `--keep-audio`: Preserve original audio in output video

### Detection and Tracking Parameters
- `--thresh`: Face detection threshold (default: `0.4`)
  - Higher values reduce false positives but may miss faces
  - Lower values catch more faces but may detect non-faces
- `--reid-threshold`: Similarity threshold for person re-identification (default: `0.7`)
  - Higher values (e.g., 0.8) ensure stricter matching but may lose tracking
  - Lower values (e.g., 0.6) maintain tracking better but may match wrong people
- `--max-frames-without-faces`: Frames to continue tracking without detection (default: `30`)
  - Higher values maintain tracking through occlusions
  - Lower values reset tracking more quickly when target is lost
- `--disable-tracker-reset`: Disable automatic face tracker reset

### Partial Video Processing
Different parts of a video may require different parameter settings for optimal results. The script supports processing specific segments:

```bash
python deface/main.py /path/to/input_directory \
--debug-start 120 \  # Start at 2 minutes
--debug-duration 30 \  # Process 30 seconds
--thresh 0.3 \  # Lower threshold for this segment
--reid-threshold 0.65  # Adjusted ReID threshold
```

This feature can be integrated with [multi-cam_video_editor](https://github.com/mitsoul/multi-cam_video_editor) to:
1. Split problematic videos into segments
2. Process each segment with optimized parameters
3. Recombine the processed segments


## Credits

- Project idea: [Divide-By-0](https://github.com/Divide-By-0/) and [MIT SOUL](http://soul.mit.edu/)
- Original deface project: [ORB-HD/deface](https://github.com/ORB-HD/deface)
- YOLO11: [Ultralytics](https://github.com/ultralytics/ultralytics)
- TorchReID: [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)