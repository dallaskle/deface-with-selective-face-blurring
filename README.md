
# `deface` with Selective Face Blurring for Lecture Videos

This project is a modified version of the [deface](https://github.com/ORB-HD/deface) repository, designed specifically for treating lecture videos to protect student privacy while keeping the professor's face visible.

## Features
- **Selective Face Blurring:** At the beginning of a video, the professor's face can be selected to remain unblurred while all other faces are blurred throughout the lecture.
- **Privacy Preservation:** Helps anonymize student faces in educational videos, ideal for online lecture recordings.

## Usage
The CLI usage remains the same as the original Deface repository. For details on how to run the tool, refer to the [Deface documentation](https://github.com/ORB-HD/deface).

## Example Command
`python deface/main.py /path/to/lecture.mp4`

In this modified version, a face is selected at the start of the video for exclusion from the blurring process.

## Other Experiments
- Tested [face_recognition](https://github.com/ageitgey/face_recognition), [EgoBlur](https://github.com/facebookresearch/EgoBlur/tree/main), OpenCV Multi-Object Tracking API and Youtube Studio (in-built blur) to blur student faces. All fell short of providing a high accuracy consistent blur.


## Credit
Project idea from [Divide-By-0](https://github.com/Divide-By-0/) and [MIT SOUL](http://soul.mit.edu/)
