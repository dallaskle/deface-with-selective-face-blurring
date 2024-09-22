
# Deface with Selective Face Blurring for Lecture Videos

This project is a modified version of the [Deface](https://github.com/ox-vgg/deface) repository, designed specifically for treating lecture videos to protect student privacy while keeping the professor's face visible.

## Features
- **Selective Face Blurring:** At the beginning of a video, the professor's face can be selected to remain unblurred while all other faces are blurred throughout the lecture.
- **Privacy Preservation:** Helps anonymize student faces in educational videos, ideal for online lecture recordings.

## Usage
The CLI usage remains the same as the original Deface repository. For details on how to run the tool, refer to the [Deface documentation](https://github.com/ox-vgg/deface).

## Example Command
\`\`\`bash
python deface.py --input /path/to/lecture.mp4 --output /path/to/output.mp4
\`\`\`

In this modified version, a face is selected at the start of the video for exclusion from the blurring process.
