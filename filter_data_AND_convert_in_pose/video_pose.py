import os

import cv2
from holistic import load_holistic
import argparse

import cv2

def load_video_frames(cap: cv2.VideoCapture, target_width: int, target_height: int):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to the target width and height
        resized_frame = cv2.resize(frame, (target_width, target_height))
        yield cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    cap.release()

def pose_video(input_path: str, output_path: str, format: str, target_width=640, target_height=480):
    # Load video frames
    print('Loading video ...')
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load and resize frames using load_video_frames generator
    frames = list(load_video_frames(cap, target_width, target_height))

    # Perform pose estimation
    print('Estimating pose ...')
    if format == 'mediapipe':
        pose = load_holistic(frames,
                             fps=fps,
                             width=target_width,
                             height=target_height,
                             progress=True,
                             additional_holistic_config={'model_complexity': 1})
    else:
        raise NotImplementedError('Pose format not supported')

    # Write to disk
    print('Saving to disk ...')
    with open(output_path, "wb") as f:
        pose.write(f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',
                        choices=['mediapipe'],
                        default='mediapipe',
                        type=str,
                        help='type of pose estimation to use')
    parser.add_argument('-i', required=True, type=str, help='path to input video file')
    parser.add_argument('-o', required=True, type=str, help='path to output pose file')

    args = parser.parse_args()

    if not os.path.exists(args.i):
        raise FileNotFoundError(f"Video file {args.i} not found")

    pose_video(args.i, args.o, args.format)

main()