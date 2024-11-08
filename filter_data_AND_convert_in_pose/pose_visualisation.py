from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
import argparse
import os

def visualisation(pose_file_name, video_mp4_name):
    with open(pose_file_name, "rb") as f:
        pose = Pose.read(f.read())

    for c in pose.header.components:
        print(c.name, c.points)
        print("---------------------------------------------------------------")
    v = PoseVisualizer(pose)

    v.save_video(video_mp4_name, v.draw())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='path to input pose file name')
    parser.add_argument('-o', required=True, type=str, help='path to output video mp4 name')

    args = parser.parse_args()

    if not os.path.exists(args.i):
        raise FileNotFoundError(f"Video file {args.i} not found")

    visualisation(args.i, args.o)

main()
