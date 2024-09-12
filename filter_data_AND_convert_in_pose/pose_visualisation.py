from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


with open("pose_videos/education-nationale/accompagner.pose", "rb") as f:
    pose = Pose.read(f.read())

v = PoseVisualizer(pose)

v.save_video("accompagner.mp4", v.draw())