from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


with open("achat.pose", "rb") as f:
    pose = Pose.read(f.read())

for c in pose.header.components:
    print(c.name, c.points)
    print("---------------------------------------------------------------")
v = PoseVisualizer(pose)

v.save_video("achat.mp4", v.draw())
