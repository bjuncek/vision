import torchvision, torch

vr = torchvision.io.VideoReader(
    "/home/bjuncek/work/vision_bruno/test/assets/videos/SOX5yA1l24A.mp4"
)

frames = []
for data in vr:
    frames.append(data)

print(len(frames))