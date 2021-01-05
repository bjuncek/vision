import torchvision, torch

vr = torchvision.io.VideoReader(
    "/home/bjuncek/work/vision_bruno/test/assets/videos/RATRACE_wave_f_nm_np1_fr_goo_37.avi"
)

frames = []
for data in vr:
    frames.append(data)

print(len(frames))