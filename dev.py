import torch
import torchvision

video_path = '/home/bjuncek/work/vision_bruno/test/assets/videos/SchoolRulesHowTheyHelpUs_wave_f_nm_np1_ba_med_0.avi'

reader = torch.classes.torchvision.Video(video_path, "video", True)
i = 0
while i<50:
    i += 1
    print(reader.next_tensor_usemove(""))
