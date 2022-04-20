
from moviepy.editor import VideoFileClip
from pipeline import PipeLine
import sys
mode=sys.argv[1]
path=sys.argv[2]
print(mode)
print(path)

# if not mode:
#     mode =1
# mode = "0"
# path = "videos/project_video.mp4"
input_fn = path
if mode == "0":
    video_name ="_debug_mode_out"
else:
    video_name = "_normal_out"    

tmp = input_fn.split('.')
tmp[0] += video_name
output_fn = '.'.join(tmp)

p = PipeLine()

video_input1 = VideoFileClip(input_fn)#.subclip(35, 42)
processed_video = video_input1.fl_image(p.run_Pipeline)

processed_video.write_videofile(output_fn, audio=False)


