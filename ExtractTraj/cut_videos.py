
#!/usr/bin/env python
'''author: zhouhuayi'''
import os
import cv2
from config import video_info_dict

#####################################################
# video_id_str = "drawer_04"
# video_id_str = "pouring_05"
# video_id_str = "uncover_02"
# video_id_str = "unscrew_01"
# video_id_str = "openbox_01"

# video_id_str = "invert_01"
# video_id_str = "redirect_02"
# video_id_str = "reorient_01" 
# video_id_str = "dualpap_01" 
# video_id_str = "insertpen_01"
# video_id_str = "stacking_01"

# video_id_str = "pouring_06"
video_id_str = "unscrew_02"
#####################################################

for left_or_right in ["L", "R"]:

    if left_or_right == "L":
        source_video_path = os.path.join("./assets/cam_left/", video_info_dict[video_id_str]["cam_l_name"] + ".avi")
    if left_or_right == "R":
        source_video_path = os.path.join("./assets/cam_right/", video_info_dict[video_id_str]["cam_r_name"] + ".avi")
    print(source_video_path)
    
    save_frames_path = source_video_path[:-4]
    if not os.path.exists(save_frames_path):
        os.mkdir(save_frames_path)

    crop_list = []      # y0, y1, x0, x1

    capture = cv2.VideoCapture(source_video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[video basic info]: fps %.2f, frames %d, width %d, height %d."%(fps, frames, width, height))
    print("[video total time]: frames/fps = %d/%.2f = %d seconds"%(frames, fps, int(frames/fps)))

    duration = -1               # How many seconds cut a frame, -1 means saved all frames
    framenow = 0                # index for get the specific frame
    count = 0                   # index of frames having been saved
    chinese_path = True         # if is has chinese characters in path

    while(framenow < frames):
        # if count == 30: break
        capture.set(cv2.CAP_PROP_POS_FRAMES, framenow)
        ret, frame_image = capture.read()
        
        image_name = str(count).zfill(6)+'.jpg'
        save_path = os.path.join(save_frames_path, image_name)
        
        if len(crop_list) != 0:
            [y0, y1, x0, x1] = crop_list
            frame_image = frame_image[y0:y1, x0:x1]
        
        if chinese_path:
            cv2.imencode('.jpg', frame_image)[1].tofile(save_path)
        else:
            cv2.imwrite(save_path, frame_image)
        
        if duration != -1:
            framenow += duration * fps
        else:
            framenow += 1
            
        count += 1

    capture.release()