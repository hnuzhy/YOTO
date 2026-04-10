
import os
import cv2

if __name__ == "__main__":

    frames_dirs = {
        # "Video_20241128090342905_drawer",
        # "Video_20241125112516999_pouring",
        # "Video_20241122141747978_unscrew",
        # "Video_20241120181731061_uncover",
        # "Video_20241122163050766_openbox",
        
        # "Video_20250122193711380_redirect",
        # "Video_20250122150737697_invert",
        
        # "Video_20250220144445293_reorient",
        # "Video_20250225123729086_insertpen",
        # "Video_20250304144113965_stacking",
        
        "Video_20250530194005505_unscrew",
        "Video_20250530194413211_pouring",
    }
    
    fps, width, height = 25, 1280, 1024
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for frames_dir in frames_dirs:
        frames_name_list = os.listdir(frames_dir)
        frames_name_list.sort()
        
        output_path = f"./{frames_dir}_slimed.mp4"
        print("output_path:", output_path)
        vout = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frames_name in frames_name_list:
            frame = cv2.imread(os.path.join(frames_dir, frames_name))
            vout.write(frame)
        vout.release()
        print("finished !", frames_dir)
